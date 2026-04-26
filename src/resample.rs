use anyhow::Result;
use ndarray::{Array2, Axis, concatenate, s};
use std::{collections::HashMap, path::PathBuf};
use tracing::info;
use crate::{
    audio::{post_process::{loudness_norm, pre_emphasis_base_tension}, read_audio, write_audio},
    consts::{FEATURE_EXT, HIFI_CONFIG, HOP_SIZE, ORIGIN_HOP_SIZE, SAMPLE_RATE},
    model::{get_remover, get_vocoder},
    utils::{
        cache::{CACHE_MANAGER, Features}, dynamic_range_compression, growl::growl,
        interp::{akima, interp1d}, mel::mel, midi_to_hz,
        parser::{flag_parser, pitch_parser, pitch_string_to_cents, tempo_parser}, reflect_pad_2d
    },
};
const SR: f32 = SAMPLE_RATE as f32;
const THOP_ORIGIN: f32 = ORIGIN_HOP_SIZE as f32 / SR;
const THOP_ORIGIN_HALF: f32 = THOP_ORIGIN / 2.0;
const THOP: f32 = HOP_SIZE as f32 / SR;
pub struct Resampler {
    in_file: PathBuf,
    out_file: PathBuf,
    pitch: f32,
    velocity: f32,
    flags: HashMap<String, Option<f32>>,
    offset: f32,
    length: f32,
    consonant: f32,
    cutoff: f32,
    volume: f32,
    modulation: f32,
    tempo: f32,
    pitchbend: Vec<f32>,
}
impl Resampler {
    pub fn new(args: Vec<String>) -> Result<()> {
        Self {
            in_file: PathBuf::from(&args[0]),
            out_file: PathBuf::from(&args[1]),
            pitch: pitch_parser(&args[2])? as f32,
            velocity: args[3].parse::<f32>()? / 100.0,
            flags: flag_parser(&args[4])?,
            offset: args[5].parse::<f32>()? / 1000.0,
            length: args[6].parse::<f32>()? / 1000.0,
            consonant: args[7].parse::<f32>()? / 1000.0,
            cutoff: args[8].parse::<f32>()? / 1000.0,
            volume: args[9].parse::<f32>()? / 100.0,
            modulation: args[10].parse::<f32>()? / 100.0,
            tempo: tempo_parser(&args[11])? * 96.0,
            pitchbend: pitch_string_to_cents(&args[12])?,
        }.resample()
    }
    fn get_features(&mut self) -> Result<Features> {
        [("Hb", 100.), ("Hv", 100.), ("Ht", 0.), ("g", 0.)].iter()
            .for_each(|(k, v)| { self.flags.entry(k.to_string()).or_insert(Some(*v)); });
        let features_path = self.in_file.with_file_name(format!(
            "{}_{}{}",
            self.in_file.file_stem().unwrap().to_str().unwrap(),
            self.flags.iter()
                .filter(|(k, _)| ["Hb","Hv","Ht","g"].contains(&k.as_str()))
                .map(|(k, v)| format!("{}{}", k, v.as_ref().unwrap()))
                .collect::<Vec<_>>()
                .join("_"),
            FEATURE_EXT
        ));
        if let Some(features) = CACHE_MANAGER.load_features_cache(&features_path, self.flags.contains_key("G")) {
            return Ok(features);
        }
        info!("Generating features (cache not found or forced): {}", features_path.display());
        let (bre, voicing, tension, gender) = (
            self.flags["Hb"].unwrap(),
            self.flags["Hv"].unwrap(),
            self.flags["Ht"].unwrap(),
            self.flags["g"].unwrap()
        );
        info!("Breath: {}, Voicing: {}, Tension: {}", bre, voicing, tension);
        let mut wave = read_audio(&self.in_file)?;
        info!("Wave length: {}", wave.len());
        let (bre_scale, voicing_scale) = (bre.clamp(0.,500.)/100., voicing.clamp(0.,150.)/100.);
        if tension != 0. || bre != voicing {
            info!("Applying HNSEP separation for breath/voicing/tension adjustment");
            let hnsep_path = self.in_file.with_file_name(format!(
                "{}_hnsep",
                self.in_file.file_stem().unwrap().to_str().unwrap()
            ));
            let seg_output = CACHE_MANAGER.load_hnsep_cache(&hnsep_path, self.flags.contains_key("G")).unwrap_or_else(|| {
                info!("Generating HNSEP features: {}", hnsep_path.display());
                let seg = get_remover().lock().unwrap().run(&wave);
                let _ = CACHE_MANAGER.save_hnsep_cache(&hnsep_path, seg);
                CACHE_MANAGER.load_hnsep_cache(&hnsep_path, false).unwrap()
            });
            if tension != 0. {
                let mut voicing_seg = seg_output.iter().map(|&s| voicing_scale * s).collect();
                pre_emphasis_base_tension(&mut voicing_seg, -tension.clamp(-100.0, 100.0));
                wave.iter_mut().zip(&seg_output).zip(&voicing_seg).for_each(|((w, &s), &em)| *w = bre_scale * (*w - s) + em);
            } else {
                wave.iter_mut().zip(&seg_output).for_each(|(w, &s)| *w = bre_scale * (*w - s) + voicing_scale * s);
            }
        } else if bre != 100. || voicing != 100. {
            info!("Applying simple volume scaling: {}", bre / 100.);
            wave.iter_mut().for_each(|x| *x *= bre_scale);
        }
        let wave_max = wave.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let scale = if wave_max >= 0.5 {
            info!("Scaling audio to max 0.5 (current: {:.3})", wave_max);
            let s = 0.5 / wave_max;
            wave.iter_mut().for_each(|x| *x *= s);
            s
        } else {
            info!("Audio volume acceptable (max: {:.3})", wave_max);
            1.0
        };
        let mut mel_origin = mel(&mut wave, gender.clamp(-600.,600.) / 100., 1.);
        info!("Gender adjustment: {}, Mel shape: {:?}", gender, mel_origin.dim());
        dynamic_range_compression(&mut mel_origin);
        let features = Features { mel_origin, scale };
        let _ = CACHE_MANAGER.save_features_cache(&features_path, &features);
        Ok(features)
    }
    fn resample(&mut self) -> Result<()> {
        let features = self.get_features()?;
        if self.out_file.file_name().and_then(|s| s.to_str()) == Some("nul") {
            info!("Null output file - skipping write");
            return Ok(());
        }
        let (mut mel_origin, scale) = (features.mel_origin, features.scale);
        info!("Modulation: {:.1}, Scale: {:.1}, Mel shape: {:?}", self.modulation, scale, mel_origin.dim());
        let mel_cols = mel_origin.ncols();
        let mut t_origin: Vec<f32> = (0..mel_cols).map(|i| i as f32 + 0.5).collect();
        let mut t_total = t_origin.last().unwrap() * THOP_ORIGIN + THOP_ORIGIN_HALF;
        let vel = 2.0f32.powf(1.0 - self.velocity);
        let (start, cutoff) = (self.offset, self.cutoff);
        let end = if cutoff < 0.0 { start - cutoff } else { t_total - cutoff };
        let (con, length_req) = (start + self.consonant, self.length);
        let mut stretch_len = end - con;
        info!("Time params: start={:.4}, end={:.4}, con={:.4}, stretch_len={:.4}, length_req={:.4}", start, end, con, stretch_len, length_req);
        if HIFI_CONFIG.loop_mode || self.flags.contains_key("He") {
            info!("Enabling loop mode");
            let start_idx = ((con / THOP_ORIGIN + 0.5).floor() as usize).clamp(0, mel_cols);
            let end_idx = ((end / THOP_ORIGIN + 0.5).floor() as usize).clamp(0, mel_cols);
            let pad_size = (length_req / THOP_ORIGIN).floor() as usize + 1;
            let padded_mel = reflect_pad_2d(mel_origin.slice(s![.., start_idx..end_idx]), pad_size);
            mel_origin = concatenate![Axis(1), mel_origin.slice(s![.., 0..start_idx]), padded_mel];
            stretch_len = pad_size as f32 * THOP_ORIGIN;
            t_origin = (0..mel_origin.ncols()).map(|i| i as f32 + 0.5).collect();
            t_total = t_origin.last().unwrap() * THOP_ORIGIN + THOP_ORIGIN_HALF;
        }
        let scal_ratio = if stretch_len < length_req {
            info!("Stretching (ratio: {:.4})", length_req / stretch_len);
            length_req / stretch_len
        } else {
            info!("No stretching needed (ratio: 1.0)");
            1.0
        };
        let stretch = |t: f32| if t < vel * con { t / vel } else { con + (t - vel * con) / scal_ratio };
        let stretched_frames = ((con * vel + (t_total - con) * scal_ratio) / THOP).floor() as usize + 1;
        let mut stretched_mel: Vec<f32> = (0..stretched_frames).map(|i| i as f32 + 0.5).collect();
        let slice_start = ((start * vel / THOP + 0.5).floor() as usize).saturating_sub(HIFI_CONFIG.fill);
        let slice_end = (((length_req + con * vel) / THOP + 0.5).floor() as usize).saturating_add(HIFI_CONFIG.fill).clamp(0, stretched_frames);
        stretched_mel.truncate(slice_end);
        if slice_start > 0 { stretched_mel.drain(0..slice_start); }
        stretched_mel.iter_mut().for_each(|t| *t = stretch(*t));
        let t_area_origin: Vec<f32> = t_origin.iter().map(|&x| x * THOP_ORIGIN).collect();
        stretched_mel.iter_mut().for_each(|t| *t = t.clamp(0.0, *t_area_origin.last().unwrap()));
        info!("Stretched time axis length: {}", stretched_mel.len());
        let mel_render = interp1d(&t_area_origin, &mel_origin, &stretched_mel);
        let t: Vec<f32> = (0..mel_render.ncols()).map(|i| i as f32 * THOP).collect();
        info!("Render mel shape: {:?}, Processing pitch...", mel_render.dim());
        let mut pitch: Vec<f32> = self.pitchbend.iter().map(|&pb| pb / 100.0 + self.pitch).collect();
        if let Some(&t_flag) = self.flags.get("t").and_then(|x| x.as_ref()) {
            pitch.iter_mut().for_each(|p| *p += t_flag / 100.0);
        }
        let t_pitch: Vec<f32> = (0..pitch.len()).map(|i| 60.0 * i as f32 / self.tempo + start * vel - slice_start as f32 * THOP).collect();
        let pitch_render = akima(&t_pitch, &pitch, &t);
        let f0_render: Vec<f32> = pitch_render.iter().map(|&x| midi_to_hz(x)).collect();
        info!("F0 render length: {}", f0_render.len());
        let wav_con = get_vocoder().lock().unwrap().run(mel_render, f0_render);
        let (new_start, new_end) = (start * vel - slice_start as f32 * THOP, (length_req + con * vel) - slice_start as f32 * THOP);
        let start_sample = (new_start * SR).max(0.0) as usize;
        let end_sample = (new_end * SR).min(wav_con.len() as f32) as usize;
        let mut render = wav_con[start_sample..end_sample].to_vec();
        info!("Vocoder output length: {}, Cropped audio length: {}", wav_con.len(), render.len());
        if let Some(&a_flag) = self.flags.get("A").and_then(|x| x.as_ref()) {
            let a_clamped = a_flag.clamp(-100.0, 100.0);
            info!("Applying amplitude modulation (A={:.1})", a_clamped);
            let mut pitch_derivative = vec![0.0; pitch_render.len()];
            pitch_derivative[0] = (pitch_render[1] - pitch_render[0]) / (t[1] - t[0]);
            for i in 1..pitch_render.len()-1 {
                pitch_derivative[i] = (pitch_render[i+1] - pitch_render[i-1]) / (t[i+1] - t[i-1]);
            }
            pitch_derivative[pitch_render.len()-1] = (pitch_render[pitch_render.len()-1] - pitch_render[pitch_render.len()-2]) / (t[pitch_render.len()-1] - t[pitch_render.len()-2]);
            let gain_at_mel_frames: Vec<f32> = pitch_derivative.iter().map(|&d| 5.0f32.powf(1e-4 * a_clamped * d)).collect();
            let audio_time: Vec<f32> = (0..render.len()).map(|i| new_start + (new_end - new_start) * i as f32 / render.len() as f32).collect();
            let interpolated_gain = interp1d(&t, &Array2::from_shape_vec((1, gain_at_mel_frames.len()), gain_at_mel_frames).unwrap(), &audio_time);
            render.iter_mut().zip(interpolated_gain.row(0)).for_each(|(s, &g)| *s *= g);
            info!("Amplitude modulation applied");
        }
        render.iter_mut().for_each(|x| *x /= scale);
        let new_max = render.iter().map(|x| x.abs()).fold(0.0, f32::max);
        if let Some(&hg) = self.flags.get("HG").and_then(|x| x.as_ref()) {
            info!("Applying growl (strength: {:.1})", hg);
            growl(&mut render, SR, 80.0, hg.clamp(-100.0, 100.0) / 100.0);
        }
        if HIFI_CONFIG.wave_norm {
            if let Some(&p_strength) = self.flags.get("P").and_then(|x| x.as_ref()) {
                loudness_norm(&mut render, SR, -16.0, p_strength.clamp(-100.0, 100.0) as u8);
            }
        }
        if new_max > HIFI_CONFIG.peak_limit { render.iter_mut().for_each(|x| *x /= new_max); }
        render.iter_mut().for_each(|x| *x *= self.volume);
        write_audio(&self.out_file, &render)?;
        info!("Successfully processed: {} -> {}", self.in_file.display(), self.out_file.display());
        Ok(())
    }
}