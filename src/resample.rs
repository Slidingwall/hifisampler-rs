use anyhow::Result;
use ndarray::{Array2, Axis, concatenate, s};
use std::{collections::HashMap, path::PathBuf};
use tracing::info;
use crate::{
    audio::{post_process::{loudness_norm, pre_emphasis_base_tension}, read_audio, write_audio},
    consts::{SAMPLE_RATE, ORIGIN_HOP_SIZE, HOP_SIZE, FEATURE_EXT, HIFI_CONFIG},
    model::{get_remover, get_vocoder},
    utils::{
        cache::{CACHE_MANAGER, Features}, dynamic_range_compression, growl::growl, interp::Akima, interp1d, 
        midi_to_hz, mel::mel, parser::{flag_parser, pitch_parser, pitch_string_to_cents, tempo_parser}, reflect_pad_2d
    },
};
const SR_F64: f64 = SAMPLE_RATE as f64;
const THOP_ORIGIN: f64 = ORIGIN_HOP_SIZE as f64 / SR_F64;
const THOP_ORIGIN_HALF: f64 = THOP_ORIGIN / 2.0;
const THOP: f64 = HOP_SIZE as f64 / SR_F64;
const THOP_HALF: f64 = THOP / 2.0;
pub struct Resampler {
    in_file: PathBuf,
    out_file: PathBuf,
    pitch: f64,
    velocity: f64,
    flags: HashMap<String, Option<f64>>,
    offset: f64,
    length: f64,
    consonant: f64,
    cutoff: f64,
    volume: f64,
    modulation: f64,
    tempo: f64,
    pitchbend: Vec<f64>,
}
impl Resampler {
    pub fn new(args: Vec<String>) -> Result<()> {
        Self {
            in_file: PathBuf::from(args[0].to_string()),
            out_file: PathBuf::from(args[1].to_string()),
            pitch: pitch_parser(&args[2])? as f64,
            velocity: args[3].parse::<f64>()? / 100.,
            flags: flag_parser(&args[4])?,
            offset: args[5].parse::<f64>()? / 1000.,
            length: args[6].parse::<f64>()? / 1000.,
            consonant: args[7].parse::<f64>()? / 1000.,
            cutoff: args[8].parse::<f64>()? / 1000.,
            volume: args[9].parse::<f64>()? / 100.,
            modulation: args[10].parse::<f64>()? / 100.,
            tempo: tempo_parser(&args[11])? * 96.,
            pitchbend: pitch_string_to_cents(&args[12])?,
        }.render()
    }
    fn render(&mut self) -> Result<()> {
        let mut features = self.get_features()?;
        self.resample(&mut features)
    }
    fn get_features(&mut self) -> Result<Features> {
        [("Hb", 100.), ("Hv", 100.), ("Ht", 0.), ("g", 0.)]
            .iter()
            .for_each(|(k, v)| { self.flags.entry(k.to_string()).or_insert(Some(*v)); });
        let flag_suf = self.flags.iter()
            .filter(|(k, _)| ["Hb", "Hv", "Ht", "g"].contains(&k.as_str()))
            .map(|(k, v)| format!("{}{}", k, v.as_ref().unwrap())) 
            .collect::<Vec<_>>()
            .join("_");
        let stem = self.in_file.file_stem().unwrap().to_str().unwrap();
        let cache_name = format!("{}_{}{}", stem, flag_suf, FEATURE_EXT);
        let features_path = self.in_file.with_file_name(cache_name);
        let force_gen = self.flags.contains_key("G");
        if let Some(features) = CACHE_MANAGER.load_features_cache(&features_path, force_gen) {
            return Ok(features);
        }
        info!("Generating features (cache not found or forced): {}", features_path.display());
        let features = self.generate_features()?;
        CACHE_MANAGER.save_features_cache(&features_path, &features);
        Ok(features)
    }
    fn generate_features(&self) -> Result<Features> {
        let bre = self.flags.get("Hb").and_then(|o| o.as_ref()).copied().unwrap();
        let voicing = self.flags.get("Hv").and_then(|o| o.as_ref()).copied().unwrap();
        let tension = self.flags.get("Ht").and_then(|o| o.as_ref()).copied().unwrap();
        info!("Breath: {}, Voicing: {}, Tension: {}", bre, voicing, tension);
        let mut wave = read_audio(&self.in_file)?;
        info!("Wave length: {}", wave.len());
        if tension != 0. || bre != voicing {
            info!("Applying HNSEP separation for breath/voicing/tension adjustment");
            let stem = self.in_file.file_stem().unwrap().to_str().unwrap();
            let hnsep_path = self.in_file.with_file_name(format!("{}_hnsep", stem));
            let force_gen = self.flags.contains_key("G");
            let seg_output = if !force_gen && hnsep_path.exists() {
                CACHE_MANAGER.load_hnsep_cache(&hnsep_path, force_gen).unwrap()
            } else {
                info!("Generating HNSEP features: {}", hnsep_path.display());
                let remover_arc = get_remover();
                let mut remover = remover_arc.lock().unwrap();
                let seg = remover.run(&wave);
                CACHE_MANAGER.save_hnsep_cache(&hnsep_path, seg).unwrap()
            };
            let (bre_scale, voicing_scale) = (bre.clamp(0., 500.) / 100., voicing.clamp(0., 150.) / 100.);
            if tension != 0. {
                let mut voicing_seg = seg_output.iter()
                    .map(|&s| voicing_scale * s)
                    .collect::<Vec<f64>>();
                pre_emphasis_base_tension(&mut voicing_seg, -tension.clamp(-100., 100.) / 50.);
                wave.iter_mut()
                    .zip(seg_output.iter())
                    .zip(voicing_seg.iter())
                    .for_each(|((w, &s), &em)| {
                        *w = bre_scale * (*w - s) + em;
                    });
            } else {
                wave.iter_mut()
                    .zip(seg_output.iter())
                    .for_each(|(w, &s)| {
                        *w = bre_scale * (*w - s) + voicing_scale * s;
                    });
            };
        } else if bre != 100. || voicing != 100. {
            info!("Applying simple volume scaling: {}", bre / 100.);
            let bre_scale = bre.clamp(0., 500.) / 100.; 
            wave.iter_mut().for_each(|x| *x *= bre_scale);
        }
        let wave_max = wave.iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let scale = if wave_max >= 0.5 {
            info!("Scaling audio to max 0.5 (current: {:.3})", wave_max);
            let s = 0.5 / wave_max;
            wave.iter_mut().for_each(|x| *x *= s);
            s
        } else {
            info!("Audio volume acceptable (max: {:.3})", wave_max);
            1.0
        };
        let gender = self.flags.get("g").and_then(|o| o.as_ref()).copied().unwrap().clamp(-600., 600.);
        info!("Gender adjustment: {}", gender);
        let mut mel_origin = mel(&mut wave, gender / 100., 1.);
        info!("Mel shape: {:?}", mel_origin.dim());
        dynamic_range_compression(&mut mel_origin);
        Ok(Features { mel_origin, scale })
    }
    fn resample(&self, features: &mut Features) -> Result<()> {
        if self.out_file.file_name().and_then(|s| s.to_str()) == Some("nul") {
            info!("Null output file - skipping write");
            return Ok(());
        }
        let mel_origin = &mut features.mel_origin;
        info!(
            "Modulation: {:.1}, Scale: {:.1}, Mel shape: {:?}",
            self.modulation, features.scale, mel_origin.dim()
        );
        let mel_cols = mel_origin.ncols();
        let mut t_origin = Vec::with_capacity(mel_cols);
        for i in 0..mel_cols {
            let val = i as f64 * THOP_ORIGIN + THOP_ORIGIN_HALF;
            t_origin.push(val);
        }
        let mut t_total = t_origin.last().copied().unwrap() + THOP_ORIGIN_HALF;
        let vel = (1.0 - self.velocity).exp2();
        let start = self.offset;
        let cutoff = self.cutoff;
        let end = if cutoff < 0.0 { start - cutoff } else { t_total - cutoff };
        let con = start + self.consonant;
        let length_req = self.length;
        let mut stretch_len = end - con;
        info!(
            "Time params: start={:.4}, end={:.4}, con={:.4}, stretch_len={:.4}, length_req={:.4}",
            start, end, con, stretch_len, length_req
        );
        if HIFI_CONFIG.loop_mode || self.flags.contains_key("He") {
            info!("Enabling loop mode");
            let start_idx = (((con + THOP_ORIGIN_HALF) / THOP_ORIGIN).floor() as usize).clamp(0, mel_cols);
            let end_idx = (((end + THOP_ORIGIN_HALF) / THOP_ORIGIN).floor() as usize).clamp(start_idx, mel_cols);
            let mel_loop = mel_origin.slice(s![.., start_idx..end_idx]);
            let pad_size = (length_req / THOP_ORIGIN).floor() as usize + 1;
            let padded_mel = reflect_pad_2d(mel_loop, pad_size);
            *mel_origin = concatenate![Axis(1), mel_origin.slice(s![.., 0..start_idx]), padded_mel];
            stretch_len = pad_size as f64 * THOP_ORIGIN;
            t_origin = Vec::with_capacity(mel_origin.ncols()); 
            for i in 0..mel_origin.ncols() {
                let val = i as f64 * THOP_ORIGIN + THOP_ORIGIN_HALF;
                t_origin.push(val);
            }
            t_total = t_origin.last().copied().unwrap() + THOP_ORIGIN_HALF;
            info!("Looped mel shape: {:?}, new total time: {:.4}", mel_origin.dim(), t_total);
        }
        let scal_ratio = if stretch_len < length_req {
            info!("Stretching (ratio: {:.4})", length_req / stretch_len);
            length_req / stretch_len
        } else {
            info!("No stretching needed (ratio: 1.0)");
            1.0
        };
        let stretch = |t: f64| -> f64 {
            if t < vel * con { t / vel } else { con + (t - vel * con) / scal_ratio }
        };
        let stretched_frames = ((con * vel + (t_total - con) * scal_ratio) / THOP)
            .floor() as usize + 1;
        let mut stretched_mel = Vec::with_capacity(stretched_frames);
        for i in 0..stretched_frames {
            let val = i as f64 * THOP + THOP_HALF;
            stretched_mel.push(val);
        }
        let slice_start = (((start * vel + THOP_HALF) / THOP).floor() as usize)
            .saturating_sub(HIFI_CONFIG.fill);
        let slice_end = stretched_frames.saturating_sub(
            stretched_frames.saturating_sub(
                ((length_req + con * vel + THOP_HALF) / THOP).floor() as usize
            ).saturating_sub(HIFI_CONFIG.fill)
        );
        stretched_mel = stretched_mel[slice_start..slice_end].to_vec();
        info!("Stretched time axis length: {}", stretched_mel.len());
        stretched_mel.iter_mut().for_each(|t| {
            *t = stretch(*t).clamp(0.0, t_origin.last().copied().unwrap());
        });
        let mel_render = interp1d(&t_origin, &mel_origin, &stretched_mel);
        info!("Render mel shape: {:?}, Processing pitch...", mel_render.dim());
        let mut pitch_base = Vec::with_capacity(self.pitchbend.len());
        for &pb in &self.pitchbend {
            let base = pb + self.pitch;
            let val = self.flags.get("t")
                .and_then(|o| o.as_ref())
                .map_or(base, |&t| base + t.clamp(-1200., 1200.) / 100.0);
            pitch_base.push(val);
        }
        let new_start = start * vel - slice_start as f64 * THOP;
        let new_end = (con * vel + length_req) - slice_start as f64 * THOP;
        let mut t = Vec::with_capacity(mel_render.ncols());
        for i in 0..mel_render.ncols() {
            let val = i as f64 * THOP;
            t.push(val);
        }
        let t_scale = (self.pitchbend.len() as f64 - 1.) / (mel_render.ncols() as f64 * THOP);
        let pitch_render = Akima::new(&pitch_base)
            .sample_with_slice(&t.iter()
                .map(|&x| x.clamp(0., mel_render.ncols() as f64 * THOP) * t_scale)
                .collect::<Vec<_>>());
        let mut f0_render = Vec::with_capacity(pitch_render.len());
        for &x in &pitch_render {
            f0_render.push(midi_to_hz(x));
        }
        info!("F0 render length: {}", f0_render.len());
        let mut render = {
            let vocoder_arc = get_vocoder();
            let mut vocoder = vocoder_arc.lock().unwrap();
            let mut wav_con = vocoder.run(mel_render, &f0_render);
            info!("Vocoder output length: {}", wav_con.len());
            let (start_idx, end_idx) = (
                (new_start * SR_F64).floor() as usize,
                (new_end * SR_F64).floor() as usize,
            );
            let (start_idx, end_idx) = (
                start_idx.clamp(0, wav_con.len()),
                end_idx.clamp(start_idx, wav_con.len()),
            );
            if start_idx < end_idx {
                wav_con.truncate(end_idx);
                wav_con.drain(0..start_idx);
                wav_con
            } else {
                Vec::new()
            }
        };
        let render_len = render.len();
        info!("Cropped audio length: {}", render_len);
        if let Some(&a_flag) = self.flags.get("A").and_then(|o| o.as_ref()).filter(|&&a| a != 0.0) {
            info!("Applying amplitude modulation (A={:.1})", a_flag);
            let mut gain_data = Vec::with_capacity(pitch_render.len());
            for i in 0..pitch_render.len() {
                let grad = match i {
                    0 => (pitch_render[1] - pitch_render[0]) / (t[1] - t[0] + 1e-9),
                    i if i == pitch_render.len() - 1 => (pitch_render[i] - pitch_render[i-1]) / (t[i] - t[i-1] + 1e-9),
                    _ => (pitch_render[i+1] - pitch_render[i-1]) / (t[i+1] - t[i-1] + 1e-9),
                };
                gain_data.push(5.0f64.powf(1e-4 * a_flag.clamp(-100.0, 100.0) * grad));
            }
            let mut audio_time = Vec::with_capacity(render_len);
            for i in 0..render_len {
                let val = new_start + (new_end - new_start) / render_len as f64 * i as f64;
                audio_time.push(val);
            }
            render.iter_mut()
                .zip(interp1d(
                    &t,
                    &Array2::from_shape_vec((1, gain_data.len()), gain_data).unwrap(),
                    &audio_time
                ).row(0).iter())
                .for_each(|(r, g)| *r *= g);
            info!("Amplitude modulation applied");
        }
        render.iter_mut().for_each(|x| *x /= features.scale);
        let max = render.iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        if let Some(&hg) = self.flags.get("HG").and_then(|o| o.as_ref()) {
            info!("Applying growl (strength: {:.1})", hg);
            growl(&mut render, SR_F64, 80.0, hg.clamp(0.0, 100.0) / 100.0);
        }
        if HIFI_CONFIG.wave_norm {
            let p_strength = self.flags.get("P")
                .and_then(|o| o.as_ref())
                .copied()
                .unwrap_or(100.0)
                .clamp(0.0, 100.0) as u8; 
            loudness_norm(&mut render, SR_F64,  -16.0, p_strength);
        }
        if max > HIFI_CONFIG.peak_limit {
            render.iter_mut().for_each(|x| *x *= self.volume / max);
        } else {
            render.iter_mut().for_each(|x| *x *= self.volume);
        }
        write_audio(&self.out_file, &render)?;
        info!("Successfully processed: {} -> {}", self.in_file.display(), self.out_file.display());
        Ok(())
    }
}