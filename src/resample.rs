use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Axis};
use std::{
    collections::HashMap,
    path::PathBuf,
};
use tracing::info;
use crate::{
    audio::{
        post_process::{loudness_norm, pre_emphasis_base_tension},
        read_audio, write_audio,
    },
    consts::{self, HIFI_CONFIG},
    model::{get_mel_analyzer, get_remover, get_vocoder},
    utils::{
        cache::{CACHE_MANAGER, Features}, dynamic_range_compression, growl::growl, interp::Akima, interp1d, midi_to_hz, parser::{pitch_parser, pitch_string_to_midi, tempo_parser, flag_parser}, reflect_pad_2d,
    },
};
const SR_F64: f64 = consts::SAMPLE_RATE as f64;
const THOP_ORIGIN: f64 = consts::ORIGIN_HOP_SIZE as f64 / SR_F64;
const THOP_ORIGIN_HALF: f64 = THOP_ORIGIN / 2.0;
const THOP: f64 = consts::HOP_SIZE as f64 / SR_F64;
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
            pitchbend: pitch_string_to_midi(&args[12])?,
        }.render()
    }
    fn render(&mut self) -> Result<()> {
        let features = self.get_features()?;
        self.resample(features)
    }
    fn get_features(&mut self) -> Result<Features> {
        [("Hb", 100.), ("Hv", 100.), ("Ht", 0.), ("g", 0.)]
            .iter()
            .for_each(|(k, v)| { self.flags.entry(k.to_string()).or_insert(Some(*v)); });
        let flag_suffix = self.flags.iter()
            .filter(|(k, _)| ["Hb", "Hv", "Ht", "g"].contains(&k.as_str()))
            .map(|(k, v)| format!("{}{}", k, v.as_ref().unwrap())) 
            .collect::<Vec<_>>()
            .join("_");
        let stem = self.in_file.file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid input file name (no stem)"))?;
        let cache_name = if flag_suffix.is_empty() {
            format!("{}{}", stem, consts::FEATURE_EXT)
        } else {
            format!("{}_{}{}", stem, flag_suffix, consts::FEATURE_EXT)
        };
        let features_path = self.in_file.with_file_name(cache_name);
        let force_generate = self.flags.contains_key("G");
        if let Some(features) = CACHE_MANAGER.load_features_cache(&features_path, force_generate)? {
            return Ok(features);
        }
        info!("Generating features (cache not found or forced): {}", features_path.display());
        let features = self.generate_features()?;
        CACHE_MANAGER.save_features_cache(&features_path, &features)?
            .ok_or_else(|| anyhow!("Failed to save features to {}", features_path.display()))?;
        Ok(features)
    }
    fn generate_features(&self) -> Result<Features> {
        let breath = self.flags.get("Hb").and_then(|o| o.as_ref()).copied().unwrap_or(100.);
        let voicing = self.flags.get("Hv").and_then(|o| o.as_ref()).copied().unwrap_or(100.);
        let tension = self.flags.get("Ht").and_then(|o| o.as_ref()).copied().unwrap_or(0.);
        info!("Breath: {}, Voicing: {}, Tension: {}", breath, voicing, tension);
        let mut wave = read_audio(&self.in_file)?;
        info!("Wave length: {}", wave.len());
        if tension != 0. || breath != voicing {
            info!("Applying HNSEP separation for breath/voicing/tension adjustment");
            let stem = self.in_file.file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow!("Invalid file stem: {}", self.in_file.display()))?;
            let hnsep_path = self.in_file.with_file_name(format!("{}_hnsep", stem));
            let force_generate = self.flags.contains_key("G");
            let seg_output = if !force_generate && hnsep_path.exists() {
                CACHE_MANAGER.load_hnsep_cache(&hnsep_path, force_generate)?
                    .ok_or_else(|| anyhow!("Invalid HNSEP cache: {}", hnsep_path.display()))?
            } else {
                info!("Generating HNSEP features: {}", hnsep_path.display());
                let remover_arc = get_remover()?;
                let mut remover = remover_arc.lock()
                    .map_err(|e| anyhow!("HNSEP mutex poisoned: {}", e))?;
                let seg = remover.run(&wave)?;
                CACHE_MANAGER.save_hnsep_cache(&hnsep_path, &seg)?;
                seg
            };
            let (breath_scale, voicing_scale) = (breath.clamp(0., 500.) / 100., voicing.clamp(0., 150.) / 100.);
            let seg_flat = seg_output.as_slice()
                .ok_or_else(|| anyhow!("HNSEP output not contiguous"))?;
            if wave.len() != seg_flat.len() {
                return Err(anyhow!("Length mismatch: wave={}, HNSEP={}", wave.len(), seg_flat.len()));
            }
            if tension != 0. {
                let voicing_seg: Vec<f64> = seg_flat.iter().map(|&s| voicing_scale * s).collect();
                let emphasized = pre_emphasis_base_tension(&voicing_seg, -tension.clamp(-100., 100.) / 50.);
                wave.iter_mut()
                    .zip(seg_flat)
                    .zip(emphasized.iter())
                    .for_each(|((w, &s), &em)| {
                        *w = breath_scale * (*w - s) + em;
                    });
            } else {
                wave.iter_mut()
                    .zip(seg_flat)
                    .for_each(|(w, &s)| {
                        *w = breath_scale * (*w - s) + voicing_scale * s;
                    });
            };
        } else if breath != 100. || voicing != 100. {
            info!("Applying simple volume scaling: {}", breath / 100.);
            let breath_scale = breath.clamp(0., 500.) / 100.; 
            wave.iter_mut().for_each(|x| *x *= breath_scale);
        }
        let wave_max = wave.iter()
            .map(|x| x.abs())
            .filter(|&x| x.is_finite()) 
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_default();
        let scale = if wave_max >= 0.5 {
            info!("Scaling audio to max 0.5 (current: {:.3})", wave_max);
            let s = 0.5 / wave_max;
            wave.iter_mut().for_each(|x| *x *= s);
            s
        } else {
            info!("Audio volume acceptable (max: {:.3})", wave_max);
            1.0
        };
        let gender = self.flags.get("g").and_then(|o| o.as_ref()).copied().unwrap_or(0.).clamp(-600., 600.);
        info!("Gender adjustment: {}", gender);
        let mut mel_origin = get_mel_analyzer()?.call(&wave, gender / 100., 1.);
        info!("Mel shape: {:?}", mel_origin.dim());
        dynamic_range_compression(&mut mel_origin);
        Ok(Features { mel_origin, scale })
    }
    fn resample(&self, features: Features) -> Result<()> {
        if self.out_file.file_name().and_then(|s| s.to_str()) == Some("nul") {
            info!("Null output file - skipping write");
            return Ok(());
        }
        let mut mel_origin = features.mel_origin;
        info!(
            "Modulation: {:.1}, Scale: {:.1}, Mel shape: {:?}",
            self.modulation, features.scale, mel_origin.dim()
        );
        let mel_cols = mel_origin.ncols();
        let mut t_area_origin = Vec::with_capacity(mel_cols);
        for i in 0..mel_cols {
            let val = i as f64 * THOP_ORIGIN + THOP_ORIGIN_HALF;
            t_area_origin.push(val);
        }
        let mut total_time = t_area_origin.last().copied().unwrap_or_default() + THOP_ORIGIN_HALF;
        let vel = (1.0 - self.velocity).exp2();
        let start = self.offset;
        let cutoff = self.cutoff;
        let end = if cutoff < 0.0 { start - cutoff } else { total_time - cutoff };
        let con = start + self.consonant;
        let length_req = self.length;
        let mut stretch_length = end - con;
        info!(
            "Time params: start={:.4}, end={:.4}, con={:.4}, stretch_length={:.4}, length_req={:.4}",
            start, end, con, stretch_length, length_req
        );
        if HIFI_CONFIG.loop_mode || self.flags.contains_key("He") {
            info!("Enabling loop mode");
            let start_idx = (((con + THOP_ORIGIN_HALF) / THOP_ORIGIN)
                .floor() as usize)
                .clamp(0, mel_cols);
            let end_idx = (((end + THOP_ORIGIN_HALF) / THOP_ORIGIN)
                .floor() as usize)
                .clamp(start_idx, mel_cols);
            let mel_loop = mel_origin.slice(s![.., start_idx..end_idx]);
            let pad_size = (length_req / THOP_ORIGIN).floor() as usize + 1;
            let padded_mel = reflect_pad_2d(mel_loop, pad_size);
            mel_origin = ndarray::concatenate(
                Axis(1),
                &[mel_origin.slice(s![.., 0..start_idx]).view(), padded_mel.view()]
            )?;
            stretch_length = pad_size as f64 * THOP_ORIGIN;
            t_area_origin = Vec::with_capacity(mel_origin.ncols()); 
            for i in 0..mel_origin.ncols() {
                let val = i as f64 * THOP_ORIGIN + THOP_ORIGIN_HALF;
                t_area_origin.push(val);
            }
            total_time = t_area_origin.last().copied().unwrap_or_default() + THOP_ORIGIN_HALF;
            info!("Looped mel shape: {:?}, new total time: {:.4}", mel_origin.dim(), total_time);
        }
        let scaling_ratio = if stretch_length < length_req {
            info!("Stretching (ratio: {:.4})", length_req / stretch_length);
            length_req / stretch_length
        } else {
            info!("No stretching needed (ratio: 1.0)");
            1.0
        };
        let stretch = |t: f64| -> f64 {
            if t < vel * con { t / vel } else { con + (t - vel * con) / scaling_ratio }
        };
        let stretched_n_frames = ((con * vel + (total_time - con) * scaling_ratio) / THOP)
            .floor() as usize + 1;
        let mut stretched_t_mel = Vec::with_capacity(stretched_n_frames);
        for i in 0..stretched_n_frames {
            let val = i as f64 * THOP + THOP_HALF;
            stretched_t_mel.push(val);
        }
        let start_left = ((start * vel + THOP_HALF) / THOP).floor() as usize;
        let cut_left = start_left.saturating_sub(HIFI_CONFIG.fill);
        let required_frames = ((length_req + con * vel + THOP_HALF) / THOP).floor() as usize;
        let end_right = stretched_n_frames.saturating_sub(required_frames);
        let cut_right = end_right.saturating_sub(HIFI_CONFIG.fill);
        let (slice_start, slice_end) = (cut_left, stretched_n_frames.saturating_sub(cut_right));
        stretched_t_mel = if slice_start < slice_end {
            if slice_start == 0 && slice_end == stretched_t_mel.len() {
                stretched_t_mel 
            } else {
                stretched_t_mel[slice_start..slice_end].to_vec()
            }
        } else {
            Vec::new()
        };
        info!("Stretched time axis length: {}", stretched_t_mel.len());
        let t_area_max = t_area_origin.last().copied().unwrap_or_default();
        stretched_t_mel.iter_mut().for_each(|t| {
            *t = stretch(*t).clamp(0.0, t_area_max);
        });
        let mel_render = if stretched_t_mel.is_empty() {
            info!("Empty stretched time axis - skipping interpolation");
            Array2::zeros((mel_origin.nrows(), 0))
        } else {
            interp1d(&t_area_origin, &mel_origin, &stretched_t_mel)
        };
        info!("Render mel shape: {:?}", mel_render.dim());
        info!("Processing pitch");
        let mut pitch_base = Vec::with_capacity(self.pitchbend.len());
        for &pb in &self.pitchbend {
            let base = pb + self.pitch;
            let val = self.flags.get("t")
                .and_then(|o| o.as_ref())
                .map_or(base, |&t| base + t.clamp(-1200., 1200.) / 100.0);
            pitch_base.push(val);
        }
        let new_start = start * vel - cut_left as f64 * THOP;
        let new_end = (con * vel + length_req) - cut_left as f64 * THOP;
        let mut t_pitch = Vec::with_capacity(self.pitchbend.len());
        for i in 0..self.pitchbend.len() {
            let val = 60.0 * i as f64 / self.tempo + new_start;
            t_pitch.push(val);
        }
        let pitch_interp = Akima::new(&t_pitch,&pitch_base);
        let mut t = Vec::with_capacity(mel_render.ncols());
        for i in 0..mel_render.ncols() {
            let val = i as f64 * THOP;
            t.push(val);
        }
        let pitch_last = *t_pitch.last().unwrap_or(&0.0);
        let mut t_clamped = Vec::with_capacity(t.len());
        for &x in &t {
            let val = x.clamp(new_start, pitch_last);
            t_clamped.push(val);
        }
        let pitch_render = pitch_interp.sample_with_slice(&t_clamped);
        let pitch_render_len = pitch_render.len();
        let mut f0_render = Vec::with_capacity(pitch_render_len);
        for &x in &pitch_render {
            f0_render.push(midi_to_hz(x));
        }
        info!("F0 render length: {}", f0_render.len());
        let mut render = {
            let vocoder_arc = get_vocoder()?;
            let mut vocoder = vocoder_arc.lock()
                .map_err(|e| anyhow!("Vocoder mutex poisoned: {}", e))?;
            let wav_con = if mel_render.ncols() == 0 {
                Vec::new()
            } else {
                vocoder.run(mel_render, &f0_render)?
            };
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
                wav_con[start_idx..end_idx].to_vec()
            } else {
                Vec::new()
            }
        };
        let render_len = render.len();
        info!("Cropped audio length: {}", render_len);
        if let Some(&a_flag) = self.flags.get("A").and_then(|o| o.as_ref()).filter(|&&a| a != 0.0) {
            info!("Applying amplitude modulation (A={:.1})", a_flag);
            if pitch_render_len > 1 && !t.is_empty() && !render.is_empty() {
                let mut gain_data = Vec::with_capacity(pitch_render_len);
                for i in 0..pitch_render_len {
                    let grad = match i {
                        0 => (pitch_render[1] - pitch_render[0]) / (t[1] - t[0]),
                        i if i == pitch_render_len - 1 => (pitch_render[i] - pitch_render[i-1]) / (t[i] - t[i-1]),
                        _ => (pitch_render[i+1] - pitch_render[i-1]) / (t[i+1] - t[i-1]),
                    };
                    gain_data.push(5.0f64.powf(1e-4 * a_flag.clamp(-100.0, 100.0) * grad));
                }
                let mut audio_time = Vec::with_capacity(render_len);
                for i in 0..render_len {
                    let val = new_start + (new_end - new_start) / render_len as f64 * i as f64;
                    audio_time.push(val);
                }
                let interpolated_gain = interp1d(
                    &t,
                    &Array2::from_shape_vec((1, gain_data.len()), gain_data)
                        .map_err(|e| anyhow!("Invalid gain array: {}", e))?,
                    &audio_time
                );
                render.iter_mut()
                    .zip(interpolated_gain.row(0).iter())
                    .for_each(|(r, g)| *r *= g);
                info!("Amplitude modulation applied");
            } else {
                info!("Insufficient data for amplitude modulation - skipping");
            }
        }
        render.iter_mut().for_each(|x| *x /= features.scale);
        let max = render.iter()
            .map(|x| x.abs())
            .filter(|&x| x.is_finite()) 
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(1.0);
        if let Some(&hg) = self.flags.get("HG").and_then(|o| o.as_ref()) {
            info!("Applying growl (strength: {:.1})", hg);
            render = growl(&render, SR_F64, 80.0, hg.clamp(0.0, 100.0) / 100.0);
        }
        if HIFI_CONFIG.wave_norm {
            let p_strength = self.flags.get("P")
                .and_then(|o| o.as_ref())
                .copied()
                .unwrap_or(100.0)
                .clamp(0.0, 100.0) as u8; 
            render = loudness_norm(&render, SR_F64,  -16.0, p_strength);
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