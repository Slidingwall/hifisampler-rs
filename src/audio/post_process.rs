use bs1770::{ChannelLoudnessMeter, gated_mean};
use ndarray::{Array2, Axis};
use rustfft::num_complex::Complex;
use crate::{
    consts::{FFT_SIZE, HOP_SIZE, HIFI_CONFIG, SAMPLE_RATE},
    utils::{stft::{stft_core, istft_core}, reflect_pad_1d}, 
};
pub fn pre_emphasis_base_tension(wave: &[f64], b: f64) -> Vec<f64> {
    let original_len = wave.len();
    let padded_len = ((original_len + HOP_SIZE - 1) / HOP_SIZE) * HOP_SIZE;
    let mut padded_wave = wave.to_vec();
    padded_wave.resize(padded_len, 0.0);
    
    let complex_spec = stft_core(&padded_wave, None, None);
    let (spec_amp, spec_phase) = (
        complex_spec.mapv(|c| c.norm()),
        complex_spec.mapv(|c| c.arg())
    );
    let mut spec_amp_db = spec_amp.mapv(|x| x.max(1e-9).ln());
    spec_amp_db.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(j, mut bin)| {
            let filter = b * (1.0 - (SAMPLE_RATE as f64 * j as f64) / (FFT_SIZE / 1500 + 3000) as f64);
            bin.iter_mut().for_each(|amp_db| *amp_db += filter.clamp(-2.0, 2.0));
        });
    let complex_spec_istft = Array2::from_shape_fn((FFT_SIZE / 2 + 1, complex_spec.ncols()), |(k, t)| {
        let phase = spec_phase[(k, t)];
        let amp = spec_amp_db[(k, t)].exp();
        Complex::new(
            amp * phase.cos(),
            amp * phase.sin()
        )
    });
    let filtered_wave = istft_core(&complex_spec_istft, padded_wave.len(), None, None);
    let original_max = padded_wave.iter().map(|x| x.abs()).max_by(|a, b| a.total_cmp(b)).unwrap();
    let filtered_max = filtered_wave.iter().map(|x| x.abs()).max_by(|a, b| a.total_cmp(b)).unwrap();
    filtered_wave.into_iter()
        .take(original_len)
        .map(|x| x * (original_max / filtered_max) * ((b / -15.0).max(0.0) + 1.0))
        .collect()
}
fn rms_db(audio_segment: &[f64]) -> f64 {
    let sum_sq: f64 = audio_segment.iter()
        .map(|&x| x * x) 
        .sum();
    let rms=(sum_sq / audio_segment.len() as f64).sqrt();
    if rms < 1e-10 {
        return f64::NEG_INFINITY;
    }
    20.0 * rms.log10()
}
fn linear_fade(length: usize, fade_in: bool, sample_rate: f64) -> Vec<f64> {
    let fade_len = ((0.2 * sample_rate) as usize).min(length / 4);
    let mut fade = Vec::with_capacity(length);
    if fade_in {
        fade.resize(length, 1.0);
        for i in 0..fade_len {
            let val = if fade_len == 1 { 0.0 } else { (1.0 / (fade_len - 1) as f64) * i as f64 };
            fade[i] = val;
        }
    } else {
        fade.extend(std::iter::repeat(1.0).take(length - fade_len));
        for i in 0..fade_len {
            let val = if fade_len == 1 { 1.0 } else { 1.0 - (1.0 / (fade_len - 1) as f64) * i as f64 };
            fade.push(val);
        }
    }
    fade
}
pub fn loudness_norm(
    audio: &[f64],
    sample_rate: f64,
    loudness_target: f64,
    norm_strength: u8,
) -> Vec<f64> {
    let original_len = audio.len();
    if original_len == 0 {
        return Vec::new();
    }
    let (mut processed, start_idx, need_restore) = if HIFI_CONFIG.trim_silence {
        let frame_len = (0.02 * sample_rate) as usize;
        let hop_len = (0.01 * sample_rate) as usize;
        let mut start = None;
        let mut end = 0;
        let max_i = audio.len().saturating_sub(frame_len);
        for i in (0..=max_i).step_by(hop_len) {
            if rms_db(&audio[i..i + frame_len]) > HIFI_CONFIG.silence_threshold {
                start.get_or_insert(i);
                end = i;
            }
        }
        match start {
            Some(s) => {
                let e = ((end / hop_len + 1 + ((0.1 * sample_rate) as usize / hop_len)) * hop_len + frame_len).min(audio.len());
                (audio[s..e].to_vec(), s, true)
            }
            None => (audio.to_vec(), 0, false),
        }
    } else {
        (audio.to_vec(), 0, false)
    };
    let trimmed_len = processed.len();
    let min_len = (0.4 * sample_rate) as usize;
    if processed.len() < min_len {
        let pad_right = min_len - processed.len();
        reflect_pad_1d(&mut processed, 0, pad_right);
    }
    let mut meter = ChannelLoudnessMeter::new(sample_rate as u32);
    meter.push(processed.iter().map(|&x| x as f32));
    let measured_loudness = gated_mean(meter.into_100ms_windows().as_ref()).loudness_lkfs() as f64;
    let gain = 10.0f64.powf((loudness_target - measured_loudness) * norm_strength as f64 * 0.0005);
    processed.iter_mut().for_each(|x| *x *= gain);
    if processed.len() > trimmed_len {
        processed.truncate(trimmed_len);
    }
    let mut output = vec![0.0; original_len];
    if need_restore {
        let fade_out = linear_fade(processed.len(), false, sample_rate);
        output[start_idx..start_idx + processed.len()]
            .iter_mut()
            .zip(processed.iter())
            .zip(fade_out.iter())
            .for_each(|((out, proc), fade)| *out = proc * fade);
        let available_length = processed.len().min(original_len - start_idx);
        let end_pos = start_idx + available_length;
        if end_pos < original_len {
            let remain_length = original_len - end_pos;
            let fade_length = ((0.2 * sample_rate) as usize).min(available_length / 4);
            let crossfade_length = fade_length.min(remain_length);
            if crossfade_length > 0 {
                for i in 0..remain_length {
                    let fade_val = if i < crossfade_length {
                        0.0 + (1.0 / (crossfade_length - 1) as f64) * i as f64
                    } else {
                        1.0
                    };
                    output[end_pos + i] = audio[end_pos + i] * fade_val;
                }
            }
        }
    } else {
        output[..processed.len()].copy_from_slice(&processed);
    }
    output.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
    if original_len < min_len {
        output.truncate(original_len);
    }
    output
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_loudness_norm() {
        let audio: Vec<f64> = (0..44100).map(|i| 0.5 * (i as f64 * 0.01).sin()).collect();
        let rate = 44100.0;
        let peak = 1.0;
        let target_loudness = -23.0;
        let normalized = loudness_norm(&audio, rate,  target_loudness, 100);
        assert_eq!(normalized.len(), audio.len());
        assert!(normalized.iter().all(|&x| x.abs() <= peak));
        let mut meter = ChannelLoudnessMeter::new(rate.round() as u32);
        meter.push(normalized.iter().map(|&x| x as f32));
        let power = meter.into_100ms_windows();
        let integrated_power = gated_mean(power.as_ref());
        let measured_loudness = integrated_power.loudness_lkfs() as f64;
        assert!((measured_loudness - target_loudness).abs() < 1.0);
    }
    #[test]
    fn test_edge_cases() {
        let signal = vec![0.1; 44100];
        let extreme_quiet = loudness_norm(&signal, 44100.0,  -50.0, 100);
        let extreme_loud = loudness_norm(&signal, 44100.0,  0.0, 100);
        assert!(extreme_quiet.iter().all(|&x| x.abs() <= 1.0));
        assert!(extreme_loud.iter().all(|&x| x.abs() <= 1.0));
    }
}