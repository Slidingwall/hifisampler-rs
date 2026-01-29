use bs1770::{ChannelLoudnessMeter, gated_mean};
use ndarray::{Array2, Axis};
use rustfft::num_complex::Complex;
use crate::{
    consts::{FFT_SIZE, HOP_SIZE, HIFI_CONFIG, SAMPLE_RATE},
    utils::{stft::{stft_core, istft_core}, linspace, reflect_pad_1d}, 
};
pub fn pre_emphasis_base_tension(wave: &[f64], b: f64) -> Vec<f64> {
    let original_len = wave.len();
    let freq_bins = FFT_SIZE / 2 + 1;
    let padded_len = ((original_len + HOP_SIZE - 1) / HOP_SIZE) * HOP_SIZE;
    let mut padded_wave = Vec::with_capacity(padded_len);
    padded_wave.extend_from_slice(wave); 
    padded_wave.resize(padded_len, 0.0);
    let complex_spec = stft_core(&padded_wave, None, None)
        .expect("STFT computation failed");
    let (spec_amp, spec_phase) = (
        complex_spec.t().mapv(|c| c.norm()),
        complex_spec.t().mapv(|c| c.arg())
    );
    let mut spec_amp_db = spec_amp.mapv(|x| x.clamp(1e-9, f64::INFINITY).ln());
    let sr = SAMPLE_RATE as f64;
    let filter_coeff = -b * sr / (freq_bins as f64 * 3000.0); 
    spec_amp_db.axis_iter_mut(Axis(1))
        .enumerate()
        .for_each(|(j, mut bin)| {
            let filter = filter_coeff * j as f64 + b;
            let clamped_filter = filter.clamp(-2.0, 2.0);
            bin.iter_mut().for_each(|amp_db| *amp_db += clamped_filter);
        });
    let complex_spec_istft = Array2::from_shape_fn((freq_bins, complex_spec.ncols()), |(k, t)| {
        let amp_db = spec_amp_db[(t, k)];
        let phase = spec_phase[(t, k)];
        let amp = amp_db.exp();
        Complex::new(amp * phase.cos(), amp * phase.sin())
    });
    let filtered_wave = istft_core(&complex_spec_istft, padded_wave.len(), None, None)
        .expect("ISTFT computation failed");
    let original_max = padded_wave.iter().map(|x| x.abs()).max_by(|a, b| a.total_cmp(b)).unwrap_or(0.0);
    let filtered_max = filtered_wave.iter().map(|x| x.abs()).max_by(|a, b| a.total_cmp(b)).unwrap_or(1e-9);
    let gain_coeff = (b / -15.0).clamp(0.0, 0.33) + 1.0;
    filtered_wave.into_iter()
        .take(original_len)
        .map(|x| x * (original_max / filtered_max) * gain_coeff)
        .collect()
}
fn rms_db(audio_segment: &[f64]) -> f64 {
    if audio_segment.is_empty() {
        return -f64::INFINITY;
    }
    let len_f64 = audio_segment.len() as f64;
    let sum_sq: f64 = audio_segment.iter().map(|&x| x * x).sum(); 
    if sum_sq < len_f64 * 1e-20 {
        return -f64::INFINITY;
    }
    20.0 * (sum_sq / len_f64).log10()
}
fn linear_fade(length: usize, fade_in: bool, sample_rate: f64) -> Vec<f64> {
    if length == 0 {
        return Vec::new();
    }
    let fade_len = ((0.2 * sample_rate + 0.5) as usize)
        .min(length)
        .min(length / 4);
    let mut fade = Vec::with_capacity(length);
    if fade_in {
        fade.extend(linspace(0.0, 1.0, fade_len));
        fade.extend(std::iter::repeat(1.0).take(length - fade_len));
    } else {
        fade.extend(std::iter::repeat(1.0).take(length - fade_len));
        fade.extend(linspace(1.0, 0.0, fade_len));
    }
    fade
}
pub fn loudness_norm(
    audio: &[f64],
    sample_rate: f64,
    _: f64,
    loudness_target: f64,
    norm_strength: u8,
) -> Vec<f64> {
    let original_len = audio.len();
    if original_len == 0 {
        return Vec::new();
    }
    let (mut processed, start_idx, need_restore) = if HIFI_CONFIG.trim_silence {
        let frame_len = (0.02 * sample_rate + 0.5) as usize;
        let hop_len = (0.01 * sample_rate + 0.5) as usize;
        let padding_frames = (0.1 * sample_rate + 0.5) as usize / hop_len;
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
                let e = ((end / hop_len + 1 + padding_frames) * hop_len + frame_len).min(audio.len());
                (audio[s..e].to_vec(), s, true)
            }
            None => (audio.to_vec(), 0, false),
        }
    } else {
        (audio.to_vec(), 0, false)
    };
    let min_len = (0.4 * sample_rate + 0.5) as usize;
    if processed.len() < min_len {
        processed = reflect_pad_1d(&processed, 0, min_len - processed.len());
    }
    let sample_rate_u32 = sample_rate
        .clamp(1.0, u32::MAX as f64) 
        .round() as u32; 
    let mut meter = ChannelLoudnessMeter::new(sample_rate_u32);
    meter.push(processed.iter().map(|&x| x as f32));
    let current_loudness = gated_mean(meter.into_100ms_windows().as_ref()).loudness_lkfs() as f64;
    let strength = norm_strength as f64 / 100.0;
    let gain = 10.0f64.powf((loudness_target - current_loudness) * strength / 20.0)
        .clamp(1e-3, 100.0);
    processed.iter_mut().for_each(|x| *x *= gain);
    let mut output = vec![0.0; original_len];
    if need_restore {
        let avail_len = processed.len().min(original_len - start_idx);
        let fade_out = linear_fade(avail_len, false, sample_rate);
        output[start_idx..start_idx + avail_len]
            .iter_mut()
            .zip(processed.iter().take(avail_len))
            .zip(fade_out.iter())
            .for_each(|((out, proc), fade)| *out = proc * fade);
        let remaining = original_len - start_idx - avail_len;
        if remaining > 0 {
            let fade_in = linear_fade(remaining, true, sample_rate);
            let src_range = start_idx + avail_len..start_idx + avail_len + remaining;
            if let (Some(output_slice), Some(audio_slice)) = (
                output.get_mut(src_range.clone()),
                audio.get(src_range),
            ) {
                output_slice
                    .iter_mut()
                    .zip(audio_slice)
                    .zip(fade_in.iter())
                    .for_each(|((out, audio), fade)| *out = audio * fade);
            }
        }
    } else {
        let copy_len = processed.len().min(original_len);
        output[..copy_len].copy_from_slice(&processed[..copy_len]);
    }
    output.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
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
        let normalized = loudness_norm(&audio, rate, peak, target_loudness, 100);
        assert_eq!(normalized.len(), audio.len());
        assert!(normalized.iter().all(|&x| x.abs() <= peak));
        let mut meter = ChannelLoudnessMeter::new(rate.round() as u32);
        meter.push(normalized.iter().map(|&x| x as f32).collect::<Vec<_>>().into_iter());
        let power = meter.into_100ms_windows();
        let integrated_power = gated_mean(power.as_ref());
        let measured_loudness = integrated_power.loudness_lkfs() as f64;
        assert!((measured_loudness - target_loudness).abs() < 1.0);
    }
    #[test]
    fn test_empty_input() {
        let empty_signal = Vec::new();
        let normalized = loudness_norm(&empty_signal, 44100.0, 1.0, -23.0, 100);
        assert!(empty_signal.is_empty());
        assert!(normalized.is_empty());
    }
    #[test]
    fn test_edge_cases() {
        let signal = vec![0.1; 44100];
        let extreme_quiet = loudness_norm(&signal, 44100.0, 1.0, -50.0, 100);
        let extreme_loud = loudness_norm(&signal, 44100.0, 1.0, 0.0, 100);
        assert!(extreme_quiet.iter().all(|&x| x.abs() <= 1.0));
        assert!(extreme_loud.iter().all(|&x| x.abs() <= 1.0));
    }
}