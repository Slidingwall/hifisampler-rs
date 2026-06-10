use ebur128::{EbuR128, Mode};
use ndarray::{Array2, Axis};
use crate::{audio::base_coeff::BASE_COEFF, consts::{HIFI_CONFIG, SAMPLE_RATE}, utils::reflect_pad_1d};
pub fn pre_emphasis_base_tension(spec: &mut Array2<f32>, b: f32) {
    let mut orig_max = 0.0f32;
    let mut f_max = 0.0f32;
    spec.axis_iter_mut(Axis(0)).enumerate().for_each(|(j, mut bin)| {
        let coeff = b * BASE_COEFF[j];
        let scale = coeff.clamp(-2.0, 2.0).exp();
        for v in bin.iter_mut() {
            if *v > orig_max { orig_max = *v; }
            *v *= scale;
            if *v > f_max { f_max = *v; }
        }
    });
    let gain = (orig_max / f_max) * ((-b / 15.0).clamp(0.0, 0.33) + 1.0);
    spec.mapv_inplace(|x| x * gain);
}
pub fn loudness_norm(wave: &mut Vec<f32>, target: f32, norm_strength: u8) {
    let orig_len = wave.len();
    if orig_len == 0 { return; }
    let (mut val_start, mut val_end, mut need_restore) = (0, orig_len, false);
    if HIFI_CONFIG.trim_silence {
        if 882 <= orig_len {
            let n_windows = (orig_len - 882) / 441 + 1;
            let energy_thresh = 10.0f32.powf(HIFI_CONFIG.silence_threshold / 10.0) * 882 as f32;
            let mut sum_sq: f32 = wave[0..882].iter().map(|&x| x * x).sum();
            let mut start_idx = if sum_sq > energy_thresh { Some(0) } else { None };
            let mut end_idx = 0;
            for i in 1..n_windows {
                let prev_start = (i - 1) * 441;
                let new_start = i * 441;
                for j in prev_start..new_start {
                    sum_sq -= wave[j] * wave[j];
                }
                let prev_end = prev_start + 882;
                let new_end = new_start + 882;
                for j in prev_end..new_end {
                    sum_sq += wave[j] * wave[j];
                }
                if sum_sq > energy_thresh {
                    start_idx.get_or_insert(i);
                    end_idx = i;
                }
            }
            if let Some(s) = start_idx {
                val_start = s * 441;
                val_end = (end_idx * 441 + 5733).min(orig_len);
                need_restore = true;
            }
        }
    }
    let val_len = val_end - val_start;
    if val_len == 0 { return; }
    if val_len < 17640 {
        reflect_pad_1d(wave, 0, 17640 - val_len);
    }
    let measure_end = (val_start + val_len.max(17640)).min(wave.len());
    let audio_to_measure = &wave[val_start..measure_end];
    let mut ebu = EbuR128::new(1, SAMPLE_RATE, Mode::I)
        .expect("Failed to create EbuR128");
    ebu.add_frames_f32(audio_to_measure)
        .expect("Failed to add frames to EbuR128");
    let loudness_lkfs = ebu.loudness_global().unwrap_or(-150.0) as f32;
    let gain = 10.0f32.powf(
        (target - loudness_lkfs) * norm_strength as f32 * 0.0005,
    );
    if need_restore {
        let fade_len = 8820.min(val_len >> 2);
        let fade_scale = 1.0 / (fade_len - 1) as f32;
        let vf = val_len - fade_len;
        for (i, x) in wave[val_start..val_end].iter_mut().enumerate() {
            let mut g = gain;
            if i >= vf {
                g *= (i - vf) as f32 * fade_scale;
            }
            *x *= g;
        }
        wave[..val_start].fill(0.0);
        wave[val_end..].fill(0.0);
    } else {
        for x in &mut wave[val_start..val_end] {
            *x *= gain;
        }
    }
    wave.truncate(orig_len);
    wave.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
}