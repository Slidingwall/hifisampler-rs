use bs1770::{ChannelLoudnessMeter, gated_mean};
use ndarray::{Array2, Axis};
use crate::{
    consts::{FFT_SIZE, HIFI_CONFIG, SAMPLE_RATE},
    utils::{reflect_pad_1d}, 
};
pub fn pre_emphasis_base_tension(spec: &mut Array2<f32>, b: f32) {
    let mut orig_max = 0.0f32;
    let mut f_max = 0.0f32;
    spec.axis_iter_mut(Axis(0)).enumerate().for_each(|(j, mut bin)| {
        let coeff = b * (1.0 - j as f32 * SAMPLE_RATE as f32 / (FFT_SIZE / 1500 + 3000) as f32);
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
pub fn loudness_norm(
    wave: &mut Vec<f32>,
    sample_rate: f32,
    target: f32,
    norm_strength: u8,
) {
    let orig_len = wave.len();
    if orig_len == 0 { return; }
    let (mut val_start, mut val_end, mut need_restore) = (0, orig_len, false);
    if HIFI_CONFIG.trim_silence {
        let fl = (0.02 * sample_rate) as usize;
        let hl = (0.01 * sample_rate) as usize;
        if fl <= orig_len {
            let n_windows = (orig_len - fl) / hl + 1;
            let energy_thresh = 10.0f32.powf(HIFI_CONFIG.silence_threshold / 10.0) * fl as f32;
            let mut sum_sq: f32 = wave[0..fl].iter().map(|&x| x * x).sum();
            let mut start_idx = if sum_sq > energy_thresh { Some(0) } else { None };
            let mut end_idx = 0;
            for i in 1..n_windows {
                let prev_start = (i - 1) * hl;
                let new_start = i * hl;
                for j in prev_start..new_start {
                    sum_sq -= wave[j] * wave[j];
                }
                let prev_end = prev_start + fl;
                let new_end = new_start + fl;
                for j in prev_end..new_end {
                    sum_sq += wave[j] * wave[j];
                }
                if sum_sq > energy_thresh {
                    start_idx.get_or_insert(i);
                    end_idx = i;
                }
            }
            if let Some(s) = start_idx {
                val_start = s * hl;
                val_end = ((end_idx + 11) * hl + fl).min(orig_len);
                need_restore = true;
            }
        }
    }
    let val_len = val_end - val_start;
    if val_len == 0 { return; }
    let min_len = (0.4 * sample_rate) as usize;
    if val_len < min_len { reflect_pad_1d(wave, 0, min_len - val_len); }
    let measure_end = (val_start + val_len.max(min_len)).min(wave.len());
    let mut meter = ChannelLoudnessMeter::new(sample_rate as u32);
    meter.push(wave[val_start..measure_end].iter().copied());
    let gain = 10.0f32.powf(
        (target - gated_mean(meter.into_100ms_windows().as_ref()).loudness_lkfs()) * norm_strength as f32 * 0.0005,
    );
    if need_restore {
        let fade_len = ((0.2 * sample_rate) as usize).min(val_len >> 2);
        let fade_scale = 1.0 / (fade_len.max(1) - 1) as f32;
        wave[val_start..val_end]
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                *x *= gain;
                if i >= val_len - fade_len {
                    *x *= (i - (val_len - fade_len)) as f32 * fade_scale;
                }
            });
        wave[..val_start].fill(0.0);
        wave[val_end..].fill(0.0);
    } else {
        wave[val_start..val_end].iter_mut().for_each(|x| *x *= gain);
    }
    wave.truncate(orig_len);
    wave.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
}