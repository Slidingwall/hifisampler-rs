use bs1770::{ChannelLoudnessMeter, gated_mean};
use ndarray::{Axis, azip};
use oxifft::Complex;
use crate::{
    consts::{FFT_SIZE, HOP_SIZE, HIFI_CONFIG, SAMPLE_RATE},
    utils::{stft::{stft_core, istft_core}, reflect_pad_1d}, 
};
pub fn pre_emphasis_base_tension(wave: &mut Vec<f32>, b: f32) {
    let orig_len = wave.len();
    let orig_max = wave.iter().fold(0f32, |m, &x| m.max(x.abs()));
    wave.resize(((orig_len + HOP_SIZE - 1) / HOP_SIZE) * HOP_SIZE, 0.0);
    let mut spec = stft_core(wave, FFT_SIZE, HOP_SIZE);
    let mut spec_amp = spec.mapv(|c| c.norm().max(1e-9).ln());
    spec_amp.axis_iter_mut(Axis(0)).enumerate().for_each(|(j, mut bin)| {
        bin.iter_mut().for_each(|v| *v += (b * (1.0 - j as f32 * (SAMPLE_RATE as f32 / (FFT_SIZE / 1500 + 3000) as f32))).clamp(-2.0f32, 2.0f32));
    });
    azip!((comp in &mut spec, &amp_db in &spec_amp) {
        *comp = Complex::new(amp_db.exp() * comp.arg().cos(), amp_db.exp() * comp.arg().sin());
    });
    let mut filtered = istft_core(&spec, wave.len(), FFT_SIZE, HOP_SIZE);
    let f_max = filtered.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let gain = (orig_max / f_max) * ((-b / 15.0).clamp(0.0, 0.33) + 1.0);
    wave.truncate(orig_len);
    wave.iter_mut()
        .zip(filtered.drain(..orig_len))
        .for_each(|(w, f)| *w = f * gain);
    wave.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
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
            let (mut start, mut end) = (None, 0);
            (0..=orig_len - fl).step_by(hl)
                .filter(|&i| {
                    let f = &wave[i..i+fl];
                    (f.iter().map(|&x| x.powi(2)).sum::<f32>() / fl as f32).sqrt() >= 1e-10 
                    && 20.0 * (f.iter().map(|&x| x.powi(2)).sum::<f32>() / fl as f32).sqrt().log10() > HIFI_CONFIG.silence_threshold
                })
                .for_each(|i| { start.get_or_insert(i); end = i; });
            if let Some(s) = start {
                val_start = s;
                val_end = ((end / hl + 11) * hl + fl).min(orig_len);
                need_restore = true;
            }
        }
    }
    let val_len = val_end - val_start;
    if val_len == 0 { return; }
    if val_len < (0.4 * sample_rate) as usize {
        reflect_pad_1d(wave, 0, (0.4 * sample_rate) as usize - val_len);
    }
    let mut meter = ChannelLoudnessMeter::new(sample_rate as u32);
    meter.push(wave[val_start..(val_start + val_len.max((0.4 * sample_rate) as usize)).min(wave.len())].iter().copied());
    let gain = 10.0f32.powf((target - gated_mean(meter.into_100ms_windows().as_ref()).loudness_lkfs()) * norm_strength as f32 * 0.0005);
    wave[val_start..val_end].iter_mut().for_each(|x| *x *= gain);
    if need_restore {
        wave[..val_start].fill(0.0);
        wave[val_end..].fill(0.0);
        let fade_len = ((0.2 * sample_rate) as usize).min(val_len >> 2);
        wave[val_start..val_end].iter_mut().enumerate().for_each(|(i, x)| {
            *x *= if i >= val_len - fade_len { 
                (i - (val_len - fade_len)) as f32 / (fade_len - 1).max(1) as f32 
            } else { 1.0 };
        });
    }
    wave.truncate(orig_len);
    wave.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
}