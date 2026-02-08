use biquad::{Biquad, Coefficients, DirectForm1, ToHertz};
use crate::utils::lerp;
const VIBRATO_FACTOR: f32 = 1.0 / 12.0;
const HP_CUTOFF_HZ: f32 = 20.0;
const Q_HIGHPASS: f32 = 0.7071067811865476;
fn forward_backward_filter<F: Biquad<f32>>(
    signal: &mut [f32],
    filter: &mut F,
    repeats: usize,
) {
    (0..repeats).for_each(|_| {
    signal.iter_mut().for_each(|sample| *sample = filter.run(*sample));
    filter.reset_state();
    signal.iter_mut().rev().for_each(|sample| *sample = filter.run(*sample));
    filter.reset_state();
});
}
#[inline]
fn create_highpass_coeffs(sr: f32, cutoff: f32) -> biquad::Coefficients<f32> {
    Coefficients::<f32>::from_params(
        biquad::Type::HighPass,
        sr.hz(),
        cutoff.hz(),
        Q_HIGHPASS,
    )
    .expect("Failed to create highpass coefficients: invalid sample rate or cutoff frequency")
}
fn highpass_2nd(audio: &mut [f32], sr: f32, cutoff: f32) {
    let mut filter = DirectForm1::new(create_highpass_coeffs(sr, cutoff));
    forward_backward_filter(audio, &mut filter, 1);
}
fn highpass(
    audio: &[f32],
    sr: f32,
    cutoff: f32,
) -> (Vec<f32>, Vec<f32>) { 
    let mut high = audio.to_vec(); 
    let mut filter = DirectForm1::new(create_highpass_coeffs(sr, cutoff));
    forward_backward_filter(&mut high, &mut filter, 2);
    let low = audio.iter()
        .zip(high.iter())
        .map(|(a, h)| a - h)
        .collect::<Vec<f32>>();
    (high, low)
}
fn square_lfo(num: usize, sr: f32, freq: f32) -> Vec<f32> {
    let mut lfo = Vec::with_capacity(num);
    let samples = (sr / freq) as usize;
    let half_samples = samples / 2;
    for n in 0..num {
        if (n % samples) < half_samples {
            lfo.push(1.0);
        } else {
            lfo.push(-1.0);
        }
    }
    lfo
}
fn linear_interp(idx: &[f32], x: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(idx.len());
    let max_idx = x.len() - 1;
    for &i in idx {
        let floor_idx = i.floor() as usize;
        let val = if floor_idx >= max_idx {
            x[max_idx]
        } else {
            lerp(x[floor_idx], x[floor_idx + 1], i.fract())
        };
        output.push(val);
    }
    output
}
#[inline]
fn rms(data: &[f32]) -> f32 {
    let sum_sq = data.iter().fold(0.0, |acc, &x| acc + x * x);
    (sum_sq * (1.0 / data.len() as f32)).sqrt()
}
fn apply_pitch_modulation(
    band: &[f32],
    sr: f32,
    lfo: &[f32],
    strength: f32,
) -> Vec<f32> {
    let band_len = band.len();
    let vibrato = strength * VIBRATO_FACTOR;
    let mut buf = lfo.iter()
        .map(|&l| 2.0f32.powf(l * vibrato))
        .collect::<Vec<f32>>(); 
    let mean_ratio = buf.iter().sum::<f32>() / band_len as f32;
    let ratio_0 = buf[0];
    let mut cumulative = 0.0;
    for (i, val) in buf.iter_mut().enumerate() {
        cumulative += *val;
        *val = (cumulative - ratio_0) - (i as f32) * mean_ratio;
    }
    highpass_2nd(&mut buf, sr, HP_CUTOFF_HZ);
    let max_idx =(band_len - 1) as f32;
    for (i, val) in buf.iter_mut().enumerate() {
        *val = (i as f32 + *val).clamp(0.0, max_idx);
    }
    let mut modulated = linear_interp(&buf, band);
    let gain = rms(band) / rms(&modulated);
    modulated.iter_mut().for_each(|m| *m *= gain);
    modulated
}
pub fn growl(
    audio: &mut Vec<f32>,
    sr: f32,
    freq: f32,
    strength: f32,
) {
    let orig_len = audio.len();
    if orig_len == 0 {
        return;
    }
    let orig_audio = std::mem::take(audio);
    let (high, mut complement) = highpass(&orig_audio, sr, 400.0);
    let mod_band = apply_pitch_modulation(
        &high,
        sr,
        &square_lfo(orig_len, sr, freq),
        strength,
    );
    complement.iter_mut()
        .zip(mod_band.iter())
        .for_each(|(c, m)| *c += m);
    *audio = complement;
}