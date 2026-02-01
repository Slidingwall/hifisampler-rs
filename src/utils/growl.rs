use biquad::{Biquad, Coefficients, DirectForm1, ToHertz};
use crate::utils::lerp;
const VIBRATO_FACTOR: f64 = 1.0 / 12.0;
const HP_CUTOFF_HZ: f64 = 20.0;
const Q_HIGHPASS: f64 = 0.7071067811865476;
fn forward_backward_filter<F: Biquad<f64>>(
    signal: &mut [f64],
    filter: &mut F,
    repeats: usize,
) {
    for _ in 0..repeats {
        for sample in signal.iter_mut() {
            *sample = filter.run(*sample);
        }
        filter.reset_state();
        for sample in signal.iter_mut().rev() {
            *sample = filter.run(*sample);
        }
        filter.reset_state();
    }
}
#[inline]
fn create_highpass_coeffs(sr_f64: f64, cutoff: f64) -> biquad::Coefficients<f64> {
    Coefficients::<f64>::from_params(
        biquad::Type::HighPass,
        sr_f64.hz(),
        cutoff.hz(),
        Q_HIGHPASS,
    )
    .expect("Failed to create highpass coefficients: invalid sample rate or cutoff frequency")
}
fn highpass_2nd(audio: &mut [f64], sr: f64, cutoff: f64) {
    let mut filter = DirectForm1::new(create_highpass_coeffs(sr, cutoff));
    forward_backward_filter(audio, &mut filter, 1);
}
fn highpass(
    audio: &[f64],
    sr: f64,
    cutoff: f64,
) -> (Vec<f64>, Vec<f64>) { 
    let mut high = audio.to_vec(); 
    let mut filter = DirectForm1::new(create_highpass_coeffs(sr, cutoff));
    forward_backward_filter(&mut high, &mut filter, 2);
    let low = audio.iter()
        .zip(high.iter())
        .map(|(a, h)| a - h)
        .collect::<Vec<f64>>();
    (high, low)
}
fn square_lfo(num_samples: usize, sr: f64, freq: f64) -> Vec<f64> {
    let mut lfo = Vec::with_capacity(num_samples);
    let samples_per_period = (sr * (1.0 / freq)) as usize;
    for n in 0..num_samples {
        if (n as f64 / samples_per_period as f64).fract() < 0.5 {
            lfo.push(1.0);
        } else {
            lfo.push(-1.0);
        }
    }
    lfo
}
fn linear_interp(idx: &[f64], x: &[f64]) -> Vec<f64> {
    let mut output = Vec::with_capacity(idx.len());
    for &i in idx {
        let i_floor = i.floor() as usize;
        let val = if i_floor + 1 >= x.len() {
            x[x.len() - 1]
        } else {
            lerp(x[i_floor], x[i_floor + 1], i - (i_floor as f64))
        };
        output.push(val);
    }
    output
}
#[inline]
fn rms(data: &[f64]) -> f64 {
    let sum_sq = data.iter().fold(0.0, |acc, &x| acc + x * x);
    (sum_sq * (1.0 / data.len() as f64)).sqrt()
}
fn apply_pitch_modulation(
    band: &[f64],
    sr: f64,
    lfo: &[f64],
    strength: f64,
) -> Vec<f64> {
    let band_len = band.len();
    let mut buf = lfo.iter()
        .map(|&l| 2.0f64.powf(l * (strength * VIBRATO_FACTOR)))
        .collect::<Vec<f64>>(); 
    let mean_ratio = buf.iter().sum::<f64>() * (1.0 / band_len as f64);
    let ratio_0 = buf[0];
    let mut cumulative = 0.0;
    for (i, val) in buf.iter_mut().enumerate() {
        cumulative += *val;
        *val = (cumulative - ratio_0) - (i as f64) * mean_ratio;
    }
    highpass_2nd(&mut buf, sr, HP_CUTOFF_HZ);
    for (i, val) in buf.iter_mut().enumerate() {
        *val = (i as f64 + *val).clamp(0.0, (band_len - 1) as f64);
    }
    let mut modulated = linear_interp(&buf, band);
    let gain = rms(band) / rms(&modulated);
    modulated.iter_mut().for_each(|m| *m *= gain);
    modulated
}
pub fn growl(
    audio: &mut Vec<f64>,
    sample_rate: f64,
    frequency: f64,
    strength: f64,
) {
    let original_len = audio.len();
    if original_len == 0 {
        return;
    }
    let complement_original = std::mem::take(audio);
    let (high, mut complement) = highpass(&complement_original, sample_rate, 400.0);
    let modulated_band = apply_pitch_modulation(
        &high,
        sample_rate,
        &square_lfo(original_len, sample_rate, frequency),
        strength,
    );
    complement.iter_mut()
        .zip(modulated_band.iter())
        .for_each(|(c, m)| *c += m);
    *audio = complement;
}