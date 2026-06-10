use biquad::{Biquad, Coefficients, DirectForm1, ToHertz};
use once_cell::sync::Lazy;
use crate::{consts::SAMPLE_RATE, utils::lerp};
fn forward_backward_filter<F: Biquad<f32>>(signal: &mut [f32], filter: &mut F, repeats: usize) {
    for _ in 0..repeats {
        signal.iter_mut().for_each(|s| *s = filter.run(*s));
        filter.reset_state();
        signal.iter_mut().rev().for_each(|s| *s = filter.run(*s));
        filter.reset_state();
    }
}
#[inline(always)]
fn highpass_coeffs(cutoff: f32) -> Coefficients<f32> {
    let sr = SAMPLE_RATE as f32;
    Coefficients::from_params(
        biquad::Type::HighPass,
        sr.hz(),
        cutoff.hz(),
        1.0 / 2.0f32.sqrt(),
    )
    .expect("Failed to create highpass coefficients")
}
static HIGH_400_COEFF: Lazy<Coefficients<f32>> = Lazy::new(|| highpass_coeffs(400.0));
static HIGH_20_COEFF: Lazy<Coefficients<f32>> = Lazy::new(|| highpass_coeffs(20.0));
pub fn growl(audio: &mut Vec<f32>, freq: f32, strength: f32) {
    let len = audio.len();
    if len == 0 || strength <= 0.0 || freq <= 0.0 {
        return;
    }
    let mut high = audio.clone();
    let mut filter_400 = DirectForm1::new(*HIGH_400_COEFF);
    forward_backward_filter(&mut high, &mut filter_400, 2);
    for (a, h) in audio.iter_mut().zip(high.iter()) {
        *a = *a - *h;
    }
    let sr = SAMPLE_RATE as f32;
    let cycle = (sr / freq) as usize;
    let half = cycle / 2;
    let factor_up = (strength / 12.0).exp2();
    let mut buf: Vec<f32> = (0..len)
        .map(|n| if n % cycle < half { factor_up } else { 1.0 / factor_up })
        .collect();
    let mean = buf.iter().sum::<f32>() / len as f32;
    let init = buf[0];
    let mut cumulative = 0.0;
    for (i, v) in buf.iter_mut().enumerate() {
        cumulative += *v;
        *v = cumulative - init - i as f32 * mean;
    }
    if len > 100 {
        let mut filter_20 = DirectForm1::new(*HIGH_20_COEFF);
        forward_backward_filter(&mut buf, &mut filter_20, 1);
    }
    for (i, v) in buf.iter_mut().enumerate() {
        let idx = i as f32 + *v;
        let idx_clamped = idx.clamp(0.0, len as f32 - 1.0);
        let f = idx_clamped.floor() as usize;
        let frac = idx_clamped.fract();
        let modulated_val = if f >= len - 1 {
            high[len - 1]
        } else {
            lerp(high[f], high[f + 1], frac)
        };
        *v = modulated_val;
    }
    let mut sum_h_sq = 0.0;
    let mut sum_m_sq = 0.0;
    for (h, m) in high.iter().zip(buf.iter()) {
        sum_h_sq += h * h;
        sum_m_sq += m * m;
    }
    let rms_h = (sum_h_sq / len as f32).sqrt();
    let rms_m = (sum_m_sq / len as f32).sqrt();
    let scale = if rms_m > 1e-10 { rms_h / rms_m } else { 0.0 };
    for (a, m) in audio.iter_mut().zip(buf.iter()) {
        *a += m * scale;
    }
}