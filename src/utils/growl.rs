use biquad::{Biquad, Coefficients, DirectForm1, ToHertz};
use crate::utils::lerp;
fn forward_backward_filter<F: Biquad<f32>>(signal: &mut [f32], filter: &mut F, repeats: usize) {
    for _ in 0..repeats {
        signal.iter_mut().for_each(|s| *s = filter.run(*s));
        filter.reset_state();
        signal.iter_mut().rev().for_each(|s| *s = filter.run(*s));
        filter.reset_state();
    }
}
#[inline(always)]
fn highpass_coeffs(sr: f32, cutoff: f32) -> Coefficients<f32> {
    Coefficients::from_params(
        biquad::Type::HighPass,
        sr.hz(),
        cutoff.hz(),
        1.0 / 2.0f32.sqrt()
    ).expect("Failed to create highpass coefficients: invalid sample rate or cutoff frequency")
}
pub fn growl(audio: &mut Vec<f32>, sr: f32, freq: f32, strength: f32) {
    let len = audio.len();
    if len == 0 || strength <= 0.0 || freq <= 0.0 { return; }
    let orig = std::mem::take(audio);
    let mut high = orig.clone();
    forward_backward_filter(&mut high, &mut DirectForm1::new(highpass_coeffs(sr, 400.0)), 2);
    let mut out = orig.iter().zip(&high).map(|(o, h)| o - h).collect::<Vec<_>>();
    let cycle = (sr / freq) as usize;
    let half = cycle / 2;
    let mut buf: Vec<f32> = (0..len)
        .map(|n| if n % cycle < half { 1.0 } else { -1.0 })
        .map(|lfo| 2.0f32.powf(lfo * strength / 12.0))
        .collect();
    let mean = buf.iter().sum::<f32>() / len as f32;
    let init = buf[0];
    let mut cumulative = 0.0;
    buf.iter_mut().enumerate().for_each(|(i, v)| {
        cumulative += *v;
        *v = cumulative - init - i as f32 * mean;
    });
    if len > 100 {
        forward_backward_filter(&mut buf, &mut DirectForm1::new(highpass_coeffs(sr, 20.0)), 1);
    }
    buf.iter_mut().enumerate().for_each(|(i, v)| *v = (i as f32 + *v).clamp(0.0, len as f32 - 1.0));
    let mut modulated = buf.iter().map(|&idx| {
        let f = idx.floor() as usize;
        if f >= len - 1 { high[len-1] } else { lerp(high[f], high[f+1], idx.fract()) }
    }).collect::<Vec<_>>();
    let rms_h = (high.iter().map(|&x| x*x).sum::<f32>() / len as f32).sqrt();
    let rms_m = (modulated.iter().map(|&x| x*x).sum::<f32>() / len as f32).sqrt();
    if rms_m > 1e-10 { modulated.iter_mut().for_each(|x| *x *= rms_h / rms_m); }
    out.iter_mut().zip(&modulated).for_each(|(o, m)| *o += m);
    *audio = out;
}