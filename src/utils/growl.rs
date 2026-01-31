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
    let coeffs = create_highpass_coeffs(sr, cutoff); 
    let mut filter = DirectForm1::new(coeffs);
    forward_backward_filter(audio, &mut filter, 1);
}
fn highpass(audio: &[f64], sr: f64, cutoff: f64) -> (Vec<f64>, Vec<f64>) { 
    let coeffs = create_highpass_coeffs(sr, cutoff); 
    let mut high = audio.to_vec();
    let mut filter = DirectForm1::new(coeffs);
    forward_backward_filter(&mut high, &mut filter, 2);
    let low = audio.iter()
        .zip(high.iter())
        .map(|(a, h)| a - h)
        .collect::<Vec<f64>>();
    (high, low) 
}
fn square_lfo(num_samples: usize, sr: f64, freq: f64) -> Vec<f64> {
    let samples_per_period = (sr * (1.0 / freq)) as usize;
    (0..num_samples)
        .map(|n| {
            let phase = (n % samples_per_period) as f64 * (1.0 / samples_per_period as f64);
            if phase < 0.5 { 1.0 } else { -1.0 }
        })
        .collect()
}
fn linear_interp(idx: &[f64], x: &[f64]) -> Vec<f64> {
    idx.iter()
        .map(|&i| {
            let i_floor = i.floor() as usize;
            lerp(x[i_floor], x[i_floor + 1], i - (i_floor as f64))
        })
        .collect()
}
#[inline]
fn rms(data: &[f64]) -> f64 {
    let sum_sq = data.iter().fold(0.0, |acc, &x| acc + x * x);
    (sum_sq * (1.0 / data.len() as f64)).sqrt()
}
fn apply_pitch_modulation(band: &[f64], sr: f64, lfo: &[f64], strength: f64) -> Vec<f64> {
    let mut ratio = Vec::with_capacity(band.len());
    ratio.extend(lfo.iter().map(|&l| {
        2.0f64.powf(l * (strength * VIBRATO_FACTOR))
    }));
    let mut drift = Vec::with_capacity(band.len());
    let mean_ratio = ratio.iter().sum::<f64>() * (1.0 / band.len() as f64);
    ratio.iter().scan(0.0, |state, &r| {
        *state += r;
        Some(*state - ratio[0])
    }).enumerate().for_each(|(i, c)| {
        let ideal_val = (i as f64) * mean_ratio;
        drift.push(c - ideal_val);
    });
    highpass_2nd(&mut drift, sr, HP_CUTOFF_HZ); 
    let idx = drift.iter().enumerate()
        .map(|(i, d)| (i as f64 + d).clamp(0.0, (band.len() - 1) as f64))
        .collect::<Vec<_>>();
    let mut modulated = linear_interp(&idx, band);
    let gain_ratio =rms(band) / rms(&modulated);
    modulated.iter_mut().for_each(|m| *m *= gain_ratio);
    modulated
}
pub fn growl(audio: &[f64], sample_rate: f64, frequency: f64, strength: f64) -> Vec<f64> {
    let (band, mut complement) = highpass(audio, sample_rate, 400.);
    let lfo = square_lfo(audio.len(), sample_rate, frequency);
    let modulated_band = apply_pitch_modulation(&band, sample_rate, &lfo, strength);
    complement.iter_mut()
        .zip(modulated_band.iter())
        .for_each(|(c, m)| *c += m);
    complement
}
#[cfg(test)]
mod tests {
    use crate::utils::linspace;
    const EPSILON: f64 = 1e-6; 
    use super::*;
    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr, $epsilon:expr) => {
            assert!(
                ($a - $b).abs() < $epsilon,
                "Assertion failed: {} ≈ {} (epsilon: {}), but difference is {}",
                $a, $b, $epsilon, ($a - $b).abs()
            );
        };
        ($a:expr, $b:expr) => {
            assert_approx_eq!($a, $b, EPSILON);
        };
    }
    #[test]
    fn test_cumsum() {
        let ratio = vec![1.0, 2.0, 3.0];
        let first_ratio = ratio[0];
        let cumsum = ratio.iter().scan(0.0, |state, &r| {
            *state += r;
            Some(*state - first_ratio)
        }).collect::<Vec<_>>();
        assert_eq!(cumsum, vec![0.0, 2.0, 5.0]);
    }
    #[test]
    fn test_growl_no_strength() {
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(growl(&audio, 44100.0, 80.0, 0.0), audio);
    }
    #[test]
    fn test_highpass_2nd_vs_4th() {
        let mut audio = linspace(0.0, 1.0, 1000);
        let sr = 44100.0;
        let cutoff = 20.0;
        let (audio_4th, _) = highpass(&audio, sr, cutoff);
        highpass_2nd(&mut audio, sr, cutoff);
        assert_ne!(audio, audio_4th);
    }
    #[test]
    fn test_linear_interp_alignment() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let idx = vec![0.5, 1.2, 3.7, 4.9];
        let rust_interp = linear_interp(&idx, &x);
        let python_expected = vec![1.5, 2.2, 4.7, 5.0];
        for (r, p) in rust_interp.iter().zip(python_expected.iter()) {
            assert_approx_eq!(*r, *p);
        }
    }
    #[test]
    fn test_square_lfo_high_freq() {
        let lfo = square_lfo(100, 44100.0, 100000.0);
        assert!(lfo.iter().all(|&v| v == 1.0 || v == -1.0));
    }
}