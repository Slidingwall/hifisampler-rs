use anyhow::{Context, Result, anyhow};
use biquad::{Biquad, Coefficients, DirectForm1, ToHertz, Type};
use crate::utils::lerp;
const MAX_VIBRATO_CENTS: f64 = 100.0;
const HP_CUTOFF_HZ: f64 = 20.0;
const MIN_NYQ_FRAC: f64 = 0.01;
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
fn make_coefficients(
    filter: Type<f64>,
    fs: f64,
    f0: f64,
    q: f64,
) -> Result<Coefficients<f64>> {
    Coefficients::<f64>::from_params(filter, fs.hz(), f0.hz(), q)
        .map_err(|_| anyhow!("Can't make filter coefficients."))
}
#[inline]
fn create_highpass_coeffs(sr_f64: f64, cutoff: f64) -> Result<biquad::Coefficients<f64>> {
    let nyq = sr_f64 / 2.0;
    let clamped_cutoff = cutoff.clamp(MIN_NYQ_FRAC * nyq, 0.99 * nyq);
    make_coefficients(
        biquad::Type::HighPass,
        sr_f64,
        clamped_cutoff,
        1.0 / 2.0_f64.sqrt(),
    ).context("Failed to create highpass coefficients")
}
fn highpass_2nd(audio: &mut [f64], sr: f64, cutoff: f64) -> Result<()> {
    let coeffs = create_highpass_coeffs(sr, cutoff)?;
    let mut filter = DirectForm1::new(coeffs);
    forward_backward_filter(audio, &mut filter, 1);
    Ok(())
}
fn highpass(audio: &[f64], sr: f64, cutoff: f64) -> Result<(Vec<f64>, Vec<f64>)> { 
    let coeffs = create_highpass_coeffs(sr, cutoff)?; 
    let (mut high, mut filter1, mut filter2) = (audio.to_vec(), DirectForm1::new(coeffs), DirectForm1::new(coeffs));
    forward_backward_filter(&mut high, &mut filter1, 1);
    forward_backward_filter(&mut high, &mut filter2, 1);
    let low = audio.iter().zip(high.iter()).map(|(a, h)| a - h).collect();
    Ok((high, low)) 
}
fn square_lfo(num_samples: usize, sr: f64, freq: f64) -> Vec<f64> {
    if num_samples == 0 || freq <= 0.0 {
        return vec![0.0; num_samples];
    }
    let samples_per_period = (sr / freq).max(1.0) as usize; 
    let mut lfo = Vec::with_capacity(num_samples);
    lfo.extend((0..num_samples).map(|n| {
        let phase = (n % samples_per_period) as f64 / samples_per_period as f64;
        if phase < 0.5 { 1.0 } else { -1.0 }
    }));
    lfo
}
fn linear_interp(idx: &[f64], x: &[f64]) -> Vec<f64> {
    if x.is_empty() || idx.is_empty() {
        return vec![0.0; idx.len()];
    }
    let max_x_idx = (x.len() - 1) as f64;
    idx.iter()
        .map(|&i| {
            let i_clamped = i.clamp(0.0, max_x_idx);
            let i_floor = i_clamped.floor() as usize;
            let i_ceil = (i_floor + 1).min(x.len() - 1);
            lerp(x[i_floor], x[i_ceil], i_clamped - i_floor as f64)
        })
        .collect()
}
#[inline]
fn rms(data: &[f64]) -> f64 {
    let sum_sq = data.iter().map(|&x| x * x).sum::<f64>();
    (sum_sq / data.len() as f64).sqrt()
}
fn apply_pitch_modulation(band: &[f64], sr: f64, lfo: &[f64], strength: f64) -> Result<Vec<f64>> {
    if band.len() != lfo.len() {
        return Err(anyhow!(
            "Band/LFO length mismatch: band={}, lfo={}",
            band.len(),
            lfo.len()
        ));
    }
    if band.is_empty() {
        return Ok(Vec::new());
    }
    let band_len = band.len();
    let mut ratio = Vec::with_capacity(band.len());
    ratio.extend(lfo.iter().map(|&l| {
        2.0f64.powf((l * strength * MAX_VIBRATO_CENTS) / 1200.0)
    }));
    let first_ratio = ratio[0];
    let cumsum = ratio.iter().scan(0.0, |state, &r| {
        *state += r;
        Some(*state - first_ratio)
    }).collect::<Vec<_>>();
    let mean_ratio = ratio.iter().sum::<f64>() / band_len as f64;
    let ideal = (0..band_len)
        .map(|i| i as f64 * mean_ratio)
        .collect::<Vec<_>>();
    let mut drift = cumsum.iter()
        .zip(ideal.iter())
        .map(|(c, i)| c - i)
        .collect::<Vec<_>>();
    if band_len > 100 {
        highpass_2nd(&mut drift, sr, HP_CUTOFF_HZ)?; 
    }
    let max_idx = (band_len - 1) as f64;
    let idx = (0..band_len)
        .zip(drift.iter())
        .map(|(i, d)| (i as f64 + d).clamp(0.0, max_idx))
        .collect::<Vec<_>>();
    let modulated = linear_interp(&idx, band);
    let (rms_orig, rms_new) = (rms(band), rms(&modulated));
    if rms_new > 1e-10 {
        Ok(modulated.iter().map(|&m| m * (rms_orig / rms_new)).collect())
    } else {
        Ok(modulated)
    }
}
pub fn growl(audio: &[f64], sample_rate: f64, frequency: f64, strength: f64) -> Vec<f64> {
    if strength <= 0.0 || frequency <= 0.0 {
        return audio.to_vec();
    }
    let (band, complement) = highpass(audio, sample_rate, 400.).unwrap();
    let lfo = square_lfo(audio.len(), sample_rate, frequency);
    let modulated_band = apply_pitch_modulation(&band, sample_rate, &lfo, strength).unwrap();
    complement.iter()
        .zip(modulated_band.iter())
        .map(|(c, m)| c + m)
        .collect()
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
    fn test_lerp_import() {
        assert_approx_eq!(lerp(1.0, 3.0, 0.5), 2.0);
    }
    #[test]
    fn test_highpass_2nd_vs_4th() {
        let mut audio = linspace(0.0, 1.0, 1000);
        let sr = 44100.0;
        let cutoff = 20.0;
        let (audio_4th, _) = highpass(&audio, sr, cutoff).unwrap();
        highpass_2nd(&mut audio, sr, cutoff).unwrap();
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
        assert_eq!(lfo.len(), 100);
        assert!(lfo.iter().all(|&v| v == 1.0 || v == -1.0));
    }
}