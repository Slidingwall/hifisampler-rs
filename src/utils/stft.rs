use lazy_static::lazy_static;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{collections::HashMap, sync::Arc};
use parking_lot::RwLock;
use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Zip};
use crate::consts;
lazy_static! {
    static ref HANN_WINDOWS: RwLock<HashMap<usize, Arc<Vec<f64>>>> = RwLock::new(HashMap::new());
    static ref FFT_PAIRS: RwLock<HashMap<usize, (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>)>> = RwLock::new(HashMap::new());
}
fn get_hann_window(fft_size: usize) -> Arc<Vec<f64>> {
    HANN_WINDOWS
        .write()
        .entry(fft_size)
        .or_insert_with(|| {
            Arc::new(
                (0..fft_size)
                    .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / fft_size as f64).cos()))
                    .collect()
            )
        })
        .clone()
}
fn get_fft_pair(fft_size: usize) -> (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>) {
    FFT_PAIRS
        .write()
        .entry(fft_size)
        .or_insert_with(|| {
            let mut planner = FftPlanner::new();
            (
                Arc::from(planner.plan_fft_forward(fft_size)),
                Arc::from(planner.plan_fft_inverse(fft_size))
            )
        })
        .clone()
}
pub fn stft_core(
    signal: &[f64],
    fft_size: Option<usize>,
    hop_size: Option<usize>,
) -> Result<Array2<Complex<f64>>> {
    let fft_size = fft_size.unwrap_or(consts::FFT_SIZE);
    let hop_size = hop_size.unwrap_or(consts::HOP_SIZE);
    if signal.len() < fft_size {
        return Err(anyhow!("Signal length {} < FFT size {}", signal.len(), fft_size));
    }
    if hop_size > fft_size {
        return Err(anyhow!("Hop size {} > FFT size {}", hop_size, fft_size));
    }
    let freq_bins = fft_size / 2 + 1;
    let num_frames = (signal.len() - fft_size) / hop_size + 1;
    let window = get_hann_window(fft_size);
    let (forward_fft, _) = get_fft_pair(fft_size);
    let mut complex_spec = Array2::zeros((freq_bins, num_frames));
    let mut fft_buffer = vec![Complex::new(0.0, 0.0); fft_size];
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let frame = &signal[start..start + fft_size];
        Zip::from(&mut fft_buffer)
            .and(frame)
            .and(&*window)
            .for_each(|buf, &sig, &win| *buf = Complex::new(sig * win, 0.0));
        forward_fft.process(&mut fft_buffer);
        complex_spec.slice_mut(s![.., frame_idx])
            .assign(&ndarray::aview1(&fft_buffer[..freq_bins]));
    }
    Ok(complex_spec)
}
pub fn istft_core(
    spec: &Array2<Complex<f64>>,
    target_len: usize,
    fft_size: Option<usize>,
    hop_size: Option<usize>,
) -> Result<Vec<f64>> {
    let fft_size = fft_size.unwrap_or(consts::FFT_SIZE);
    let hop_size = hop_size.unwrap_or(consts::HOP_SIZE);
    let freq_bins = fft_size / 2 + 1;
    if spec.nrows() != freq_bins {
        return Err(anyhow!("Spectrum bins {} != expected {}", spec.nrows(), freq_bins));
    }
    let num_frames = spec.ncols();
    let max_possible_len = (num_frames - 1) * hop_size + fft_size;
    if max_possible_len < target_len {
        return Err(anyhow!(
            "Insufficient frames: max length {} < target {}",
            max_possible_len,
            target_len
        ));
    }
    let window = get_hann_window(fft_size);
    let (_, inverse_fft) = get_fft_pair(fft_size);
    let mut signal = vec![0.0; target_len];
    let mut window_sum = vec![0.0; target_len];
    let mut ifft_buffer = vec![Complex::new(0.0, 0.0); fft_size];
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        if start >= target_len {
            break;
        }
        let end = start + fft_size;
        let actual_end = end.min(target_len);
        let process_len = actual_end - start;
        let frame_spec = spec.slice(s![.., frame_idx]);
        ifft_buffer[..freq_bins].copy_from_slice(frame_spec.as_slice().ok_or_else(|| {
            anyhow!("Spectrum frame is not contiguous")
        })?);
        for i in 1..freq_bins - 1 {
            ifft_buffer[fft_size - i] = ifft_buffer[i].conj();
        }
        inverse_fft.process(&mut ifft_buffer);
        Zip::from(&mut signal[start..actual_end])
            .and(&ifft_buffer[..process_len])
            .and(&window[..process_len])
            .and(&mut window_sum[start..actual_end])
            .for_each(|sig, &c, &win, sum| {
                *sig += c.re * win / fft_size as f64;
                *sum += win * win;
            });
    }
    signal.iter_mut().zip(window_sum.iter()).for_each(|(sig, &sum)| {
        *sig /= sum.max(1e-8); 
    });
    Ok(signal)
}