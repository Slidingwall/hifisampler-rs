use lazy_static::lazy_static;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{collections::HashMap, sync::Arc};
use parking_lot::RwLock;
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
                    .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64 / fft_size as f64)).cos())
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
) -> Array2<Complex<f64>> {
    let fft_size = fft_size.unwrap_or(consts::FFT_SIZE);
    let hop_size = hop_size.unwrap_or(consts::HOP_SIZE);
    let mut complex_spec = Array2::zeros((fft_size / 2 + 1, (signal.len() - fft_size) / hop_size + 1));
    let mut fft_buffer = vec![Complex::new(0.0, 0.0); fft_size];
    let window = get_hann_window(fft_size);
    let (forward_fft, _) = get_fft_pair(fft_size);
    for frame_idx in 0..complex_spec.ncols() {
        let start = frame_idx * hop_size;
        Zip::from(&mut fft_buffer)
            .and(&signal[start..start + fft_size])
            .and(&*window)
            .for_each(|buf, &sig, &win| *buf = Complex::new(sig * win, 0.0));
        forward_fft.process(&mut fft_buffer);
        complex_spec.slice_mut(s![.., frame_idx]).assign(
            &ndarray::ArrayView1::from(&fft_buffer[..fft_size / 2 + 1])
        );
    }
    complex_spec
}
pub fn istft_core(
    spec: &Array2<Complex<f64>>,
    target_len: usize,
    fft_size: Option<usize>,
    hop_size: Option<usize>,
) -> Vec<f64> {
    let fft_size = fft_size.unwrap_or(consts::FFT_SIZE);
    let hop_size = hop_size.unwrap_or(consts::HOP_SIZE);
    let freq_bins = fft_size / 2 + 1;
    let mut signal = vec![0.0; target_len];
    let mut window_sum = vec![0.0; target_len];
    let mut ifft_buffer = vec![Complex::new(0.0, 0.0); fft_size];
    let (_, inverse_fft) = get_fft_pair(fft_size);
    let window = get_hann_window(fft_size);
    for frame_idx in 0..spec.ncols() {
        let start = frame_idx * hop_size;
        let end = start + fft_size;
        let actual_end = end.min(target_len);
        ifft_buffer[..freq_bins].copy_from_slice(
            spec.slice(s![.., frame_idx])
                .as_slice().unwrap()
        );
        for i in 1..freq_bins - 1 {
            ifft_buffer[fft_size - i] = ifft_buffer[i].conj();
        }
        inverse_fft.process(&mut ifft_buffer);
        Zip::from(&mut signal[start..actual_end])
            .and(&ifft_buffer[..actual_end - start])
            .and(&window[..actual_end - start])
            .and(&mut window_sum[start..actual_end])
            .for_each(|sig, &c, &win, sum| {
                *sig += c.re * win * (1.0 / fft_size as f64);
                *sum += win * win;
            });
    }
    signal.iter_mut().zip(window_sum.iter()).for_each(|(sig, &sum)| {
        *sig /= if sum < 1e-8 { 1e-8 } else { sum };
    });
    signal
}