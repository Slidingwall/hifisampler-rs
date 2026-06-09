use crate::consts::{FFT_SIZE, HOP_SIZE};
use std::sync::Arc;
use ndarray::{Array3, s};
use once_cell::sync::{Lazy, OnceCell};
use oxifft::{api::rfft, Complex, streaming::WindowFunction, threading::{get_default_pool, ThreadPool}}; 
static HANN_WINDOW: Lazy<Arc<Vec<f32>>> = Lazy::new(|| {
    Arc::new(WindowFunction::Hann.generate(FFT_SIZE))
});
pub fn stft_core(signal: &[f32]) -> Array3<f32> {
    let freq_bins = FFT_SIZE / 2 + 1; 
    let n_frames = (signal.len() + HOP_SIZE - 1) / HOP_SIZE;
    let window = &**HANN_WINDOW;
    let frame_data = Arc::new(
        (0..n_frames)
            .map(|_| OnceCell::<Vec<Complex<f32>>>::new())
            .collect::<Vec<_>>()
    );
    get_default_pool().parallel_for(n_frames, |frame_idx| {
        let start = frame_idx * HOP_SIZE;
        let mut real_input = vec![0.0f32; FFT_SIZE];
        let slice_end = (start + FFT_SIZE).min(signal.len());
        for (i, (&s, &w)) in signal[start..slice_end].iter().zip(window.iter()).enumerate() {
            real_input[i] = s * w;
        }
        let spectrum = rfft(&real_input);
        let _ = frame_data[frame_idx].set(spectrum);
    });
    let mut spec = Array3::zeros((2, freq_bins, n_frames));
    for (frame_idx, cell) in frame_data.iter().enumerate() {
        let data = cell.get().unwrap();
        let mut frame_view = spec.slice_mut(s![.., .., frame_idx]);
        for (f, &c) in data.iter().enumerate() {
            frame_view[[0, f]] = c.re;
            frame_view[[1, f]] = c.im;
        }
    }
    spec
}