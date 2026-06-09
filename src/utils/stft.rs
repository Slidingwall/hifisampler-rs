use std::sync::Arc;
use ndarray::{Array3, s};
use once_cell::sync::{Lazy, OnceCell};
use dashmap::DashMap;
use oxifft::{Complex, Direction, Flags, Plan, streaming::WindowFunction, threading::{get_default_pool, ThreadPool}};
static HANN_WINDOWS: Lazy<DashMap<usize, Arc<Vec<f32>>>> = Lazy::new(DashMap::new);
static FFT_PLANS: Lazy<DashMap<(usize, Direction), Arc<Plan<f32>>>> = Lazy::new(DashMap::new);
fn get_hann_window(fft_size: usize) -> Arc<Vec<f32>> {
    HANN_WINDOWS.entry(fft_size)
        .or_insert_with(|| {
            Arc::new(WindowFunction::Hann.generate(fft_size))
        }).clone()
}
fn get_fft_plan(fft_size: usize, direction: Direction) -> Arc<Plan<f32>> {
    FFT_PLANS.entry((fft_size, direction))
        .or_insert_with(|| {
            Arc::new(
                Plan::dft_1d(fft_size, direction, Flags::ESTIMATE)
                    .expect(&format!("Failed to generate FFT plan for size {} and direction {:?}", fft_size, direction))
            )
        }).clone()
}
pub fn stft_core(signal: &[f32], fft_size: usize, hop_size: usize) -> Array3<f32> {
    let freq_bins = fft_size / 2 + 1;
    if fft_size == 0 || hop_size == 0 {
        return Array3::zeros((2, freq_bins, 0));
    }
    let n_frames = (signal.len() + hop_size - 1) / hop_size;
    let window = get_hann_window(fft_size);
    let plan = get_fft_plan(fft_size, Direction::Forward);
    let frame_data = Arc::new(
        (0..n_frames)
            .map(|_| OnceCell::<Vec<Complex<f32>>>::new())
            .collect::<Vec<_>>()
    );
    get_default_pool().parallel_for(n_frames, |frame_idx| {
        let start = frame_idx * hop_size;
        let mut input = vec![Complex::default(); fft_size];
        let slice_end = (start + fft_size).min(signal.len());
        for (i, (&s, &w)) in signal[start..slice_end]
            .iter()
            .zip(window.iter())
            .enumerate()
        {
            input[i] = Complex::new(s * w, 0.0);
        }
        let mut output = vec![Complex::default(); fft_size];
        plan.execute(&mut input, &mut output);
        output.truncate(freq_bins);
        let _ = frame_data[frame_idx].set(output);
    });
    let mut spec = Array3::zeros((2, freq_bins, n_frames));
    for (frame_idx, cell) in frame_data.iter().enumerate() {
        let data = cell.get().unwrap();
        {
            let mut real_col = spec.slice_mut(s![0, .., frame_idx]);
            for (f, &c) in data.iter().enumerate() {
                real_col[f] = c.re;
            }
        }
        {
            let mut imag_col = spec.slice_mut(s![1, .., frame_idx]);
            for (f, &c) in data.iter().enumerate() {
                imag_col[f] = c.im;
            }
        }
    }
    spec
}