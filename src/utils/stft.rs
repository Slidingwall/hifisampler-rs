use std::{collections::HashMap, sync::Arc};
use ndarray::{Array2, ArrayView1, s};
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::RwLock;
use oxifft::{Complex, Direction, Flags, Plan, streaming::WindowFunction, threading::{get_default_pool, ThreadPool}};
static HANN_WINDOWS: Lazy<RwLock<HashMap<usize, Arc<Vec<f64>>>>> = Lazy::new(|| {
    RwLock::new(HashMap::new())
});
static FFT_PLANS: Lazy<RwLock<HashMap<(usize, Direction), Arc<Plan<f64>>>>> = Lazy::new(|| {
    RwLock::new(HashMap::new())
});
static ISTFT_WINDOW_SQ: Lazy<Arc<Vec<f64>>> = Lazy::new(|| {
    let window = get_hann_window(crate::consts::FFT_SIZE);
    Arc::new(window.iter().map(|&w| w * w).collect())
});
fn get_hann_window(fft_size: usize) -> Arc<Vec<f64>> {
    HANN_WINDOWS
        .write()
        .entry(fft_size)
        .or_insert_with(|| {
            Arc::new(WindowFunction::Hann.generate(fft_size))
        })
        .clone()
}
fn get_fft_plan(fft_size: usize, direction: Direction) -> Arc<Plan<f64>> {
    FFT_PLANS
        .write()
        .entry((fft_size, direction))
        .or_insert_with(|| {
            Arc::new(
                Plan::dft_1d(fft_size, direction, Flags::ESTIMATE)
                    .expect(&format!("Failed to generate FFT plan for size {} and direction {:?}", fft_size, direction))
            )
        })
        .clone()
}
pub fn stft_core(
    signal: &[f64],
    fft_size: usize,
    hop_size: usize,
) -> Array2<Complex<f64>> {
    let freq_bins = fft_size / 2 + 1; 
    if fft_size == 0 || hop_size == 0 || signal.len() < fft_size {
        return Array2::from_shape_vec((freq_bins, 0), Vec::new()).unwrap();
    }
    let window_coeffs = get_hann_window(fft_size);
    let plan = get_fft_plan(fft_size, Direction::Forward);
    let num_frames = (signal.len() - fft_size) / hop_size + 1;
    let mut spectrogram = Array2::from_shape_fn((freq_bins, num_frames), |_| Complex::zero()); 
    let pool = get_default_pool();
    let frame_results: Arc<Vec<OnceCell<Vec<Complex<f64>>>>> = Arc::new(
        (0..num_frames)
            .map(|_| OnceCell::new()) 
            .collect()
    );
    pool.parallel_for(num_frames, |frame_idx| {
        let start = frame_idx * hop_size;
        let mut input = Vec::with_capacity(fft_size);
        let mut fft_output = vec![Complex::zero(); fft_size];
        input.extend(
            signal[start..start + fft_size]
                .iter()
                .zip(window_coeffs.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
        );
        plan.execute(&input, &mut fft_output);
        fft_output.truncate(freq_bins); 
        let _ = frame_results[frame_idx].set(fft_output);
    });
    for (frame_idx, once_result) in frame_results.iter().enumerate() {
        spectrogram.slice_mut(s![.., frame_idx]).assign(&ArrayView1::from(once_result.get().unwrap()));
    }
    spectrogram
}
pub fn istft_core(
    spec: &Array2<Complex<f64>>,
    target_len: usize,
    fft_size: usize,
    hop_size: usize,
) -> Vec<f64> {
    let (freq_bins, n_frames) = (spec.nrows(), spec.ncols());
    if n_frames == 0 || freq_bins == 0 || freq_bins != fft_size / 2 + 1 {
        return vec![0.0; target_len];
    }
    let window_coeffs = get_hann_window(fft_size);
    let plan = get_fft_plan(fft_size, Direction::Backward);
    let output_len = fft_size + (n_frames - 1) * hop_size;
    let mut output = vec![0.0; output_len];
    let mut window_sum = vec![0.0; output_len];
    let scale = 1.0 / fft_size as f64;
    let pool = get_default_pool();
    let frame_iff_results: Arc<Vec<OnceCell<Vec<f64>>>> = Arc::new(
        (0..n_frames)
            .map(|_| OnceCell::new())
            .collect()
    );
    pool.parallel_for(n_frames, |frame_idx| {
        let mut full_spectrum = vec![Complex::zero(); fft_size];
        let mut frame = vec![Complex::zero(); fft_size];
        let spec_slice = spec.slice(s![.., frame_idx]);
        let spec_raw = match spec_slice.as_slice() {
            Some(s) if s.len() == freq_bins => s,
            _ => return,
        };
        full_spectrum[0..freq_bins].copy_from_slice(spec_raw);
        for i in 1..freq_bins - 1 {
            full_spectrum[fft_size - i] = full_spectrum[i].conj();
        }
        plan.execute(&full_spectrum, &mut frame);
        let mut iff_result = Vec::with_capacity(fft_size);
        for i in 0..fft_size {
            iff_result.push(frame[i].re * scale * window_coeffs[i]);
        }
        let _ = frame_iff_results[frame_idx].set(iff_result);
    });
    for (frame_idx, once_result) in frame_iff_results.iter().enumerate() {
        let iff_result = once_result.get().unwrap();
        let start = frame_idx * hop_size;
        for i in 0..fft_size {
            output[start + i] += iff_result[i];
            window_sum[start + i] += ISTFT_WINDOW_SQ[i];
        }
    }
    for i in 0..output_len {
        if window_sum[i] > 1e-10 {
            output[i] /= window_sum[i];
        }
    }
    output.resize(target_len, 0.0);
    output
}