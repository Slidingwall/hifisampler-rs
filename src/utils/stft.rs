use std::sync::Arc;
use ndarray::{Array2, ArrayView1, s};
use once_cell::sync::{Lazy, OnceCell};
use dashmap::DashMap;
use oxifft::{Complex, Direction, Flags, Plan, streaming::WindowFunction, threading::{get_default_pool, ThreadPool}};
static HANN_WINDOWS: Lazy<DashMap<usize, Arc<Vec<f64>>>> = Lazy::new(DashMap::new);
static FFT_PLANS: Lazy<DashMap<(usize, Direction), Arc<Plan<f64>>>> = Lazy::new(DashMap::new);
static ISTFT_WINDOW_SQ: Lazy<Arc<Vec<f64>>> = Lazy::new(|| {
    let window = get_hann_window(crate::consts::FFT_SIZE);
    Arc::new(window.iter().map(|&w| w * w).collect())
});
fn get_hann_window(fft_size: usize) -> Arc<Vec<f64>> {
    HANN_WINDOWS
        .entry(fft_size)
        .or_insert_with(|| {
            Arc::new(WindowFunction::Hann.generate(fft_size))
        })
        .clone()
}
fn get_fft_plan(fft_size: usize, direction: Direction) -> Arc<Plan<f64>> {
    FFT_PLANS
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
    let window = get_hann_window(fft_size);
    let plan = get_fft_plan(fft_size, Direction::Forward);
    let n_frames = (signal.len() - fft_size) / hop_size + 1;
    let mut spec = Array2::from_shape_fn((freq_bins, n_frames), |_| Complex::zero()); 
    let pool = get_default_pool();
    let result: Arc<Vec<OnceCell<Vec<Complex<f64>>>>> = Arc::new(
        (0..n_frames)
            .map(|_| OnceCell::new()) 
            .collect()
    );
    pool.parallel_for(n_frames, |frame_idx| {
        let start = frame_idx * hop_size;
        let input: Vec<Complex<f64>> = signal[start..start + fft_size]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        let mut output = vec![Complex::zero(); fft_size];
        plan.execute(&input, &mut output);
        output.truncate(freq_bins); 
        let _ = result[frame_idx].set(output);
    });
    for (frame_idx, once_result) in result.iter().enumerate() {
        spec.slice_mut(s![.., frame_idx]).assign(&ArrayView1::from(once_result.get().unwrap()));
    }
    spec
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
    let window = get_hann_window(fft_size);
    let plan = get_fft_plan(fft_size, Direction::Backward);
    let out_len = fft_size + (n_frames - 1) * hop_size;
    let mut output = vec![0.0; out_len];
    let mut win_sum = vec![0.0; out_len];
    let scale = 1.0 / fft_size as f64;
    let pool = get_default_pool();
    let result: Arc<Vec<OnceCell<Vec<f64>>>> = Arc::new(
        (0..n_frames)
            .map(|_| OnceCell::new())
            .collect()
    );
    pool.parallel_for(n_frames, |frame_idx| {
        let mut full_spec = vec![Complex::zero(); fft_size];
        let mut frame = vec![Complex::zero(); fft_size];
        let spec_slc = spec.slice(s![.., frame_idx]);
        let spec_raw = match spec_slc.as_slice() {
            Some(s) if s.len() == freq_bins => s,
            _ => return, 
        };
        full_spec[0..freq_bins].copy_from_slice(spec_raw);
        for i in 1..freq_bins - 1 {
            full_spec[fft_size - i] = full_spec[i].conj();
        }
        plan.execute(&full_spec, &mut frame);
        let ifft_result: Vec<f64> = frame
            .iter()
            .zip(window.iter())
            .map(|(frame_val, win_val)| frame_val.re * scale * win_val)
            .collect();
        let _ = result[frame_idx].set(ifft_result);
    });
    let window_sq = ISTFT_WINDOW_SQ.as_ref();
    for (frame_idx, once_result) in result.iter().enumerate() {
        let res = once_result.get().unwrap();
        let start = frame_idx * hop_size;
        for i in 0..fft_size {
            output[start + i] += res[i];
            win_sum[start + i] += window_sq[i];
        }
    }
    for i in 0..out_len {
        if win_sum[i] > 1e-10 {
            output[i] /= win_sum[i];
        }
    }
    output.resize(target_len, 0.0);
    output
}