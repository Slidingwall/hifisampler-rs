use std::sync::Arc;
use ndarray::{Array2};
use once_cell::sync::{Lazy, OnceCell};
use dashmap::DashMap;
use oxifft::{Complex, Direction, Flags, Plan, streaming::WindowFunction, threading::{get_default_pool, ThreadPool}};
static HANN_WINDOWS: Lazy<DashMap<usize, Arc<Vec<f32>>>> = Lazy::new(DashMap::new);
static FFT_PLANS: Lazy<DashMap<(usize, Direction), Arc<Plan<f32>>>> = Lazy::new(DashMap::new);
#[allow(dead_code)]
static ISTFT_WINDOW_SQ: Lazy<Arc<Vec<f32>>> = Lazy::new(|| {
    let window = get_hann_window(crate::consts::FFT_SIZE);
    Arc::new(window.iter().map(|&w| w * w).collect())
});
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
pub fn stft_core(signal: &[f32], fft_size: usize, hop_size: usize) -> (Array2<f32>, Array2<f32>) {
    let freq_bins = fft_size / 2 + 1;
    if fft_size == 0 || hop_size == 0 {
        return (
            Array2::zeros((freq_bins, 0)),
            Array2::zeros((freq_bins, 0))
        );
    }
    let n_frames = (signal.len() + hop_size - 1) / hop_size;
    let window = get_hann_window(fft_size);
    let plan = get_fft_plan(fft_size, Direction::Forward);
    let mut real = Array2::zeros((freq_bins, n_frames));
    let mut imag = Array2::zeros((freq_bins, n_frames));
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
    for (frame_idx, cell) in frame_data.iter().enumerate() {
        let data = cell.get().unwrap();
        let mut r_col = real.column_mut(frame_idx);
        let mut i_col = imag.column_mut(frame_idx);
        for (f, &c) in data.iter().enumerate() {
            r_col[f] = c.re;
            i_col[f] = c.im;
        }
    }
    (real, imag)
}
#[allow(dead_code)]
pub fn istft_core(
    spec: &(Array2<f32>, Array2<f32>),
    target_len: usize,
    fft_size: usize,
    hop_size: usize,
) -> Vec<f32> {
    let (real, imag) = spec;
    let (freq_bins, n_frames) = real.dim();
    if n_frames == 0
        || freq_bins != fft_size / 2 + 1
        || imag.dim() != (freq_bins, n_frames)
    {
        return vec![0.0; target_len];
    }
    let window = get_hann_window(fft_size);
    let plan = get_fft_plan(fft_size, Direction::Backward);
    let out_len = fft_size + (n_frames - 1) * hop_size;
    let mut output = vec![0.0; out_len];
    let mut win_sum = vec![0.0; out_len];
    let scale = 1.0 / fft_size as f32;
    let max_bins = freq_bins - 1;
    let window_sq = ISTFT_WINDOW_SQ.as_ref();
    let result = Arc::new((0..n_frames).map(|_| OnceCell::new()).collect::<Vec<_>>());
    get_default_pool().parallel_for(n_frames, |frame_idx| {
        let mut full_spec = vec![Complex::zero(); fft_size];
        let real_col = real.column(frame_idx);
        let imag_col = imag.column(frame_idx);
        for f in 0..freq_bins {
            full_spec[f] = Complex::new(real_col[f], imag_col[f]);
        }
        (1..max_bins).for_each(|i| {
            full_spec[fft_size - i] = full_spec[i].conj();
        });
        let mut frame = vec![Complex::zero(); fft_size];
        plan.execute(&full_spec, &mut frame);
        let ifft_result: Vec<f32> = frame
            .iter()
            .zip(window.iter())
            .map(|(v, w)| v.re * scale * w)
            .collect();
        let _ = result[frame_idx].set(ifft_result);
    });
    result.iter().enumerate().for_each(|(frame_idx, once_result)| {
        let res = once_result.get().unwrap();
        let start = frame_idx * hop_size;
        let output_slice = &mut output[start..start + fft_size];
        let win_slice = &mut win_sum[start..start + fft_size];
        for i in 0..fft_size {
            output_slice[i] += res[i];
            win_slice[i] += window_sq[i]; 
        }
    });
    for i in 0..out_len {
        if win_sum[i] > 1e-10 {
            output[i] /= win_sum[i];
        }
    }
    output.resize(target_len, 0.0);
    output
}