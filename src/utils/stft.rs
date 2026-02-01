use ndarray::{Array2, ArrayView1, s};
use oxifft::streaming::{stft, istft, WindowFunction};
use oxifft::Complex;
use crate::consts;
pub fn stft_core(
    signal: &[f64],
    fft_size: Option<usize>,
    hop_size: Option<usize>,
) -> Array2<Complex<f64>> {
    let fft_size = fft_size.unwrap_or(consts::FFT_SIZE);
    let hop_size = hop_size.unwrap_or(consts::HOP_SIZE);
    let oxifft_spectrogram = stft(signal, fft_size, hop_size, WindowFunction::Hann);
    let freq_bins = fft_size / 2 + 1;
    if oxifft_spectrogram.is_empty() {
        return Array2::from_shape_vec((freq_bins, 0), Vec::new()).unwrap();
    }
    let n_frames = oxifft_spectrogram.len();
    let mut complex_spec = Array2::from_shape_fn((freq_bins, n_frames), |_| Complex::zero());
    for (frame_idx, mut frame) in oxifft_spectrogram.into_iter().enumerate() {
        frame.truncate(freq_bins);
        complex_spec.slice_mut(s![.., frame_idx]).assign(&ArrayView1::from(&frame));
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
    let (freq_bins, n_frames) = (spec.nrows(), spec.ncols());
    if n_frames == 0 || freq_bins == 0 || freq_bins != fft_size / 2 + 1 {
        return vec![0.0; target_len];
    }
    let mut oxifft_spectrogram = Vec::with_capacity(n_frames);
    for frame_idx in 0..n_frames {
        let frame_vec: Vec<Complex<f64>> = spec.slice(s![.., frame_idx]).to_vec();
        oxifft_spectrogram.push(frame_vec);
    }
    let mut reconstructed = istft(&oxifft_spectrogram, hop_size, WindowFunction::Hann);
    if reconstructed.len() >= target_len {
        reconstructed.truncate(target_len);
    } else {
        reconstructed.resize(target_len, 0.0);
    }
    reconstructed
}