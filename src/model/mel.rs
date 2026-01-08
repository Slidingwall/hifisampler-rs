use crate::{
    consts,
    utils::{mel_basis::MEL_BASIS_DATA, reflect_pad_1d, stft::stft_core},
};
use ndarray::{s, Array2};
const TARGET_BINS: usize = consts::FFT_SIZE / 2 + 1; 
pub struct MelAnalyzer {
    mel_basis: Array2<f64>,
}
impl MelAnalyzer {
    pub fn new() -> Self {
        let mel_basis = Array2::from_shape_vec((128, 1025), MEL_BASIS_DATA.iter().flatten().cloned().collect())
            .expect("Invalid MEL_BASIS_DATA shape (expected 128x1025)");
        Self { mel_basis }
    }
    pub fn call(&self, wave: &[f64], key_shift: f64, speed: f64) -> Array2<f64> {
        let factor = 2f64.powf(key_shift / 12.0);
        let fft_size = (consts::FFT_SIZE as f64 * factor).round() as usize;
        let hop_length = (consts::ORIGIN_HOP_SIZE as f64 * speed).round() as usize;
        let scale_factor = consts::FFT_SIZE as f64 / fft_size as f64;
        let (pad_left, pad_right) = ((fft_size - hop_length) / 2, (fft_size - hop_length + 1) / 2);
        let padded_wave = reflect_pad_1d(wave, pad_left, pad_right);
        let complex_spec = stft_core(&padded_wave, Some(fft_size), Some(hop_length))
            .expect("STFT failed: invalid parameters or signal length");
        let (n_fft_bins, n_frames) = (complex_spec.nrows(), complex_spec.ncols());
        let spec = complex_spec.mapv(|c| c.norm());
        let processed_spec = if key_shift != 0. {
            if n_fft_bins < TARGET_BINS {
                let mut padded = Array2::zeros((TARGET_BINS, n_frames));
                padded.slice_mut(s![..n_fft_bins, ..]).assign(&spec);
                padded.mapv_inplace(|x| x * scale_factor);
                padded
            } else {
                spec.slice(s![..TARGET_BINS, ..]).to_owned()
                    .mapv(|x| x * scale_factor)
            }
        } else {
            spec
        };
        self.mel_basis.dot(&processed_spec)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{consts, utils::linspace};
    #[test]
    fn test_mel_analyzer() {
        let analyzer = MelAnalyzer::new();
        let sample_len = consts::FFT_SIZE * 10; 
        let y = linspace(0., 1., sample_len);
        let mel_spec = analyzer.call(&y, 0., 1.0);
        let (pad_left, pad_right) = ((consts::FFT_SIZE - consts::HOP_SIZE) / 2, (consts::FFT_SIZE - consts::HOP_SIZE + 1) / 2);
        let expected_frames = ((sample_len + pad_left + pad_right - consts::FFT_SIZE) / consts::HOP_SIZE) + 1;
        assert_eq!(mel_spec.dim(), (128, expected_frames));
        assert!(mel_spec.iter().all(|&x| !x.is_nan()));
    }
}