use crate::{
    consts,
    utils::{mel_basis::MEL_BASIS_DATA, reflect_pad_1d, stft::stft_core},
};
use ndarray::{Array2, ArrayView1, Axis, parallel::prelude::*, s};
const TARGET_BINS: usize = consts::FFT_SIZE / 2 + 1;
pub fn mel(wave: &mut Vec<f64>, key_shift: f64, speed: f64) -> Array2<f64> {
    let fft_size = (consts::FFT_SIZE as f64 * 2f64.powf(key_shift / 12.0)).round() as usize;
    let hop_length = (consts::ORIGIN_HOP_SIZE as f64 * speed).round() as usize;
    let scale_factor = consts::FFT_SIZE as f64 / fft_size as f64;
    reflect_pad_1d(wave, (fft_size - hop_length) / 2, (fft_size - hop_length + 1) / 2);
    let complex_spec = stft_core(&wave, Some(fft_size), Some(hop_length));
    let n_frames = complex_spec.ncols();
    let mut spec = Array2::zeros((complex_spec.nrows(), n_frames));
    par_azip!((spec_elem in &mut spec, complex_elem in &complex_spec) {
        *spec_elem = complex_elem.norm();
    });
    let processed_spec = if key_shift != 0. {
        let mut target = Array2::zeros((TARGET_BINS, n_frames));
        let source_view = if complex_spec.nrows() < TARGET_BINS {
            spec.view()
        } else {
            spec.slice(s![..TARGET_BINS, ..])
        };
        target.slice_mut(s![..source_view.nrows(), ..]).assign(&source_view);
        target.par_mapv_inplace(|x| x * scale_factor);
        target
    } else {
        spec
    };
    let mut mel_spec = Array2::zeros((128, n_frames));
    par_azip!((mut mel_row in mel_spec.axis_iter_mut(Axis(0)), nonzeros in ArrayView1::from(&MEL_BASIS_DATA)) {
        for (frame_idx, mel_val) in mel_row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for &(freq_idx, weight) in *nonzeros {
                if freq_idx < processed_spec.nrows() {
                    sum += processed_spec[(freq_idx, frame_idx)] * weight;
                }
            }
            *mel_val = sum;
        }
    });
    mel_spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{consts, utils::linspace};
    #[test]
    fn test_mel_analyzer() {
        let sample_len = consts::FFT_SIZE * 10;
        let mut y = linspace(0., 1., sample_len);
        let mel_spec = mel(&mut y, 0., 1.0);
        let (pad_left, pad_right) = ((consts::FFT_SIZE - consts::HOP_SIZE) / 2, (consts::FFT_SIZE - consts::HOP_SIZE + 1) / 2);
        let expected_frames = ((sample_len + pad_left + pad_right - consts::FFT_SIZE) / consts::HOP_SIZE) + 1;
        assert_eq!(mel_spec.dim(), (128, expected_frames));
        assert!(mel_spec.iter().all(|&x| !x.is_nan()));
    }
}