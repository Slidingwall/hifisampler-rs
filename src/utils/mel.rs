use crate::{
    consts::{FFT_SIZE, ORIGIN_HOP_SIZE},
    utils::{mel_basis::MEL_BASIS_DATA, reflect_pad_1d, stft::stft_core},
};
use ndarray::{Array2, ArrayView1, Axis, azip, s};
const TARGET_BINS: usize = FFT_SIZE / 2 + 1;
pub fn mel(wave: &mut Vec<f64>, key_shift: f64, speed: f64) -> Array2<f64> {
    let fft_size = (FFT_SIZE as f64 * 2f64.powf(key_shift / 12.0)).round() as usize;
    let hop_len = (ORIGIN_HOP_SIZE as f64 * speed).round() as usize;
    let scale = FFT_SIZE as f64 / fft_size as f64;
    reflect_pad_1d(wave, (fft_size - hop_len) / 2, (fft_size - hop_len + 1) / 2);
    let comp_spec = stft_core(&wave, fft_size, hop_len);
    let n_frames = comp_spec.ncols();
    let mut spec = Array2::zeros((comp_spec.nrows(), n_frames));
    azip!((spec_elem in &mut spec, comp_elem in &comp_spec) {
        *spec_elem = comp_elem.norm();
    });
    let proc_spec = if key_shift != 0. {
        let mut target = Array2::zeros((TARGET_BINS, n_frames));
        let src_view = spec.slice(s![..TARGET_BINS, ..]);
        target.slice_mut(s![..src_view.nrows(), ..]).assign(&src_view);
        target.mapv_inplace(|x| x * scale);
        target
    } else {
        spec
    };
    let mut mel_spec = Array2::zeros((128, n_frames));
    azip!((mut mel_row in mel_spec.axis_iter_mut(Axis(0)), nonzeros in ArrayView1::from(&MEL_BASIS_DATA)) {
        for (frame_idx, mel_val) in mel_row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for &(freq_idx, weight) in *nonzeros {
                if freq_idx < proc_spec.nrows() {
                    sum += proc_spec[(freq_idx, frame_idx)] * weight;
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
    use crate::{utils::linspace};
    #[test]
    fn test_mel_analyzer() {
        let sample_len = FFT_SIZE * 10;
        let mut y = linspace(0., 1., sample_len);
        let mel_spec = mel(&mut y, 0., 1.0);
        let (pad_left, pad_right) = ((FFT_SIZE - ORIGIN_HOP_SIZE) / 2, (FFT_SIZE - ORIGIN_HOP_SIZE + 1) / 2);
        let expected_frames = ((sample_len + pad_left + pad_right - FFT_SIZE) / ORIGIN_HOP_SIZE) + 1;
        assert_eq!(mel_spec.dim(), (128, expected_frames));
        assert!(mel_spec.iter().all(|&x| !x.is_nan()));
    }
}