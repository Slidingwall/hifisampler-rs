use crate::{
    consts::{FFT_SIZE, ORIGIN_HOP_SIZE},
    utils::{mel_basis::MEL_BASIS_DATA, reflect_pad_1d, stft::stft_core},
};
use ndarray::{Array2, ArrayView1, Axis, azip, concatenate, s};
const TARGET_BINS: usize = FFT_SIZE / 2 + 1;
pub fn mel(wave: &mut Vec<f32>, key_shift: f32, speed: f32) -> Array2<f32> {
    let fft_size = (FFT_SIZE as f32 * 2f32.powf(key_shift / 12.0)).round() as usize;
    let hop_len = (ORIGIN_HOP_SIZE as f32 * speed).round() as usize;
    let scale = FFT_SIZE as f32 / fft_size as f32;
    reflect_pad_1d(wave, (fft_size - hop_len) / 2, (fft_size - hop_len + 1) / 2);
    let comp_spec = stft_core(&wave, fft_size, hop_len);
    let (freq_bins, n_frames) = (comp_spec.nrows(), comp_spec.ncols());
    let mut spec = Array2::zeros((freq_bins, n_frames));
    azip!((spec_elem in &mut spec, comp_elem in &comp_spec) {
        *spec_elem = comp_elem.norm();
    });
    if key_shift != 0. {
        if freq_bins < TARGET_BINS {
            spec = concatenate(Axis(0), &[spec.view(),Array2::zeros((TARGET_BINS - freq_bins, n_frames)).view()]).unwrap();
        } else if freq_bins > TARGET_BINS {
            spec = spec.slice(s![..TARGET_BINS, ..]).to_owned();
        }
        spec.mapv_inplace(|x| x * scale);
    }
    let mut mel_spec = Array2::zeros((128, n_frames));
    let proc_bins = spec.nrows();
    azip!((mut mel_row in mel_spec.axis_iter_mut(Axis(0)), nonzeros in ArrayView1::from(&MEL_BASIS_DATA)) {
        mel_row.iter_mut().enumerate().for_each(|(frame_idx, mel_val)| {
            let sum = nonzeros
                .iter() 
                .filter(|&&(freq_idx, _)| freq_idx < proc_bins) 
                .fold(0.0, |acc, &(freq_idx, weight)| {
                    acc + spec[(freq_idx, frame_idx)] * weight 
                });
            *mel_val = sum;
        });
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