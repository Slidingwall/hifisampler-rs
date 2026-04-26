use crate::{
    consts::{FFT_SIZE, ORIGIN_HOP_SIZE},
    utils::{mel_basis::MEL_BASIS_DATA, reflect_pad_1d, stft::stft_core},
};
use ndarray::{Array2, ArrayView1, Axis, azip, concatenate, s};
const TARGET_BINS: usize = FFT_SIZE / 2 + 1;
pub fn mel(wave: &mut Vec<f32>, key_shift: f32, speed: f32) -> Array2<f32> {
    let fft_size = (FFT_SIZE as f32 * 2f32.powf(key_shift / 12.0)).round() as usize;
    reflect_pad_1d(
        wave, 
        (fft_size - (ORIGIN_HOP_SIZE as f32 * speed).round() as usize) / 2, 
        (fft_size - (ORIGIN_HOP_SIZE as f32 * speed).round() as usize + 1) / 2
    );
    let comp_spec = stft_core(wave, fft_size, (ORIGIN_HOP_SIZE as f32 * speed).round() as usize);
    let mut spec = comp_spec.mapv(|c| c.norm());
    if key_shift != 0. {
        spec = match comp_spec.nrows().cmp(&TARGET_BINS) {
            std::cmp::Ordering::Less => concatenate(Axis(0), &[spec.view(), Array2::zeros((TARGET_BINS - comp_spec.nrows(), comp_spec.ncols())).view()]).unwrap(),
            std::cmp::Ordering::Greater => spec.slice(s![..TARGET_BINS, ..]).to_owned(),
            std::cmp::Ordering::Equal => spec,
        };
        spec.mapv_inplace(|x| x * FFT_SIZE as f32 / fft_size as f32);
    }
    let mut mel_spec = Array2::zeros((128, comp_spec.ncols()));
    azip!((mut mel_row in mel_spec.axis_iter_mut(Axis(0)), nonzeros in ArrayView1::from(&MEL_BASIS_DATA)) {
        for frame_idx in 0..comp_spec.ncols() {
            let mut sum = 0.0;
            for &(freq_idx, weight) in *nonzeros {
                sum += spec[(freq_idx, frame_idx)] * weight;
            }
            mel_row[frame_idx] = sum;
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
        let (pad_left, pad_right) = ((FFT_SIZE - ORIGIN_HOP_SIZE) / 2, (FFT_SIZE - ORIGIN_HOP_SIZE) + 1 / 2);
        let expected_frames = ((sample_len + pad_left + pad_right - FFT_SIZE) / ORIGIN_HOP_SIZE) + 1;
        assert_eq!(mel_spec.dim(), (128, expected_frames));
        assert!(mel_spec.iter().all(|&x| !x.is_nan()));
    }
}