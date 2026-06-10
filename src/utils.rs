pub mod interp;
pub mod stft;
pub mod parser;
pub mod cache;
pub mod growl;
pub mod mel;
mod hann_window;
mod mel_basis;
use ndarray::{Array2, ArrayView2, s};
#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}
#[inline(always)]
pub fn midi_to_hz(x: f32) -> f32 {
    (x / 12.0).exp2() * 8.1757989156437073336828122976033
}
pub fn reflect_pad_2d(arr: ArrayView2<f32>, pad: usize) -> Array2<f32> {
    let (rows, n) = arr.dim();
    let mut out = Array2::zeros((rows, n + pad));
    out.slice_mut(s![.., 0..n]).assign(&arr);
    if n == 1 {
        for i in 0..pad {
            out.slice_mut(s![.., n + i]).assign(&arr.slice(s![.., 0]));
        }
        return out;
    }
    let period = 2 * (n - 1);
    for i in 0..pad {
        let r = (n + i) % period;
        out.slice_mut(s![.., n + i])
            .assign(&arr.slice(s![.., if r < period - r { r } else { period - r }]));
    }
    out
}
pub fn reflect_pad_1d(s: &mut Vec<f32>, left: usize, right: usize) {
    let len = s.len();
    let len_1 = len - 1;
    let len_2 = len - 2;
    s.reserve(left + right);
    s.resize(left + len + right, 0.0);
    s.copy_within(0..len, left);
    (0..left).for_each(|i| {
        let m_idx = 1 + (i % len_1);
        s[i] = s[left + m_idx];
    });
    (0..right).for_each(|i| {
        let m_idx = len_2 - (i % len_1);
        s[left + len + i] = s[left + m_idx];
    });
}