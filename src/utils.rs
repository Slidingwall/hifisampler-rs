pub mod interp;
pub mod stft;
pub mod parser;
pub mod cache;
pub mod growl;
pub mod mel_basis;
use ndarray::{Array2, ArrayView2, s};
use std::{cmp::Ordering, f64::EPSILON};
#[inline(always)]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
#[inline(always)]
pub fn midi_to_hz(x: f64) -> f64 {
    440. * (x / 12. - 5.75).exp2()
}
#[inline(always)]
pub fn dynamic_range_compression(signal: &mut Array2<f64>) {
    signal.mapv_inplace(|x| x.max(1e-9).ln());
}
pub fn interp1d(x: &[f64], y: &Array2<f64>, xi: &[f64]) -> Array2<f64> {
    let n_x = x.len();
    let n_y_rows = y.nrows();
    let n_xi = xi.len();
    let mut result = Array2::zeros((n_y_rows, n_xi));
    let x0 = x[0];
    let x_end = x.last().copied().unwrap();
    for (i, &xi_val) in xi.iter().enumerate() {
        let col = match () {
            _ if xi_val >= x_end - EPSILON => &y.column(n_x - 1),
            _ if xi_val <= x0 + EPSILON => &y.column(0),
            _ => {
                let idx = x.binary_search_by(|&p| p.partial_cmp(&xi_val).unwrap_or(Ordering::Greater))
                    .unwrap_or_else(|i| i.saturating_sub(1))
                    .clamp(0, n_x - 2);
                let dx = x[idx + 1] - x[idx];
                let t = if dx.abs() < EPSILON { 0.0 } else { (xi_val - x[idx]) / dx };
                for j in 0..n_y_rows {
                    result[[j, i]] = lerp(y[[j, idx]], y[[j, idx + 1]], t);
                }
                continue;
            }
        };
        result.column_mut(i).assign(col);
    }
    result
}
pub fn reflect_pad_2d(arr: ArrayView2<f64>, pad_size: usize) -> Array2<f64> {
    let (n_rows, n_cols) = arr.dim();
    let mut padded = Array2::zeros((n_rows, n_cols + pad_size));
    padded.slice_mut(s![.., 0..n_cols]).assign(&arr);
    let reflect_len = if n_cols > 1 { n_cols - 1 } else { 1 };
    (0..pad_size).for_each(|i| {
        let target_col = n_cols + i;
        let mirror_pos = i % reflect_len;
        let reflect_idx = (n_cols - 2).saturating_sub(mirror_pos);
        padded.slice_mut(s![.., target_col]).assign(&arr.slice(s![.., reflect_idx]));
    });
    padded
}
pub fn reflect_pad_1d(signal: &mut Vec<f64>, pad_left: usize, pad_right: usize) {
    let len = signal.len();
    let original_slice = &signal[..];
    let right_pad_data: Vec<f64> = (0..pad_right)
        .map(|i| {
            let mirror_idx = (len - 2) - (i % (len - 1));
            original_slice[mirror_idx.max(0)]
        })
        .collect(); 
    let left_pad_data: Vec<f64> = (0..pad_left)
        .map(|i| {
            let mirror_idx = 1 + (i % (len - 1));
            original_slice[mirror_idx]
        })
        .collect();
    signal.extend(right_pad_data);
    signal.splice(0..0, left_pad_data);
}
#[cfg(test)]
#[inline]
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) / (n - 1) as f64;
            (0..n).map(|i| start + step * i as f64).collect()
        }
    }
}