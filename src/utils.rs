pub mod interp;
pub mod stft;
pub mod parser;
pub mod cache;
pub mod growl;
pub mod mel_basis;
use ndarray::{Array2, ArrayView2, Axis, s};
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
    signal.mapv_inplace(|x| (x * 0.1).max(1e-9).ln()); // Scaling by 0.1 is necessary here: Rust's mel magnitudes are 10x larger for unknown reasons.
}
pub fn interp1d(x: &[f64], y: &Array2<f64>, xi: &[f64]) -> Array2<f64> {
    let n_x = x.len();
    let n_y_rows = y.nrows();
    let n_xi = xi.len();
    let mut result = Array2::zeros((n_y_rows, n_xi));
    if n_x <= 1 {
        let fill_col = y.column(0); 
        result.slice_mut(s![.., ..]).assign(&fill_col.insert_axis(Axis(1)));
        return result;
    }
    let x0 = x[0];
    let x_end = x.last().copied().unwrap();
    for (i, &xi_val) in xi.iter().enumerate() {
        let col = match () {
            _ if xi_val.is_nan() || xi_val.is_infinite() => {
                result.column_mut(i).fill(0.0);
                continue;
            }
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
    if n_cols == 0 || pad_size == 0 {
        return arr.to_owned();
    }
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
pub fn reflect_pad_1d(signal: &[f64], pad_left: usize, pad_right: usize) -> Vec<f64> {
    let len = signal.len();
    match len {
        0 => vec![0.0; pad_left + pad_right],
        1 => {
            let val = signal[0];
            std::iter::repeat(val)
                .take(pad_left)
                .chain(signal.iter().cloned())
                .chain(std::iter::repeat(val).take(pad_right))
                .collect()
        }
        _ => {
            let reflect_left = (0..pad_left)
                .map(|i| {
                    let mirror_idx = 1 + (i % (len - 1));
                    signal[if mirror_idx >= len { 2 * len - mirror_idx - 1 } else { mirror_idx }]
                });
            let reflect_right = (0..pad_right)
                .map(|i| {
                    let mirror_idx = (len - 2) - (i % (len - 1));
                    signal[mirror_idx.max(0)]
                });
            reflect_left
                .chain(signal.iter().cloned())
                .chain(reflect_right)
                .collect()
        }
    }
}
