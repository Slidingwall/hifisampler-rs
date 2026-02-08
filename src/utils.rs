pub mod interp;
pub mod stft;
pub mod parser;
pub mod cache;
pub mod growl;
pub mod mel;
mod mel_basis;
use ndarray::{Array2, ArrayView2, Axis, azip, s};
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
pub fn dynamic_range_compression(s: &mut Array2<f64>) {
    s.mapv_inplace(|x| x.max(1e-9).ln());
}
pub fn interp1d(x: &[f64], y: &Array2<f64>, xi: &[f64]) -> Array2<f64> {
    let (n_r, n_xi) = (y.nrows(), xi.len());
    let mut res = Array2::zeros((n_r, n_xi));
    let (y_col0, y_col_e) = (y.column(0), y.column(x.len() - 1));
    let (x_first, x_last) = (x[0], *x.last().unwrap());
    azip!((mut res_col in res.axis_iter_mut(Axis(1)), &xi_val in xi) {
        if xi_val >= x_last - EPSILON {
            res_col.assign(&y_col_e);
        } else if xi_val <= x_first + EPSILON {
            res_col.assign(&y_col0);
        } else {
            let idx = x.binary_search_by(|&p| p.partial_cmp(&xi_val).unwrap_or(Ordering::Greater))
                .unwrap_or_else(|i| i.saturating_sub(1))
                .clamp(0, x.len() - 2);
            let t = if (x[idx+1] - x[idx]).abs() < EPSILON { 0.0 } else { (xi_val - x[idx]) / (x[idx+1] - x[idx]) };
            azip!((
                res in &mut res_col,
                &y0 in &y.column(idx),
                &y1 in &y.column(idx + 1)
            ) {
                *res = lerp(y0, y1, t);
            });
        }
    });
    res
}
pub fn reflect_pad_2d(arr: ArrayView2<f64>, pad: usize) -> Array2<f64> {
    let (n_rows, n_cols) = arr.dim(); 
    let mut pad_arr = Array2::zeros((n_rows, n_cols + pad)); 
    azip!((
        mut pad_row in pad_arr.axis_iter_mut(Axis(0)),
        arr_row in arr.axis_iter(Axis(0))
    ) {
        pad_row.slice_mut(s![0..n_cols]).assign(&arr_row);
    });
    let ref_len = if n_cols > 1 { n_cols - 1 } else { 1 };
    pad_arr.axis_iter_mut(Axis(1))
        .enumerate()
        .for_each(|(col_idx, mut pad_col)| {
            if col_idx >= n_cols {
                pad_col.assign(&arr.column((n_cols - 2).saturating_sub((col_idx - n_cols) % ref_len)))
            }
        });
    pad_arr
}
pub fn reflect_pad_1d(s: &mut Vec<f64>, left: usize, right: usize) {
    let len = s.len();
    s.reserve(left + right);
    s.resize(left + len + right, 0.0);
    s.copy_within(0..len, left);
    for i in 0..left {
        let m_idx = 1 + (i % (len - 1));
        s[i] = s[left + m_idx];
    }
    for i in 0..right {
        let m_idx = (len - 2) - (i % (len - 1));
        s[left + len + i] = s[left + m_idx];
    }
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