pub mod interp;
pub mod stft;
pub mod parser;
pub mod cache;
pub mod growl;
pub mod mel;
mod mel_basis;
use ndarray::{Array2, ArrayView2, Axis, azip, parallel::prelude::*, s};
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
    signal.par_mapv_inplace(|x| x.max(1e-9).ln());
}
pub fn interp1d(x: &[f64], y: &Array2<f64>, xi: &[f64]) -> Array2<f64> {
    let (n_y_rows, n_xi) = (y.nrows(), xi.len());
    let mut result = Array2::zeros((n_y_rows, n_xi));
    let (y_col0, y_col_last) = (y.column(0), y.column(x.len() - 1));
    par_azip!((mut result_col_mut in result.axis_iter_mut(Axis(1)), &xi_val in xi) {
        if xi_val >= *x.last().unwrap() - EPSILON {
            result_col_mut.assign(&y_col_last);
        } else if xi_val <= x[0] + EPSILON {
            result_col_mut.assign(&y_col0);
        } else {
            let idx = x.binary_search_by(|&p| p.partial_cmp(&xi_val).unwrap_or(Ordering::Greater))
                .unwrap_or_else(|i| i.saturating_sub(1))
                .clamp(0, x.len() - 2);
            let t = if (x[idx+1] - x[idx]).abs() < EPSILON { 0.0 } else { (xi_val - x[idx]) / (x[idx+1] - x[idx]) };
            azip!((
                res in &mut result_col_mut,
                &y0 in &y.column(idx),
                &y1 in &y.column(idx + 1)
            ) {
                *res = lerp(y0, y1, t);
            });
        }
    });
    result
}
pub fn reflect_pad_2d(arr: ArrayView2<f64>, pad_size: usize) -> Array2<f64> {
    let (n_rows, n_cols) = arr.dim(); 
    let mut padded = Array2::zeros((n_rows, n_cols + pad_size)); 
    par_azip!((
        mut padded_row in padded.axis_iter_mut(Axis(0)),
        arr_row in arr.axis_iter(Axis(0))
    ) {
        padded_row.slice_mut(s![0..n_cols]).assign(&arr_row);
    });
    let reflect_len = if n_cols > 1 { n_cols - 1 } else { 1 };
    padded.axis_iter_mut(Axis(1))
        .into_par_iter() 
        .enumerate()
        .for_each(|(col_idx, mut padded_col_mut)| {
            if col_idx >= n_cols {
                padded_col_mut.assign(&arr.column((n_cols - 2).saturating_sub((col_idx - n_cols) % reflect_len)))
            }
        });
    padded
}
pub fn reflect_pad_1d(signal: &mut Vec<f64>, pad_left: usize, pad_right: usize) {
    let len = signal.len();
    signal.reserve(pad_left + pad_right);
    signal.resize(pad_left + len + pad_right, 0.0);
    signal.copy_within(0..len, pad_left);
    for i in 0..pad_left {
        let mirror_idx = 1 + (i % (len - 1));
        signal[i] = signal[pad_left + mirror_idx];
    }
    for i in 0..pad_right {
        let mirror_idx = (len - 2) - (i % (len - 1));
        signal[pad_left + len + i] = signal[pad_left + mirror_idx];
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