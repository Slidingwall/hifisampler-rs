use ndarray::{Array2, Axis, azip};
use std::f32::consts::PI;
use crate::consts::EPSILON;
pub fn akima(y: &[f32], xi: &[f32]) -> Vec<f32> {
    let n = y.len();
    let mut out = Vec::with_capacity(xi.len());
    if n == 1 {
        out.resize(xi.len(), y[0]);
        return out;
    }
    if n == 2 {
        let k = y[1] - y[0];
        for &p in xi {
            out.push(y[0] + p.clamp(0.0, 1.0) * k);
        }
        return out;
    }
    let left_extrap = 3.0 * y[1] - 2.0 * y[0] - y[2];
    let right_extrap = 2.0 * y[n-1] - 3.0 * y[n-2] + y[n-3];
    let mut s = vec![left_extrap; n + 3];
    for (i, &v) in y[..n - 1].iter().enumerate() {
        s[i + 2] = y[i + 1] - v;
    }
    s[n + 1..].fill(right_extrap);
    let mut m = vec![0.0; n];
    for i in 0..n {
        let s0 = s[i];
        let s1 = s[i + 1];
        let s2 = s[i + 2];
        let s3 = s[i + 3];
        let w1 = (s3 - s2).abs();
        let w2 = (s1 - s0).abs();
        m[i] = if w1 + w2 < EPSILON {
            0.5 * (s1 + s2)
        } else {
            (w1 * s1 + w2 * s2) / (w1 + w2)
        };
    }
    let coeffs: Vec<_> = (0..n - 1)
        .map(|i| {
            let dy = y[i + 1] - y[i];
            let m0 = m[i];
            let m1 = m[i + 1];
            (y[i], m0, 3.0 * dy - 2.0 * m0 - m1, -2.0 * dy + m0 + m1)
        })
        .collect();
    let last_idx = (n - 1) as f32;
    let mut seg = 0;
    for &p in xi {
        if p <= 0.0 {
            out.push(y[0]);
        } else if p >= last_idx {
            out.push(y[n - 1]);
        } else {
            while seg + 1 < n - 1 && (seg + 1) as f32 <= p {
                seg += 1;
            }
            let u = p - seg as f32;
            let (c0, c1, c2, c3) = coeffs[seg];
            out.push(c0 + u * (c1 + u * (c2 + u * c3)));
        }
    }
    out
}
pub fn interp1d(y: &Array2<f32>, xi: &[f32]) -> Array2<f32> {
    let n_rows = y.nrows();
    let n_cols = y.ncols();
    let mut res = Array2::zeros((xi.len(), n_rows));
    if n_cols == 0 {
        return res;
    }
    let last_idx = (n_cols - 1) as f32;
    let mut cur_idx: isize = -1;
    let mut y0 = y.column(0);
    let mut y1 = y.column(0);
    for (i, &xv) in xi.iter().enumerate() {
        let mut out_row = res.row_mut(i);
        if xv <= 0.0 {
            out_row.assign(&y.column(0));
            continue;
        }
        if xv >= last_idx {
            out_row.assign(&y.column(n_cols - 1));
            continue;
        }
        let col_idx = xv.floor() as usize;
        if col_idx as isize != cur_idx {
            y0 = y.column(col_idx);
            y1 = y.column(col_idx + 1);
            cur_idx = col_idx as isize;
        }
        let frac = xv - col_idx as f32;
        for r in 0..n_rows {
            out_row[r] = y0[r] + (y1[r] - y0[r]) * frac;
        }
    }
    res
}
pub fn spec_interp(
    input: &Array2<f32>,
    output_shape: (usize, usize),
    interp_axis: Axis,
    get_pos: impl Fn(usize) -> (isize, f32) + Sync + Send,
) -> Array2<f32> {
    let mut out = Array2::zeros(output_shape);
    let input_len = input.len_of(interp_axis) as isize;
    let output_len = out.len_of(interp_axis) as usize;
    let other_axis = Axis(1 - interp_axis.0);
    let other_len = out.len_of(other_axis);
    let mut ln_input = Array2::zeros(input.raw_dim());
    azip!((l in &mut ln_input, &v in input) { *l = (v + EPSILON).ln(); });
    let ln_slice = ln_input.as_slice().unwrap();
    let out_slice = out.as_slice_mut().unwrap();
    let (out_stride_i, out_stride_k) = if interp_axis == Axis(0) {
        (other_len, 1usize)
    } else {
        (1usize, output_len)
    };
    let (ln_stride_pos, ln_stride_k) = if interp_axis == Axis(0) {
        (other_len, 1usize)
    } else {
        (1usize, input.len_of(Axis(1)))
    };
    let mut weights = [0.0f32; 7];
    for i in 0..output_len {
        let (idx, frac) = get_pos(i);
        let mut weight_sum = 0.0;
        for (ti, t) in (-3..=3).enumerate() {
            let pos = idx + t;
            if pos >= 0 && pos < input_len {
                let x = t as f32 - frac;
                weights[ti] = if x == 0.0 {
                    1.0
                } else if x.abs() < 3.0 {
                    let pix = PI * x;
                    (pix.sin() * (pix / 3.0).sin()) / (pix * pix)
                } else {
                    0.0
                };
                weight_sum += weights[ti];
            } else {
                weights[ti] = 0.0;
            }
        }
        if weight_sum > EPSILON {
            let out_base = i * out_stride_i;
            for k in 0..other_len {
                let mut sum = 0.0;
                for (ti, t) in (-3..=3).enumerate() {
                    let pos = idx + t;
                    if pos >= 0 && pos < input_len {
                        sum += ln_slice[pos as usize * ln_stride_pos + k * ln_stride_k] * weights[ti];
                    }
                }
                out_slice[out_base + k * out_stride_k] = sum / weight_sum;
            }
        }
    }
    out
}
