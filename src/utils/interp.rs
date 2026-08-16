use ndarray::{Array2, Axis, azip};
use std::f32::consts::PI;
pub fn akima(y: &[f32], xi: &[f32]) -> Vec<f32> {
    let n = y.len();
    let mut out = Vec::with_capacity(xi.len());
    if n == 0 {
        out.resize(xi.len(), 0.0);
        return out;
    }
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
    let slope = |idx: i32| -> f32 {
        if idx < 0 {
            left_extrap
        } else if idx >= (n - 1) as i32 {
            right_extrap
        } else {
            y[idx as usize + 1] - y[idx as usize]
        }
    };
    let mut m = vec![0.0; n];
    for i in 0..n {
        let i32 = i as i32;
        let s0 = slope(i32 - 2);
        let s1 = slope(i32 - 1);
        let s2 = slope(i32);
        let s3 = slope(i32 + 1);
        let w1 = (s3 - s2).abs();
        let w2 = (s1 - s0).abs();
        m[i] = if w1 + w2 < 1e-12 {
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
        let frac = xv - col_idx as f32;
        let y0 = y.column(col_idx);
        let y1 = y.column(col_idx + 1);
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
    let output_len = out.len_of(interp_axis) as isize;
    let iter_axis = Axis(1 - interp_axis.0);
    azip!((mut out_slice in out.axis_iter_mut(iter_axis), in_slice in input.axis_iter(iter_axis)) {
        for i in 0..output_len as usize {
            let (idx, frac) = get_pos(i);
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for t in -3..=3 {
                let pos = idx + t;
                if pos >= 0 && pos < input_len {
                    let x = t as f32 - frac;
                    let weight = if x == 0.0 {
                        1.0
                    } else if x.abs() < 3.0 {
                        let pix = PI * x;
                        (pix.sin() * (pix / 3.0).sin()) / (pix * pix)
                    } else {
                        0.0
                    };
                    let val = (in_slice[pos as usize] + 1e-9).ln();
                    sum += val * weight;
                    weight_sum += weight;
                }
            }
            out_slice[i] = if weight_sum > 1e-9 { sum / weight_sum } else { 0.0 };
        }
    });
    out
}