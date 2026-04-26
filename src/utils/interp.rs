use ndarray::{Array2, Axis, azip};
use std::{cmp::Ordering, f32::EPSILON};
use crate::utils::lerp;
pub fn akima(x: &[f32], y: &[f32], t: &[f32]) -> Vec<f32> {
    const EPS: f32 = 1e-9;
    let n = x.len();
    let mut out = Vec::with_capacity(t.len());
    if n < 2 {
        out.resize(t.len(), y.first().copied().unwrap_or(0.0));
        return out;
    }
    if n == 2 {
        let k = (y[1] - y[0]) / (x[1] - x[0]);
        out.extend(t.iter().map(|&p| {
            if p <= x[0] { y[0] } else if p >= x[1] { y[1] } else { y[0] + (p - x[0]) * k }
        }));
        return out;
    }
    let slopes: Vec<f32> = (0..n-1).map(|i| (y[i+1] - y[i]) / (x[i+1] - x[i])).collect();
    let mut m_ext = vec![
        2.0 * (2.0 * slopes[0] - slopes[1]) - slopes[0],
        2.0 * slopes[0] - slopes[1]
    ];
    m_ext.extend(&slopes);
    m_ext.extend([2.0 * slopes[n-2] - slopes[n-3], 2.0 * (2.0 * slopes[n-2] - slopes[n-3]) - slopes[n-2]]);
    let mut max_w = 0.0f32;
    for i in 0..n {
        let w1 = (m_ext[i+3] - m_ext[i+2]).abs();
        let w2 = (m_ext[i+1] - m_ext[i]).abs();
        max_w = max_w.max(w1 + w2);
    }
    let t_vals: Vec<f32> = (0..n)
        .map(|i| {
            let (w1, w2) = ((m_ext[i+3]-m_ext[i+2]).abs(), (m_ext[i+1]-m_ext[i]).abs());
            let sum = w1 + w2;
            if sum > EPS * max_w {
                (w1 * m_ext[i+1] + w2 * m_ext[i+2]) / sum
            } else {
                (m_ext[i+3] + m_ext[i]) * 0.5
            }
        })
        .collect();
    let coeffs: Vec<[f32; 5]> = (0..n-1)
        .map(|i| {
            let dx = x[i+1] - x[i];
            let dy = y[i+1] - y[i];
            let b = t_vals[i] * dx;
            [
                y[i], b,
                3.0*dy - 2.0*b - t_vals[i+1]*dx,
                2.0*(y[i]-y[i+1]) + b + t_vals[i+1]*dx,
                dx,
            ]
        })
        .collect();
    let mut idx = 0usize;
    for &p in t {
        if p <= x[0] {
            out.push(y[0]);
            continue;
        }
        if p >= x[n-1] {
            out.push(y[n-1]);
            continue;
        }
        while idx < coeffs.len() && x[idx+1] < p {
            idx += 1;
        }
        let [a, b, c, d, dx] = coeffs[idx];
        let u = (p - x[idx]) / dx;
        out.push(a + u * (b + u * (c + u * d)));
    }
    out
}
pub fn interp1d(x: &[f32], y: &Array2<f32>, xi: &[f32]) -> Array2<f32> {
    let mut res = Array2::zeros((y.nrows(), xi.len()));
    let (y_col0, y_col_e) = (y.column(0), y.column(x.len() - 1));
    azip!((mut res_col in res.axis_iter_mut(Axis(1)), &xi_val in xi) {
        if xi_val >= *x.last().unwrap() - EPSILON {
            res_col.assign(&y_col_e); 
        } else if xi_val <= x[0] + EPSILON {
            res_col.assign(&y_col0); 
        } else {
            let idx = x.binary_search_by(|&p| p.partial_cmp(&xi_val).unwrap_or(Ordering::Greater))
                .unwrap_or_else(|i| i.saturating_sub(1))
                .clamp(0, x.len() - 2);
            let dx = x[idx + 1] - x[idx];
            let t = if dx.abs() < EPSILON {
                0.0
            } else {
                (xi_val - x[idx]) / dx
            };
            let (y0_col, y1_col) = (y.column(idx), y.column(idx + 1));
            azip!((res in &mut res_col, &y0 in &y0_col, &y1 in &y1_col) {
                *res = lerp(y0, y1, t);
            });
        }
    });
    res
}