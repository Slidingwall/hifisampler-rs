use ndarray::{Array2, Axis, azip};
use std::{cmp::Ordering, f32::{EPSILON, consts::PI}};
pub fn akima(x: &[f32], y: &[f32], t: &[f32]) -> Vec<f32> {
    let n = y.len();
    let mut out = Vec::with_capacity(t.len());
    match n {
        0 => { out.resize(t.len(), 0.0); return out; }
        1 => { out.resize(t.len(), y[0]); return out; }
        2 => {
            let k = (y[1] - y[0]) / (x[1] - x[0]);
            t.iter().for_each(|&p| out.push(if p <= x[0] { y[0] } else if p >= x[1] { y[1] } else { y[0] + (p - x[0]) * k }));
            return out;
        }
        _ => {}
    }
    let mut m = vec![0.0; n];
    for i in 0..n {
        let w1 = (((if i == n-1 { 2.0 * y[n-1] - y[n-3] } else if i == n-2 { 2.0 * y[n-1] - y[n-2] } else { y[i+2] }) - (if i == n-1 { 2.0 * y[n-1] - y[n-2] } else if i == n-2 { y[n-1] } else { y[i+1] })) - ((if i == n-1 { 2.0 * y[n-1] - y[n-2] } else if i == n-2 { y[n-1] } else { y[i+1] }) - y[i])).abs();
        let w2 = ((y[i] - (if i == 0 { 2.0 * y[1] - y[2] } else if i == 1 { y[0] } else { y[i-1] })) - ((if i == 0 { 2.0 * y[1] - y[2] } else if i == 1 { y[0] } else { y[i-1] }) - (if i == 0 { 2.0 * y[0] - y[2] } else if i == 1 { 2.0 * y[0] - y[1] } else { y[i-2] }))).abs();
        m[i] = if w1 + w2 < 1e-12 {
            ((y[i] - (if i == 0 { 2.0 * y[1] - y[2] } else if i == 1 { y[0] } else { y[i-1] })) + ((if i == n-1 { 2.0 * y[n-1] - y[n-2] } else if i == n-2 { y[n-1] } else { y[i+1] }) - y[i])) * 0.5
        } else {
            (w1 * (y[i] - (if i == 0 { 2.0 * y[1] - y[2] } else if i == 1 { y[0] } else { y[i-1] })) + w2 * ((if i == n-1 { 2.0 * y[n-1] - y[n-2] } else if i == n-2 { y[n-1] } else { y[i+1] }) - y[i])) / (w1 + w2)
        };
    }
    let coeffs: Vec<_> = (0..n-1).map(|i| (y[i], m[i]*(x[i+1]-x[i]), 3.*(y[i+1]-y[i])-2.*m[i]*(x[i+1]-x[i])-m[i+1]*(x[i+1]-x[i]), 2.*(y[i]-y[i+1])+m[i]*(x[i+1]-x[i])+m[i+1]*(x[i+1]-x[i]), x[i+1]-x[i])).collect();
    let (x0, xn) = (x[0], x[n-1]);
    let mut idx = 0;
    for &p in t {
        if p <= x0 {
            out.push(y[0]);
        } else if p >= xn {
            out.push(y[n-1]);
        } else {
            while idx + 1 < x.len() && x[idx + 1] < p { idx += 1; }
            let (c0, c1, c2, c3, dx) = coeffs[idx];
            out.push(c0 + ((p - x[idx])/dx) * (c1 + ((p - x[idx])/dx) * (c2 + ((p - x[idx])/dx) * c3)));
        }
    }
    out
}
pub fn interp1d(x: &[f32], y: &Array2<f32>, xi: &[f32]) -> Array2<f32> {
    let mut res = Array2::zeros((y.nrows(), xi.len()));
    let n_rows = y.nrows();
    let n_cols = y.ncols();
    if n_cols == 0 { return res; }
    let x_first = x[0];
    let x_last = x[n_cols - 1];
    azip!((mut col in res.axis_iter_mut(Axis(1)), &xv in xi) {
        if xv >= x_last { col.assign(&y.column(n_cols - 1)); return; }
        if xv <= x_first { col.assign(&y.column(0)); return; }
        let i = x.binary_search_by(|&v| v.partial_cmp(&xv).unwrap_or(Ordering::Greater))
            .unwrap_or_else(|i| i.saturating_sub(1))
            .clamp(0, n_cols - 2);
        let dx = x[i + 1] - x[i];
        let t = if dx.abs() < EPSILON { 0.0 } else { (xv - x[i]) / dx };
        let y0 = y.column(i);
        let y1 = y.column(i + 1);
        for r in 0..n_rows {
            col[r] = y0[r] + (y1[r] - y0[r]) * t;
        }
    });
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
                        (PI * x).sin() * (PI * x / 3.0).sin() / (PI * PI * x * x / 3.0)
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