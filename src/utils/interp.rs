pub struct Akima {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}
impl Akima {
    const EPS: f64 = 1e-9;
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        let n = x.len();
        let mut m = Vec::with_capacity(n + 3);
        m.push(0.0);
        m.push(0.0);
        for i in 0..n-1 {
            m.push((y[i+1] - y[i]) / (x[i+1] - x[i]));
        }
        m[0] = 3.0 * m[2] - 2.0 * m[3];
        m[1] = 2.0 * m[2] - m[3];
        m.push(2.0 * m[m.len() - 1] - m[m.len() - 2]);
        m.push(2.0 * m[m.len() - 1] - m[m.len() - 2]);
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            let i_clamped = i.min(n - 4);
            let m0 = m[i_clamped];
            let m1 = m[i_clamped + 1];
            let m2 = m[i_clamped + 2];
            let m3 = m[i_clamped + 3];
            let w1 = (m3 - m2).abs();
            let w2 = (m1 - m0).abs();
            s.push(if w1 + w2 < Self::EPS {
                (m1 + m2) * 0.5
            } else {
                (w1 * m1 + w2 * m2) / (w1 + w2)
            });
        }
        let mut coeffs = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let dx = x[i+1] - x[i];
            coeffs.push([
                y[i],          
                s[i],
                (3.0 * m[2 + i] - 2.0 * s[i] - s[i+1]) / dx,
                (s[i] + s[i+1] - 2.0 * m[2 + i]) / (dx * dx),
            ]);
        }
        Self {
            x: x.to_vec(),
            coeffs,
        }
    }
    #[inline(always)]
    pub fn sample_with_slice(&self, t_slice: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(t_slice.len());
        let mut i = 0;
        for &t in t_slice {
            while i < self.x.len() - 1 && self.x[i + 1] < t {
                i += 1;
            }
            let [c0, c1, c2, c3] = self.coeffs[i];
            let r = (t - self.x[i]) / (self.x[i+1] - self.x[i]);
            result.push(c0 + r * (c1 + r * (c2 + r * c3)));
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::Akima;
    const Y: [f64; 6] = [1., 2., 4., 2., 3., 2.];
    fn default_x() -> Vec<f64> {
        (0..Y.len()).map(|i| i as f64).collect()
    }
    #[test]
    fn test_akima_interpolation() {
        let x = default_x();
        let interp = Akima::new(&x, &Y);
        let left_result = interp.sample_with_slice(&[0.0])[0];
        let right_result = interp.sample_with_slice(&[5.0])[0];
        let mid_result = interp.sample_with_slice(&[2.5])[0];
        assert_eq!(left_result, Y[0], "Left boundary error");
        assert_eq!(right_result, Y[5], "Right boundary error");
        assert!((mid_result - 3.0).abs() < 1e-6, "Midpoint interpolation error: expected ~3.0, got {}", mid_result);
    }
    #[test]
    fn test_sample_with_vec() {
        let x = default_x();
        let sample_t = vec![1.2, 3.8, 4.5];
        let interp = Akima::new(&x, &Y);
        let result = interp.sample_with_slice(&sample_t);
        assert_eq!(result.len(), sample_t.len(), "Slice sampling length error");
        for val in result {
            assert!(val.is_finite(), "Invalid sample value: {}", val);
        }
    }
    #[test]
    fn test_akima_custom_x() {
        let x = vec![0.0, 1.5, 3.0, 4.5, 6.0];
        let y = vec![1.0, 2.5, 1.0, 3.0, 2.0];
        let interp = Akima::new(&x, &y);
        let sample_result = interp.sample_with_slice(&[0.75])[0];
        let linear_ref = 1.0 + (0.75 / 1.5) * (2.5 - 1.0);
        assert!((sample_result - linear_ref).abs() < 1e-3, "Custom x interpolation error");
    }
}