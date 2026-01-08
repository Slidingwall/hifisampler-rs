pub struct Akima {
    y: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}
impl Akima {
    pub fn new(y: &[f64]) -> Self {
        let n = y.len();
        if n <= 1 {
            return Self { y: y.to_vec(), coeffs: Vec::new() };
        }
        let mut m = vec![0.0; 2];
        m.extend(y.windows(2).map(|w| w[1] - w[0]));
        m.extend([0.0; 2]);
        let s: Vec<f64> = (0..n)
            .map(|i| {
                let i_clamped = i.min(n - 4); 
                let m_slice = &m[i_clamped..i_clamped + 4];
                let (w1, w2) = ((m_slice[3] - m_slice[2]).abs(), (m_slice[1] - m_slice[0]).abs());
                if w1 + w2 < 1e-9 {
                    (m_slice[1] + m_slice[2]) / 2.0
                } else {
                    (w1 * m_slice[1] + w2 * m_slice[2]) / (w1 + w2)
                }
            })
            .collect();
        let coeffs: Vec<[f64; 4]> = (0..n - 1)
            .map(|i| {
                let m_i = m[2 + i];
                [
                    y[i],                                  
                    s[i],                                  
                    3.0 * m_i - 2.0 * s[i] - s[i + 1],    
                    s[i] + s[i + 1] - 2.0 * m_i            
                ]
            })
            .collect();
        Self { y: y.to_vec(), coeffs }
    }
    #[inline(always)]
    pub fn sample(&self, x: f64) -> f64 {
        let n = self.y.len();
        match n {
            0 => 0.0,
            1 => self.y[0],
            _ => {
                let x_clamped = x.clamp(0.0, (n - 1) as f64);
                let i = x_clamped.floor() as usize;
                let [c0, c1, c2, c3] = self.coeffs[i];
                let r = x_clamped - i as f64;
                ((c3 * r + c2) * r + c1) * r + c0
            }
        }
    }
    #[inline(always)]
    pub fn sample_with_slice(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| self.sample(xi)).collect()
    }
}
#[cfg(test)]
mod tests {
    use super::Akima;
    const X: [f64; 6] = [1., 2., 4., 2., 3., 2.];
    #[test]
    fn test_akima_interpolation() {
        let interp = Akima::new(&X);
        assert_eq!(interp.sample(0.0), X[0], "Left boundary error");
        assert_eq!(interp.sample(5.0), X[5], "Right boundary error");
        assert!((interp.sample(2.5) - 3.0).abs() < 1e-6, "Midpoint interpolation error");
    }
    #[test]
    fn test_sample_with_vec() {
        let x = vec![1.2, 3.8, 4.5];
        let result = Akima::new(&X).sample_with_slice(&x);
        assert_eq!(result.len(), x.len(), "Slice sampling length error");
    }
}