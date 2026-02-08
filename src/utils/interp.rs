pub struct Akima {
    len: usize,
    coeffs: Vec<[f32; 4]>,
}
impl Akima {
    pub fn new(y: &[f32]) -> Self {
        let n = y.len();
        let mut m = Vec::with_capacity(n + 3);
        m.push(0.0);
        m.push(0.0);
        for i in 0..n-1 {
            m.push(y[i+1] - y[i]);
        }
        m.push(0.0);
        m.push(0.0);
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            let w1 = (m[i+3] - m[i+2]).abs();
            let w2 = (m[i+1] - m[i]).abs();
            s.push(if w1 + w2 < 1e-9 {
                (m[i+1] + m[i+2]) * 0.5
            } else {
                (w1 * m[i+1] + w2 * m[i+2]) / (w1 + w2)
            });
        }
        let mut coeffs = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            coeffs.push([
                y[i],          
                s[i],
                (3.0 * m[2 + i] - 2.0 * s[i] - s[i+1]),
                (s[i] + s[i+1] - 2.0 * m[2 + i]),
            ]);
        }
        Self { len: n, coeffs }
    }
    #[inline(always)]
    pub fn sample_with_slice(&self, x: &[f32]) -> Vec<f32> {
        let mut res = Vec::with_capacity(x.len());
        let mut i = 0;
        let mut i_float = 0.;
        let max_idx = self.len - 1;
        for &t in x {
            while i < max_idx && (i_float + 1.) < t {
                i += 1;
                i_float += 1.;
            }
            let [c0, c1, c2, c3] = self.coeffs[i];
            let r = t - i_float;
            res.push(c0 + r * (c1 + r * (c2 + r * c3)));
        }
        res
    }
}