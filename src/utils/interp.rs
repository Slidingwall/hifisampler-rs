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
        (0..n - 1).map(|i| y[i + 1] - y[i]) 
            .for_each(|val| m.push(val));
        m.push(0.0);
        m.push(0.0);
        let s: Vec<f32> = (0..n)
            .map(|i| {
                let w1 = (m[i + 3] - m[i + 2]).abs();
                let w2 = (m[i + 1] - m[i]).abs();
                if w1 + w2 < 1e-9 {
                    (m[i + 1] + m[i + 2]) * 0.5
                } else {
                    (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2)
                }
            })
            .collect();
        let coeffs  = (0..n - 1)
            .map(|i| {
                [
                    y[i],
                    s[i],
                    (3.0 * m[2 + i] - 2.0 * s[i] - s[i + 1]),
                    (s[i] + s[i + 1] - 2.0 * m[2 + i]),
                ]
            })
            .collect();
        Self { len: n, coeffs }
    }
    #[inline(always)]
    pub fn sample_with_slice(&self, x: &[f32]) -> Vec<f32> {
        let mut res = Vec::with_capacity(x.len());
        let mut i = 0;
        let mut i_float = 0.;
        let max_idx = self.len - 1;
        x.iter().for_each(|&t| {
            while i < max_idx && (i_float + 1.) < t {
                i += 1;
                i_float += 1.;
            }
            let [c0, c1, c2, c3] = self.coeffs[i];
            let r = t - i_float;
            res.push(c0 + r * (c1 + r * (c2 + r * c3)));
        });
        res
    }
}