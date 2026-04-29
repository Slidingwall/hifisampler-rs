use std::path::PathBuf;
use ort::{ session::{Session, builder::GraphOptimizationLevel}, value::Value };
use ndarray::{Array2, Array4, ArrayView3, azip, s};
use oxifft::Complex;
use crate::consts::FFT_SIZE;
const OUTPUT_BIN: usize = FFT_SIZE / 2 + 1;
#[derive(Debug)]
pub struct HNSEPLoader {
    session: Session,
}
impl HNSEPLoader {
    pub fn new(model_path: &PathBuf) -> Self {
        Self {
            session: Session::builder().unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
                .commit_from_file(model_path).unwrap()
        }
    }
    pub fn run(&mut self, spec: &Array2<Complex<f32>>) -> Array2<Complex<f32>> {
        let (_, t_spec) = spec.dim();
        let target_t_spec = ((t_spec + 15) / 16) * 16;
        let mut input = Array4::from_elem((1, 2, OUTPUT_BIN, target_t_spec), 0.0);
        azip!((index (c, f, t), val in &mut input.slice_mut(s![0, .., .., ..t_spec])) {
            *val = if c as usize == 0 { spec[(f, t)].re } else { spec[(f, t)].im };
        });
        let outputs = self.session.run(vec![("input", Value::from_array(input).unwrap())]).unwrap();
        let output = ArrayView3::from_shape(
            (2, OUTPUT_BIN, target_t_spec),
            outputs.get("output").unwrap().try_extract_tensor::<f32>().unwrap().1
        ).unwrap();
        let mut spec_sep = Array2::default(spec.dim());
        azip!((index (f, t), sc_val in &mut spec_sep) {
            *sc_val *= Complex::new(output[(0, f, t)], output[(1, f, t)]);
        });
        spec_sep
    }
}