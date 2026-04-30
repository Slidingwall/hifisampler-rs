use std::path::PathBuf;
use ort::{ session::{Session, builder::GraphOptimizationLevel}, value::Value };
use ndarray::{Array2, Array4, ArrayView3, s};
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
    pub fn run(&mut self, spec: &(Array2<f32>, Array2<f32>)) -> (Array2<f32>, Array2<f32>) {
        let (real_in, imag_in) = spec;
        let (bins, frames) = real_in.dim();
        let mut input = Array4::zeros((1, 2, OUTPUT_BIN, ((frames + 15) / 16) * 16));
        input.slice_mut(s![0, 0, .., ..frames]).assign(real_in);
        input.slice_mut(s![0, 1, .., ..frames]).assign(imag_in);
        let output = self.session.run(vec![("input", Value::from_array(input).unwrap())]).unwrap();
        let view = ArrayView3::from_shape((2, OUTPUT_BIN, ((frames + 15) / 16) * 16), output.get("output").unwrap().try_extract_tensor::<f32>().unwrap().1).unwrap();
        let mut real_out = Array2::zeros((bins, frames));
        let mut imag_out = Array2::zeros((bins, frames));
        real_out.assign(&view.slice(s![0, .., ..frames]));
        imag_out.assign(&view.slice(s![1, .., ..frames]));
        (real_out, imag_out)
    }
}