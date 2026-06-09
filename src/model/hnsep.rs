use std::path::PathBuf;
use ort::{ session::{Session, builder::GraphOptimizationLevel}, value::TensorRef };
use ndarray::{Array2, Array3, ArrayView3, azip, s};
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
    pub fn run(&mut self, spec: &Array3<f32>) -> Array2<f32> {
        let (ch, bins, frames) = spec.dim();
        assert_eq!(ch, 2);
        let outputs = self.session.run(vec![("input", TensorRef::from_array_view(&spec.view().to_shape((1, 2, bins, frames)).unwrap()).unwrap())]).unwrap();
        let output_tensor = outputs.get("output").unwrap().try_extract_tensor::<f32>().unwrap();
        let output_view = ArrayView3::from_shape((2, OUTPUT_BIN, frames), output_tensor.1).unwrap();
        let mut mag_out = Array2::zeros((bins, frames));
        azip!((m in &mut mag_out, r in output_view.slice(s![0, .., ..frames]), i in output_view.slice(s![1, .., ..frames])) {
            *m = r.hypot(*i);
        });
        mag_out
    }
}