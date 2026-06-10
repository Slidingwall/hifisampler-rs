use std::path::PathBuf;
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::TensorRef};
use ndarray::{Array2, Array3, ArrayView3};
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
        let padded_frames = (frames + 16 - 1) / 16 * 16;
        let outputs = if padded_frames != frames {
            let mut padded = Array3::zeros((2, bins, padded_frames));
            padded.slice_mut(ndarray::s![.., .., 0..frames]).assign(&spec);
            let padded_view = padded.view().into_shape_with_order((1, 2, bins, padded_frames)).unwrap();
            self.session.run(vec![("input", TensorRef::from_array_view(padded_view).unwrap())]).unwrap()
        } else {
            let view = spec.view().into_shape_with_order((1, 2, bins, frames)).unwrap();
            self.session.run(vec![("input", TensorRef::from_array_view(view).unwrap())]).unwrap()
        };
        let output_tensor = outputs.get("output").unwrap().try_extract_tensor::<f32>().unwrap();
        let output_view = ArrayView3::from_shape((2, OUTPUT_BIN, padded_frames), output_tensor.1).unwrap();
        let mut mag_out = Array2::zeros((bins, frames));
        ndarray::azip!((m in &mut mag_out,
                        r in output_view.slice(ndarray::s![0, .., 0..frames]),
                        i in output_view.slice(ndarray::s![1, .., 0..frames])) {
            *m = r.hypot(*i);
        });
        mag_out
    }
}