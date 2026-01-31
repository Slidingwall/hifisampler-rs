use std::path::PathBuf;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use ndarray::{Array2, Array3, Array4, ArrayView3, Zip};
use rustfft::num_complex::Complex;
use crate::{consts, utils::stft::*};
const SEG_LENGTH: usize = 32 * consts::HOP_SIZE;
const OUTPUT_BIN: usize = consts::FFT_SIZE / 2 + 1;
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
    pub fn run(&mut self, wave: &[f64]) -> Array3<f64> {
        let original_len = wave.len();
        let tl_pad = ((SEG_LENGTH * (((original_len + consts::HOP_SIZE - 1) / SEG_LENGTH + 1) - 1) 
            - (original_len + consts::HOP_SIZE)) / 2 / consts::HOP_SIZE) * consts::HOP_SIZE;
        let tr_pad = SEG_LENGTH * (((original_len + consts::HOP_SIZE - 1) / SEG_LENGTH + 1)) 
            - (original_len + consts::HOP_SIZE) - tl_pad;
        let mut x_padded = Vec::with_capacity(tl_pad + original_len + tr_pad);
        x_padded.extend(std::iter::repeat(0.0).take(tl_pad));
        x_padded.extend_from_slice(wave);
        x_padded.extend(std::iter::repeat(0.0).take(tr_pad));
        let spec = stft_core(&x_padded, Some(consts::FFT_SIZE), Some(consts::HOP_SIZE));
        let t_spec = spec.ncols();
        let (real, imag): (Vec<_>, Vec<_>) = spec.iter().map(|&c| (c.re as f32, c.im as f32)).unzip();
        let target_t_spec = ((t_spec + 15) / 16) * 16;
        let input_value = Value::from_array(
            (
                [1, 2, OUTPUT_BIN as i64, target_t_spec as i64].to_vec(),
                Array4::from_shape_fn(
                    (1, 2, OUTPUT_BIN, target_t_spec),
                    |(_, c, f, t)| -> f32 {
                        if t < t_spec {
                            match c {
                                0 => real[f + t * OUTPUT_BIN],
                                1 => imag[f + t * OUTPUT_BIN],
                                _ => 0.0,
                            }
                        } else {
                            0.0
                        }
                    },
                ).into_raw_vec_and_offset().0
            ),
        ).unwrap();
        let outputs = self.session.run(vec![("input", input_value)]).unwrap();
        let output_data = outputs.get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .1;
        let mask = Zip::from(
            &Array2::from_shape_fn((OUTPUT_BIN, t_spec), |(f, t)| {
                output_data[f + t * OUTPUT_BIN] as f64
            })
        )
        .and(
            &Array2::from_shape_fn((OUTPUT_BIN, t_spec), |(f, t)| {
                output_data[OUTPUT_BIN * target_t_spec + f + t * OUTPUT_BIN] as f64
            })
        )
        .map_collect(|&re, &im| Complex::new(re, im));
        let x_pred_padded = istft_core(
            &(spec * &mask),
            (t_spec - 1) * consts::HOP_SIZE + consts::FFT_SIZE,
            Some(consts::FFT_SIZE),
            Some(consts::HOP_SIZE),
        );
        ArrayView3::from_shape(
            (1, 1, original_len),
            &x_pred_padded[tl_pad..tl_pad + original_len]
        ).unwrap().to_owned()
    }
}
