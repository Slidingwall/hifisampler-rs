use std::path::PathBuf;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use ndarray::{Array2, Array4, parallel::prelude::*};
use oxifft::Complex;
use crate::{consts::{FFT_SIZE, HOP_SIZE}, utils::stft::*};
const SEG_LENGTH: usize = 32 * HOP_SIZE;
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
    pub fn run(&mut self, wave: &[f64]) -> Vec<f64> {
        let original_len = wave.len();
        let tl_pad = ((SEG_LENGTH * (((original_len + HOP_SIZE - 1) / SEG_LENGTH + 1) - 1) 
            - (original_len + HOP_SIZE)) / 2 / HOP_SIZE) * HOP_SIZE;
        let tr_pad = SEG_LENGTH * (((original_len + HOP_SIZE - 1) / SEG_LENGTH + 1)) 
            - (original_len + HOP_SIZE) - tl_pad;
        let mut x_padded = Vec::with_capacity(tl_pad + original_len + tr_pad);
        x_padded.extend(std::iter::repeat(0.0).take(tl_pad));
        x_padded.extend_from_slice(wave);
        x_padded.extend(std::iter::repeat(0.0).take(tr_pad));
        let spec = stft_core(&x_padded, FFT_SIZE, HOP_SIZE);
        let t_spec = spec.ncols();
        let (real, imag): (Vec<f32>, Vec<f32>) = spec
            .par_iter() 
            .map(|&c| {
                (c.re as f32, c.im as f32)
            })
            .unzip();
        let target_t_spec = ((t_spec + 15) / 16) * 16;
        let mut arr4 = Array4::from_elem((1, 2, OUTPUT_BIN, target_t_spec), 0.0f32);
        par_azip!((index (_, c, f, t), val in &mut arr4) {
            if t < t_spec {
                *val = match c {
                    0 => real[f + t * OUTPUT_BIN],
                    1 => imag[f + t * OUTPUT_BIN],
                    _ => 0.0,
                };
            } else {
                *val = 0.0;
            }
        });
        let input_value = Value::from_array(
            (
                vec![1, 2, OUTPUT_BIN as i64, target_t_spec as i64],
                arr4.into_raw_vec_and_offset().0
            ),
        ).unwrap();
        let outputs = self.session.run(vec![("input", input_value)]).unwrap();
        let output_data = outputs.get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .1;
        let mut spec_corrected = Array2::from_elem(spec.dim(), Complex::zero());
        par_azip!((index (f, t), sc_val in &mut spec_corrected, &s_val in &spec) {
            let re = output_data[f + t * OUTPUT_BIN] as f64;
            let im = output_data[OUTPUT_BIN * target_t_spec + f + t * OUTPUT_BIN] as f64;
            *sc_val = Complex::new(
                s_val.re * re - s_val.im * im,
                s_val.re * im + s_val.im * re
            );
        });
        let mut x_pred_padded = istft_core(
            &spec_corrected,
            (t_spec - 1) * HOP_SIZE + FFT_SIZE,
            FFT_SIZE,
            HOP_SIZE,
        );
        x_pred_padded.drain(0..tl_pad);
        x_pred_padded.truncate(original_len);
        x_pred_padded
    }
}
