use std::path::PathBuf;
use ort::{ session::{Session, builder::GraphOptimizationLevel}, value::Value };
use ndarray::{Array2, Array4, azip};
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
        let orig_len = wave.len();
        let total_pad = SEG_LENGTH * (((orig_len + HOP_SIZE - 1) / SEG_LENGTH) + 1) - (orig_len + HOP_SIZE); 
        let left = (total_pad / 2 / HOP_SIZE) * HOP_SIZE; 
        let right = total_pad - left;
        let mut x_pad = Vec::with_capacity(orig_len + total_pad);
        x_pad.extend(std::iter::repeat(0.0).take(left));
        x_pad.extend_from_slice(wave);
        x_pad.extend(std::iter::repeat(0.0).take(right));
        let spec = stft_core(&x_pad, FFT_SIZE, HOP_SIZE);
        let t_spec = spec.ncols();
        let (real, imag): (Vec<f32>, Vec<f32>) = spec
            .iter() 
            .map(|&c| {
                (c.re as f32, c.im as f32)
            })
            .unzip();
        let target_t_spec = ((t_spec + 15) / 16) * 16;
        let mut arr4 = Array4::from_elem((1, 2, OUTPUT_BIN, target_t_spec), 0.0f32);
        azip!((index (_, c, f, t), val in &mut arr4) {
            if t < t_spec {
                *val = match c {
                    0 => real[f + t * OUTPUT_BIN],
                    1 => imag[f + t * OUTPUT_BIN],
                    _ => 0.0,
                };
            }
        });
        let input_value = Value::from_array(
            (
                [1, 2, OUTPUT_BIN as i64, target_t_spec as i64],
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
        azip!((index (f, t), sc_val in &mut spec_corrected, &s_val in &spec) {
            let re = output_data[f + t * OUTPUT_BIN] as f64;
            let im = output_data[OUTPUT_BIN * target_t_spec + f + t * OUTPUT_BIN] as f64;
            *sc_val = Complex::new(
                s_val.re * re - s_val.im * im,
                s_val.re * im + s_val.im * re
            );
        });
        let mut x_pred_pad = istft_core(
            &spec_corrected,
            (t_spec - 1) * HOP_SIZE + FFT_SIZE,
            FFT_SIZE,
            HOP_SIZE,
        );
        x_pred_pad.drain(0..left);
        x_pred_pad.truncate(orig_len);
        x_pred_pad
    }
}
