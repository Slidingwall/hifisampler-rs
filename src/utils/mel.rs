use crate::{consts::{FFT_SIZE, EPSILON}, utils::{interp::spec_interp, mel_basis::MEL_BASIS_DATA}};
use ndarray::{Array2, Axis};
pub fn mel(spec:&Array2<f32>,key_shift:f32)->Array2<f32>{
    let (inf, ot) = spec.dim();
    let mut mel_spec = Array2::zeros((128, ot));
    let target_time = ((ot-1)as f32 *4.).round() as usize +1;
    let mut process_mel = |data: &Array2<f32>| {
        let nrows = data.nrows();
        let ot = data.ncols();
        let data_slice = data.as_slice().unwrap();
        let mel_slice = mel_spec.as_slice_mut().unwrap();
        for (b, filter) in MEL_BASIS_DATA.iter().enumerate() {
            let n_valid = filter.iter().take_while(|&&(f, _)| f < nrows).count();
            let base = b * ot;
            for t in 0..ot {
                let mut sum = 0.0;
                for &(f, w) in &filter[..n_valid] {
                    sum += data_slice[f * ot + t] * w;
                }
                mel_slice[base + t] = sum;
            }
        }
    };
    if key_shift.abs() < EPSILON {
        process_mel(spec);
    } else {
        let fs = (-key_shift /12.).exp2();
        let scaled = (FFT_SIZE as f32 * fs).round();
        let tf = scaled as usize /2 +1;
        let factor = inf as f32 / tf as f32;
        let mut sf = spec_interp(spec,(tf.min(743),ot),Axis(0),|f| {
            let x = f as f32 * factor;
            (x.floor() as isize, x.fract())
        });
        sf.iter_mut().for_each(|v| *v = v.exp() * FFT_SIZE as f32 / scaled);
        process_mel(&sf);
    }
    spec_interp(&mel_spec, (128, target_time), Axis(1), |t| {
        let x = t as f32 /4.;
        (x.floor() as isize, x.fract())
    })
}