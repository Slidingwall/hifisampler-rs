use crate::{consts::FFT_SIZE, utils::{interp::spec_interp, mel_basis::MEL_BASIS_DATA}};
use ndarray::{Array2, Axis, azip};
pub fn mel(spec:&Array2<f32>,key_shift:f32)->Array2<f32>{
    let (inf, ot) = spec.dim();
    let mut mel_spec = Array2::zeros((128, ot));
    let target_time = ((ot-1)as f32 *4.).round() as usize +1;
    let mut process_mel = |data: &Array2<f32>| {
        azip!((mut row in mel_spec.axis_iter_mut(Axis(0)), filter in &MEL_BASIS_DATA) {
            for (t, val) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for &(f, w) in *filter { if f < data.nrows() { sum += data[(f, t)] * w; } }
                *val = sum;
            }
        });
    };
    if key_shift.abs() < 1e-6 {
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