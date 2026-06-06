use anyhow::Result;
use ndarray::{Array2, Axis, azip, concatenate, s};
use tracing::info;
use crate::{
    audio::{post_process::{loudness_norm, pre_emphasis_base_tension}, read_audio, write_audio},
    consts::{FFT_SIZE, HIFI_CONFIG, HOP_SIZE, ORIGIN_HOP_SIZE, SAMPLE_RATE},
    model::{get_remover, get_vocoder},
    utils::{cache::CACHE_MANAGER, growl::growl, interp::{akima, interp1d}, mel::mel, midi_to_hz, reflect_pad_2d, stft::stft_core},
};
use crate::server::Arguments;
const SR: f32 = SAMPLE_RATE as f32;
const THOP_ORIGIN: f32 = ORIGIN_HOP_SIZE as f32 / SR;
const THOP_ORIGIN_HALF: f32 = THOP_ORIGIN * 0.5;
const THOP: f32 = HOP_SIZE as f32 / SR;
const THOP_HALF: f32 = THOP * 0.5;
fn get_features(args: &Arguments) -> Result<(Array2<f32>, f32)> {
    let breath = args.flags.get("Hb").copied().flatten().unwrap_or(100.0);
    let voicing = args.flags.get("Hv").copied().flatten().unwrap_or(100.0);
    let tension = args.flags.get("Ht").copied().flatten().unwrap_or(0.0);
    let gender = args.flags.get("g").copied().flatten().unwrap_or(0.0);
    let fname = args.in_file.file_stem().unwrap().to_str().unwrap();
    let features_path = args.in_file.with_file_name(format!("{fname}_Hb{breath}Hv{voicing}Ht{tension}g{gender}.hifi.bin"));
    let ignore_cache = args.flags.contains_key("G");
    if let Some(feats) = CACHE_MANAGER.load_features_cache(&features_path, ignore_cache) { return Ok(feats); }
    info!("Generating features: {}", features_path.display());
    let wave = read_audio(&args.in_file)?;
    let spec_mix = stft_core(&wave, FFT_SIZE, HOP_SIZE);
    let dim = spec_mix.0.dim();
    let mut spec_amp = if tension != 0.0 || breath != voicing {
        let (bre, voi) = (breath.clamp(0.0,500.0)*0.01, voicing.clamp(0.0,150.0)*0.01);
        let seg = CACHE_MANAGER.load_hnsep_cache(&args.in_file.with_file_name(format!("{fname}.hnsep.bin")), ignore_cache)
            .unwrap_or_else(||{let s=get_remover().lock().unwrap().run(&spec_mix);CACHE_MANAGER.save_hnsep_cache(&args.in_file.with_file_name(format!("{fname}.hnsep.bin")),&s);s});
        let mut tensed = Array2::zeros(seg.dim());
        azip!((t in &mut tensed, sm in &seg) {*t = sm * voi;});
        if tension != 0.0 { pre_emphasis_base_tension(&mut tensed, -tension.clamp(-100.0,100.0)*0.02); }
        let mut out = Array2::zeros(dim);
        azip!((o in &mut out, cr in &spec_mix.0, ci in &spec_mix.1, sm in &seg, t in &tensed) {
            let mix_mag = cr.hypot(*ci);
            *o = (bre * (mix_mag - sm) + t).abs();
        });
        out
    } else {
        let mut out = Array2::zeros(dim);
        azip!((o in &mut out, r in &spec_mix.0, i in &spec_mix.1) {*o = r.hypot(*i) * breath*0.01;});
        out
    };
    let scale = 512f32.max(spec_amp.iter().fold(0.0, |m, &x| m.max(x))).recip() * 512.0;
    spec_amp.mapv_inplace(|x| x * scale);
    let features = (mel(&spec_amp, gender.clamp(-600.0,600.0)*0.01), scale);
    CACHE_MANAGER.save_features_cache(&features_path, &features);
    Ok(features)
}
pub fn resample(args: Arguments) -> Result<()> {
    let (mut mel_origin, scale) = get_features(&args)?;
    if args.out_file.as_os_str() == "nul" { info!("Null output file - skipping write"); return Ok(()); }
    info!("Modulation: {:.1}, Scale: {:.1}, Mel shape: {:?}", args.modulation, scale, mel_origin.dim());
    let mut t_area_origin = (0..mel_origin.ncols()).map(|i| i as f32 * THOP_ORIGIN + THOP_ORIGIN_HALF).collect::<Vec<f32>>();
    let mut t_total = t_area_origin.last().copied().unwrap() + THOP_ORIGIN_HALF;
    let vel = 2.0f32.powf(1.0 - args.velocity);
    let end = if args.cutoff < 0.0 { args.offset - args.cutoff } else { t_total - args.cutoff };
    let (con, length_req) = (args.offset + args.consonant, args.length);
    let mut stretch_len = end - con;
    info!("Time params: start={:.4}, end={:.4}, con={:.4}, stretch_len={:.4}, length_req={:.4}", args.offset, end, con, stretch_len, length_req);
    if HIFI_CONFIG.loop_mode || args.flags.contains_key("He") {
        info!("Enabling loop mode");
        let start_idx = ((con + THOP_ORIGIN_HALF) / THOP_ORIGIN).floor() as usize;
        let end_idx = ((end + THOP_ORIGIN_HALF) / THOP_ORIGIN).floor() as usize;
        let pad_size = ((length_req / THOP_ORIGIN).floor() as usize) + 1;
        mel_origin = concatenate![Axis(1),mel_origin.slice(s![.., ..start_idx]),reflect_pad_2d(mel_origin.slice(s![.., start_idx..end_idx]), pad_size)];
        stretch_len = pad_size as f32 * THOP_ORIGIN;
        t_area_origin = (0..mel_origin.ncols()).map(|i| i as f32 * THOP_ORIGIN + THOP_ORIGIN_HALF).collect::<Vec<f32>>();
        t_total = t_area_origin.last().unwrap() + THOP_ORIGIN_HALF;
        info!("new_total_time: {}", t_total);
    }
    let scal_ratio = if stretch_len < length_req { length_req / stretch_len } else { 1.0 };
    let stretch = |t: f32| if t < vel * con { t / vel } else { con + (t - vel * con) / scal_ratio };
    let stretched_frames = ((con * vel + (t_total - con) * scal_ratio) / THOP).floor() as usize + 1;
    let mut stretched_t_mel = (0..stretched_frames).map(|i| i as f32 * THOP + THOP_HALF).collect::<Vec<f32>>();
    let cut_left = ((args.offset * vel + THOP_HALF) / THOP).floor() as usize;
    let cut_left = cut_left.saturating_sub(HIFI_CONFIG.fill);
    let cut_right_frame = ((length_req + con * vel + THOP_HALF) / THOP).floor() as usize;
    let cut_right = (stretched_frames - cut_right_frame).saturating_sub(HIFI_CONFIG.fill);
    stretched_t_mel.truncate(stretched_t_mel.len() - cut_right);
    stretched_t_mel.drain(..cut_left);
    let stretch_t_mel = stretched_t_mel.iter().map(|&t| stretch(t).clamp(0.0, *t_area_origin.last().unwrap())).collect::<Vec<f32>>();
    info!("Stretched time axis length: {}", stretch_t_mel.len());
    let mel_render = interp1d(&t_area_origin, &mel_origin, &stretch_t_mel);
    let t = (0..mel_render.ncols()).map(|i| i as f32 * THOP).collect::<Vec<f32>>();
    let mut pitch = args.pitchbend.iter().map(|&pb| pb + args.pitch).collect::<Vec<f32>>();
    if let Some(&t_flag) = args.flags.get("t").and_then(|x| x.as_ref()) {pitch.iter_mut().for_each(|p| *p += t_flag * 0.01);}
    let (new_start, new_end) = (args.offset * vel - cut_left as f32 * THOP, length_req + con * vel - cut_left as f32 * THOP);
    let t_pitch = (0..pitch.len()).map(|i| 60.0 * i as f32 / args.tempo + new_start).collect::<Vec<f32>>();
    let pitch_clamped = t.iter().map(|&x| x.clamp(new_start, *t_pitch.last().unwrap())).collect::<Vec<f32>>();
    let pitch_render = akima(&t_pitch, &pitch, &pitch_clamped);
    let f0_render = pitch_render.iter().map(|&x| midi_to_hz(x)).collect::<Vec<f32>>();
    let wav_con = get_vocoder().lock().unwrap().run(mel_render, f0_render);
    let mut render = wav_con[(new_start * SR).max(0.0) as usize..(new_end * SR).min(wav_con.len() as f32) as usize].to_vec();
    if let Some(&a_flag) = args.flags.get("A").and_then(|x| x.as_ref()) {
        let a_clamped = a_flag.clamp(-100.0, 100.0);
        let mut pitch_derivative = vec![0.0; pitch_render.len()];
        let n = pitch_render.len();
        if n > 1 {
            pitch_derivative[0] = (pitch_render[1] - pitch_render[0]) / (t[1] - t[0]);
            (1..n-1).for_each(|i| pitch_derivative[i] = (pitch_render[i+1] - pitch_render[i-1]) / (t[i+1] - t[i-1]));
            pitch_derivative[n-1] = (pitch_render[n-1] - pitch_render[n-2]) / (t[n-1] - t[n-2]);
        }
        let gain_arr = Array2::from_shape_vec((1, pitch_derivative.len()), pitch_derivative.into_iter().map(|d| 5.0f32.powf(1e-4 * a_clamped * d)).collect()).unwrap();
        let time_vec = (0..render.len()).map(|i| new_start + (new_end - new_start) * i as f32 / render.len() as f32).collect::<Vec<_>>();
        azip!((s in &mut render, g in interp1d(&t, &gain_arr, &time_vec).row(0)) { *s *= g; });
    }
    render.iter_mut().for_each(|x| *x /= scale);
    let new_max = render.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    if let Some(&hg) = args.flags.get("HG").and_then(|x| x.as_ref()) { growl(&mut render, SR, 80.0, hg.clamp(-100.0, 100.0) * 0.01); }
    if HIFI_CONFIG.wave_norm { loudness_norm(&mut render, SR, -16.0, args.flags.get("P").and_then(|x| x.as_ref()).map_or(100, |&p| p.clamp(-100.0, 100.0) as u8)); }
    let mult = (if new_max > HIFI_CONFIG.peak_limit { HIFI_CONFIG.peak_limit/new_max } else {1.0}) * args.volume;
    render.iter_mut().for_each(|x| *x *= mult);
    write_audio(&args.out_file, &render)?;
    info!("Successfully processed: {} -> {}", args.in_file.display(), args.out_file.display());
    Ok(())
}