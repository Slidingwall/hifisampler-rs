use anyhow::Result;
use ndarray::{Array2, Axis, azip, concatenate, s};
use tracing::info;
use crate::{
    audio::{post_process::{loudness_norm, pre_emphasis_base_tension}, read_audio, write_audio},
    consts::{HIFI_CONFIG, HOP_SIZE, ORIGIN_HOP_SIZE, SAMPLE_RATE},
    model::{get_remover, get_vocoder},
    utils::{cache::CACHE_MANAGER, growl::growl, interp::{akima, interp1d}, mel::mel, midi_to_hz, reflect_pad_2d, stft::stft_core},
};
use crate::server::Arguments;
const SR: f32 = SAMPLE_RATE as f32;
const THOP_ORIGIN: f32 = ORIGIN_HOP_SIZE as f32 / SR;
const THOP: f32 = HOP_SIZE as f32 / SR;
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
    let spec_mix = stft_core(&wave); 
    let (_, freq_bins, frames) = spec_mix.dim(); 
    let mut spec_amp = Array2::zeros((freq_bins, frames));
    if tension != 0.0 || breath != voicing {
        let (bre, voi) = (breath.clamp(0.0,500.0)*0.01, voicing.clamp(0.0,150.0)*0.01);
        let seg = CACHE_MANAGER.load_hnsep_cache(&args.in_file.with_file_name(format!("{fname}.hnsep.bin")), ignore_cache)
            .unwrap_or_else(|| {
                let s = get_remover().lock().unwrap().run(&spec_mix);
                CACHE_MANAGER.save_hnsep_cache(&args.in_file.with_file_name(format!("{fname}.hnsep.bin")), &s);
                s
            });
        let mut tensed = Array2::zeros(seg.dim());
        azip!((t in &mut tensed, sm in &seg) { *t = sm * voi; });
        if tension != 0.0 {
            pre_emphasis_base_tension(&mut tensed, -tension.clamp(-100.0,100.0)*0.02);
        }
        azip!((o in &mut spec_amp, &r in spec_mix.slice(s![0, .., ..]), &i in spec_mix.slice(s![1, .., ..]), sm in &seg, t in &tensed) {
            let mix_mag = r.hypot(i);
            *o = (bre * (mix_mag - sm) + t).abs();
        });
    } else {
        let factor = breath * 0.01;
        azip!((o in &mut spec_amp, &r in spec_mix.slice(s![0, .., ..]), &i in spec_mix.slice(s![1, .., ..])) {
            *o = r.hypot(i) * factor;
        });
    }
    let scale = 512f32.max(spec_amp.iter().fold(0.0, |m, &x| m.max(x))).recip() * 512.0;
    spec_amp.mapv_inplace(|x| x * scale);
    let features = (mel(&spec_amp, gender.clamp(-600.0,600.0)*0.01), scale);
    CACHE_MANAGER.save_features_cache(&features_path, &features);
    Ok(features)
}
pub fn resample(args: Arguments) -> Result<()> {
    let (mut mel_origin, scale) = get_features(&args)?;
    if args.out_file.as_os_str() == "nul" {
        info!("Null output file - skipping write");
        return Ok(());
    }
    info!("Modulation: {:.1}, Scale: {:.1}, Mel shape: {:?}", args.modulation, scale, mel_origin.dim());
    let vel = (1.0 - args.velocity).exp2();
    let end = if args.cutoff < 0.0 {
        args.offset - args.cutoff
    } else {
        mel_origin.ncols() as f32 * THOP_ORIGIN - args.cutoff
    };
    let (con, length_req) = (args.offset + args.consonant, args.length);
    let mut stretch_len = end - con;
    info!("Time params: start={:.4}, end={:.4}, con={:.4}, stretch_len={:.4}, length_req={:.4}",
          args.offset, end, con, stretch_len, length_req);
    if HIFI_CONFIG.loop_mode || args.flags.contains_key("He") {
        info!("Enabling loop mode");
        let start_idx = (con / THOP_ORIGIN + 0.5).floor() as usize;
        let end_idx   = (end / THOP_ORIGIN + 0.5).floor() as usize;
        let pad_size = ((length_req / THOP_ORIGIN).floor() as usize) + 1;
        mel_origin = concatenate![Axis(1),
            mel_origin.slice(s![.., ..start_idx]),
            reflect_pad_2d(mel_origin.slice(s![.., start_idx..end_idx]), pad_size)
        ];
        stretch_len = pad_size as f32 * THOP_ORIGIN;
        info!("new_total_time: {}", mel_origin.ncols() as f32 * THOP_ORIGIN);
    }
    let scal_ratio = if stretch_len < length_req { length_req / stretch_len } else { 1.0 };
    let vel_con = vel * con;
    let stretch = |t: f32| if t < vel_con { t / vel } else { con + (t - vel_con) / scal_ratio };
    let stretched_frames = ((vel_con + (mel_origin.ncols() as f32 * THOP_ORIGIN - con) * scal_ratio) / THOP).floor() as usize + 1;
    let mut stretched_t_mel: Vec<f32> = (0..stretched_frames)
        .map(|i| (i as f32 + 0.5) * THOP)   
        .collect();
    let cut_left = (((args.offset * vel) / THOP + 0.5).floor() as usize).saturating_sub(HIFI_CONFIG.fill);
    let cut_right = (stretched_frames - (((length_req + vel_con) / THOP + 0.5).floor() as usize)).saturating_sub(HIFI_CONFIG.fill);
    stretched_t_mel.truncate(stretched_t_mel.len() - cut_right);
    stretched_t_mel.drain(..cut_left);
    let idx_stretched: Vec<f32> = stretched_t_mel.iter()
        .map(|&t| (stretch(t) / THOP_ORIGIN - 0.5).clamp(0.0, (mel_origin.ncols() - 1) as f32))
        .collect();
    let n_frames = idx_stretched.len();
    info!("Stretched time axis length: {}", n_frames);
    let mut pitch: Vec<f32> = args.pitchbend.iter().map(|&pb| pb + args.pitch).collect();
    if let Some(&t_flag) = args.flags.get("t").and_then(|x| x.as_ref()) {
        pitch.iter_mut().for_each(|p| *p += t_flag * 0.01);
    }
    let cut_left_f = cut_left as f32 * THOP;
    let (new_start, new_end) = (args.offset * vel - cut_left_f, length_req + vel_con - cut_left_f);
    let step_pitch = 0.625 / args.tempo;
    let t_max = new_start + (pitch.len() - 1) as f32 * step_pitch;
    let idx_pitch_clamped: Vec<f32> = (0..n_frames)
        .map(|i| {
            let t = (i as f32 + 0.5) * THOP;   
            let t_clamped = t.clamp(new_start, t_max);
            (t_clamped - new_start) / step_pitch
        })
        .collect();
    let pitch_render = akima(&pitch, &idx_pitch_clamped);
    let f0_render: Vec<f32> = pitch_render.iter().map(|&x| midi_to_hz(x)).collect();
    let mel_render = interp1d(&mel_origin, &idx_stretched);
    let mut render = get_vocoder().lock().unwrap().run(mel_render, f0_render);
    render.drain(((new_end * SR).min(render.len() as f32) as usize)..);
    render.drain(..((new_start * SR).max(0.0) as usize));
    if let Some(&a) = args.flags.get("A").and_then(|x| x.as_ref()) {
        let a = a.clamp(-100.,100.)*1e-4;
        let n = pitch_render.len();
        let mut g = vec![0.; n];
        if n>1 {
            g[0]=pitch_render[1]-pitch_render[0];
            for i in 1..n-1{g[i]=(pitch_render[i+1]-pitch_render[i-1])*0.5;}
            g[n-1]=pitch_render[n-1]-pitch_render[n-2];
        }
        for d in &mut g{*d=5f32.powf(a**d);}
        let last=(g.len()-1)as f32;
        let step=(new_end-new_start)/(render.len()as f32*THOP);
        let start=new_start/THOP;
        for(i,s)in render.iter_mut().enumerate(){
            let t=start+i as f32*step;
            *s*=if t<=0.{g[0]}else if t>=last{g[last as usize]}else{let i0=t as usize;let f=t-i0 as f32;g[i0]+(g[i0+1]-g[i0])*f};
        }
    }
    let mut new_max = 0.0f32;
    for x in render.iter_mut() {
        *x /= scale;
        let abs = x.abs();
        if abs > new_max { new_max = abs; }
    }
    if let Some(&hg) = args.flags.get("HG").and_then(|x| x.as_ref()) {
        growl(&mut render, 80.0, hg.clamp(-100.0, 100.0) * 0.01);
    }
    if HIFI_CONFIG.wave_norm {
        loudness_norm(&mut render,  -16.0,
            args.flags.get("P").and_then(|x| x.as_ref()).map_or(100, |&p| p.clamp(-100.0, 100.0) as u8));
    }
    let mult = (if new_max > HIFI_CONFIG.peak_limit { HIFI_CONFIG.peak_limit / new_max } else { 1.0 }) * args.volume;
    render.iter_mut().for_each(|x| *x *= mult);
    write_audio(&args.out_file, &render)?;
    info!("Successfully processed: {} -> {}", args.in_file.display(), args.out_file.display());
    Ok(())
}