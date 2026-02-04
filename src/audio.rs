pub mod post_process;
use crate::consts::SAMPLE_RATE;
use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::{fs::File, path::{Path, PathBuf}};
use symphonia::{
    core::{
        audio::{SampleBuffer, SignalSpec},
        io::MediaSourceStream,
        probe::Hint,
    },
    default::{get_codecs, get_probe},
};
const I16_MAX: f64 = i16::MAX as f64;
fn resample_audio(audio: &[f64], in_fs: u32, out_fs: u32) -> Result<Vec<f64>> {
    let ratio = out_fs as f64 / in_fs as f64;
    if audio.is_empty() || in_fs == out_fs {
        return Ok(audio.to_vec());
    }
    let mut resampled = Vec::with_capacity((audio.len() as f64 * ratio).ceil() as usize);
    let mut resampler = SincFixedIn::<f64>::new(
        ratio,
        2.0,
        SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            oversampling_factor: 64,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::Hann,
        },
        256,
        1,
    )?;
    for chunk in audio.chunks(256) {
        let input_chunk: Vec<f64> = chunk
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(256)
            .collect();
        let binding = resampler.process(&[&input_chunk], None)?;
        let output = binding.get(0).unwrap();
        resampled.extend_from_slice(output);
    }
    let binding=resampler.process(&[&[]], None)?;
    let final_output = binding.get(0).unwrap();
    resampled.extend_from_slice(final_output);
    Ok(resampled)
}
pub fn read_audio<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    let mut path = PathBuf::from(path.as_ref());
    if !path.exists() {
        let common_extensions = ["wav", "flac", "ogg", "mp3", "aac"];
        let found = common_extensions
            .iter()
            .find(|&&ext| {
                path.set_extension(ext);
                path.exists()
            });
        if found.is_none() {
            return Err(anyhow!(
                "No supported audio file found (tried extensions: {:?})",
                common_extensions
            ));
        }
    }
    let source = File::open(&path)?;
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let mut probed = get_probe()
        .format(&Hint::new(), mss, &Default::default(), &Default::default())?;
    let track = probed
        .format
        .default_track()
        .ok_or_else(|| anyhow!("No audio track found"))?;
    let spec = SignalSpec {
        channels: track.codec_params.channels.unwrap(),
        rate: track.codec_params.sample_rate.unwrap(),
    };
    let mut decoder = get_codecs()
        .make(&track.codec_params, &Default::default())?;
    let mut audio = Vec::with_capacity(409600);
    let mut packet_buffer = SampleBuffer::<f64>::new(4096, spec);
    let track_id = track.id;
    let channels = spec.channels.count();
    while let Ok(packet) = probed.format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        if let Ok(decoded) = decoder.decode(&packet) {
            packet_buffer.copy_interleaved_ref(decoded);
            let samples = packet_buffer.samples();
            if channels == 1 {
                audio.extend_from_slice(samples);
            } else {
                audio.extend(
                    samples
                        .chunks(channels)
                        .map(|frame| {
                            let mut sum = 0.0;
                            for &s in frame {
                                sum += s;
                            }
                            sum / channels as f64
                        }),
                );
            }
        }
    }
    if spec.rate == SAMPLE_RATE {
        Ok(audio)
    } else {
        resample_audio(&audio, spec.rate, SAMPLE_RATE)
    }
}
pub fn write_audio<P: AsRef<Path>>(path: P, audio: &[f64]) -> Result<()> {
    let mut writer = WavWriter::new(
        File::create(path.as_ref())?,
        WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        },
    )?;
    for &s in audio {
        writer.write_sample((s * I16_MAX) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::{read_audio, write_audio};
    use std::{path::Path, time::Instant};
    #[test]
    fn test_read_write() {
        let test_paths = ["test/01.wav", "test/pjs001.wav"]
            .iter()
            .map(Path::new)
            .collect::<Vec<_>>();
        for path in test_paths {
            println!("Testing: {:?}", path.as_os_str());
            let out_path = path.with_extension("out.wav");
            let now = Instant::now();
            if path.exists() {
                let audio = read_audio(path).expect("Read failed");
                println!("Read time: {:.2?}", now.elapsed());
                let write_now = Instant::now();
                write_audio(&out_path, &audio).expect("Write failed");
                println!("Write time: {:.2?}", write_now.elapsed());
            } else {
                println!("File not found: {:?} (skipped)", path.as_os_str());
            }
        }
    }
}