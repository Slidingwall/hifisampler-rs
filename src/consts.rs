pub const SAMPLE_RATE: u32 = 44100;
pub const FFT_SIZE: usize = 2048;
pub const HOP_SIZE: usize = 512;
pub const ORIGIN_HOP_SIZE: usize = 128;
pub const EPSILON: f32 = 1e-9;
use ini::Ini;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::PathBuf;
#[derive(Debug, Clone, PartialEq)]
pub struct HifiConfig {
    pub vocoder_path: PathBuf,
    pub hnsep_path: PathBuf,
    pub wave_norm: bool,
    pub trim_silence: bool,
    pub silence_threshold: f32,
    pub loop_mode: bool,
    pub peak_limit: f32,
    pub fill: usize,
    pub max_workers: usize,
}
pub static HIFI_CONFIG: Lazy<HifiConfig> = Lazy::new(|| load_hifi_config());
fn load_hifi_config() -> HifiConfig {
    let ini = match Ini::load_from_file("hificonfig.ini") {
        Ok(ini) => ini,
        Err(_) => return HifiConfig::default(),
    };
    let def_sec: HashMap<String, String> = ini.section(None::<String>)
        .map(|props| props.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
        .unwrap_or_default();
    HifiConfig {
        vocoder_path: def_sec.get("vocoder_path").cloned().map(PathBuf::from)
            .unwrap_or(PathBuf::from("./model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")),
        hnsep_path: def_sec.get("hnsep_path").cloned().map(PathBuf::from)
            .unwrap_or(PathBuf::from("./model/hnsep_model.onnx")),
        wave_norm: def_sec.get("wave_norm").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        trim_silence: def_sec.get("trim_silence").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        loop_mode: def_sec.get("loop_mode").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        silence_threshold: def_sec.get("silence_threshold").and_then(|s| s.parse().ok())
            .unwrap_or(-52.0),
        peak_limit: def_sec.get("peak_limit").and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
        fill: def_sec.get("fill").and_then(|s| s.parse().ok())
            .unwrap_or(6),
        max_workers: def_sec.get("max_workers").and_then(|s| s.parse().ok())
            .unwrap_or(2),
    }
}
impl Default for HifiConfig {
    fn default() -> Self {
        Self {
            vocoder_path: PathBuf::from("./model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"),
            hnsep_path: PathBuf::from("./model/hnsep_model.onnx"),
            wave_norm: true,
            trim_silence: true,
            silence_threshold: -52.0,
            loop_mode: true,
            peak_limit: 1.0,
            fill: 6,
            max_workers: 2,
        }
    }
}
#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use super::*;
    #[test]
    fn test_default_config() {
        let default = HifiConfig::default();
        assert_eq!(
            default.vocoder_path,
            PathBuf::from("./model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")
        );
        assert_eq!(
            default.hnsep_path,
            PathBuf::from("./model/hnsep_model.onnx")
        );
        assert_eq!(default.wave_norm, true);
        assert_eq!(default.trim_silence, true);
        assert_eq!(default.silence_threshold, -52.0);
        assert_eq!(default.loop_mode, true);
        assert_eq!(default.peak_limit, 1.0);
        assert_eq!(default.fill, 6);
        assert_eq!(default.max_workers, 2);
    }
    #[test]
    fn test_global_config_init() {
        let cfg = &HIFI_CONFIG;
        assert!(!cfg.vocoder_path.as_os_str().is_empty());
        assert!(!cfg.hnsep_path.as_os_str().is_empty());
        assert!(cfg.silence_threshold.is_finite());
        assert!(cfg.peak_limit.is_finite());
        assert!(cfg.fill > 0);
        assert!(cfg.max_workers <= 32);
    }
    #[test]
    fn test_real_ini_load() {
        let ini_exists = Path::new("hificonfig.ini").exists();
        let cfg = &HIFI_CONFIG;
        if ini_exists {
            println!("Real hificonfig.ini exists, verify parsed result is valid");
            assert!(!cfg.vocoder_path.as_os_str().is_empty());
            assert!(!cfg.hnsep_path.as_os_str().is_empty());
        } else {
            println!("Real hificonfig.ini does not exist, verify default config is returned");
            assert_eq!(**cfg, HifiConfig::default());
        }
    }
    #[test]
    fn test_parse_fault_tolerance() {
        let cfg = &HIFI_CONFIG;
        assert!(cfg.silence_threshold.is_finite());
        assert!(cfg.peak_limit.is_finite());
        assert!(cfg.fill <= 100);
        assert!(cfg.max_workers >= 1 && cfg.max_workers <= 32);
    }
}
pub const FORMANT_HR: [f32; 128] = [
    0.0242596995, 0.0242701769, 0.0242842417, 0.024301935, 0.0243232604, 0.0243482366, 0.0243768822, 0.0244092345,
    0.0244453009, 0.024485115, 0.0245287176, 0.0245761406, 0.0246274304, 0.0246826205, 0.0247417837, 0.0248049498,
    0.0248721614, 0.0249435157, 0.0250190347, 0.0250988062, 0.0251828954, 0.025271412, 0.0253643934, 0.0254619364,
    0.0255641378, 0.0256711077, 0.0257828962, 0.025899671, 0.026021501, 0.0261485334, 0.026280893, 0.0264186822,
    0.0265620816, 0.0267114788, 0.0268741194, 0.0270485599, 0.0272357352, 0.0274367891, 0.0276528988, 0.0278854407,
    0.0281358901, 0.0284059048, 0.0286973659, 0.029012356, 0.0293531902, 0.029722536, 0.0301233772, 0.0305590574,
    0.0310334712, 0.0315510817, 0.0321168639, 0.0327367224, 0.0334173255, 0.034166608, 0.0349936821, 0.0359092019,
    0.0369258635, 0.0380585343, 0.0393247679, 0.0407454893, 0.0423454382, 0.0441537835, 0.0462048613, 0.0485380292,
    0.051197011, 0.0542267747, 0.0576666519, 0.0615338944, 0.065792948, 0.0702995658, 0.0747192726, 0.0784466043,
    0.0806255713, 0.0804125443, 0.077454783, 0.0721836239, 0.0655695871, 0.0585971624, 0.0519366711, 0.0459174104,
    0.0406362191, 0.0360678844, 0.0321368948, 0.0287547559, 0.0258370955, 0.0233096574, 0.021109743, 0.019185219,
    0.0174931716, 0.0159983337, 0.0146716731, 0.0134892203, 0.0124310991, 0.01148073, 0.0106242159, 0.00984986871,
    0.00914776232, 0.00850945804, 0.00792772323, 0.00739634503, 0.00690994971, 0.00646387646, 0.00605406053, 0.00567694334,
    0.0053294017, 0.00500867981, 0.00471234322, 0.00443822704, 0.00418440625, 0.00394916441, 0.00373096508, 0.00352842594,
    0.00334031112, 0.00316550303, 0.00300299469, 0.00285187503, 0.00271132169, 0.00258058822, 0.0024589987, 0.0023459401,
    0.00224085711, 0.00214324682, 0.00205265475, 0.00196866971, 0.00189092313, 0.00181908533, 0.00175286341, 0.0016919995,
];


