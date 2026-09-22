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
0.004784035, 0.005116146, 0.005477889, 0.005872304, 0.006302774, 0.006773066, 0.007287379, 0.007850393,
0.00846733, 0.009144015, 0.009886959, 0.01070343, 0.01160155, 0.01259041, 0.01368015, 0.01488215,
0.0162091, 0.01767521, 0.01929639, 0.02109042, 0.02307716, 0.02527884, 0.02772026, 0.03042914,
0.03343636, 0.03677635, 0.04048743, 0.04461216, 0.04919778, 0.05429659, 0.05996635, 0.06627072,
0.07327963, 0.08106959, 0.08972405, 0.0993335, 0.1099956, 0.1218151, 0.1349033, 0.1493779,
0.1653613, 0.1829799, 0.2023616, 0.2236335, 0.2469181, 0.2723298, 0.299969, 0.3299161,
0.3622241, 0.39691, 0.4339455, 0.4732457, 0.5146583, 0.5579513, 0.6028014, 0.6487831,
0.6953593, 0.7418749, 0.7875535, 0.8315011, 0.872715, 0.9101024, 0.9425061, 0.9687429,
0.9876504, 0.9981428, 0.9992758, 0.9903135, 0.9707966, 0.940604, 0.9000021, 0.849674,
0.790725, 0.7246562, 0.6533062, 0.5787604, 0.5032345, 0.4289389, 0.3579382, 0.2920186,
0.2325804, 0.1805635, 0.1364205, 0.1001343, 0.07127832, 0.04911126, 0.03268787, 0.02097287,
0.01294288, 0.007664612, 0.004344733, 0.002351374, 0.00121165, 0.0005927559, 0.000274472, 0.00011991,
4.925866e-05, 1.895999e-05, 6.812386e-06, 2.275924e-06, 7.040654e-07, 2.00804e-07, 5.255811e-08, 1.256381e-08,
2.729003e-09, 5.357567e-10, 9.45285e-11, 1.490113e-11, 2.085586e-12, 2.574782e-13, 2.784594e-14, 2.619033e-15,
2.125957e-16, 1.477494e-17, 8.717227e-19, 4.327708e-20, 1.791034e-21, 6.118701e-23, 1.707767e-24, 3.852111e-26,
6.942403e-28, 9.877426e-30, 1.095529e-31, 9.347368e-34, 6.050199e-36, 2.927585e-38, 1.042814e-40, 2.690449e-43
];
pub const FORMANT_HD: [f32; 128] = [
0.002620138, 0.002669032, 0.002720065, 0.002773356, 0.002829029, 0.002887219, 0.002948071, 0.003011736,
0.003078379, 0.003148177, 0.003221318, 0.003298004, 0.003378452, 0.003462896, 0.003551586, 0.003644792,
0.003742804, 0.003845937, 0.003954527, 0.00406894, 0.00418957, 0.004316843, 0.00445122, 0.004593202,
0.004743332, 0.004902197, 0.005070437, 0.005248749, 0.005437891, 0.005638687, 0.005852039, 0.006078931,
0.006320438, 0.006577738, 0.00685212, 0.007144999, 0.007457928, 0.007792614, 0.008150937, 0.008534968,
0.00894699, 0.009389526, 0.009865364, 0.01037759, 0.01092963, 0.01152528, 0.01216876, 0.01286476,
0.01361851, 0.01443583, 0.01532325, 0.01628802, 0.0173383, 0.01848321, 0.01973298, 0.0210991,
0.02259449, 0.0242337, 0.02603309, 0.02801114, 0.03018867, 0.03258921, 0.03523933, 0.03816903,
0.04141228, 0.04500742, 0.04899779, 0.05343233, 0.05836623, 0.06386168, 0.06998855, 0.0768252,
0.08445914, 0.09298773, 0.1025185, 0.1131698, 0.1250699, 0.1383571, 0.1531778, 0.1696842,
0.1880304, 0.2083674, 0.2308358, 0.2555565, 0.2826201, 0.3120735, 0.3439059, 0.3780356,
0.4142962, 0.4524285, 0.4920772, 0.5327948, 0.5740554, 0.6152773, 0.6558533, 0.6951853,
0.732721, 0.7679852, 0.8006043, 0.830321, 0.8569971, 0.8806071, 0.9012232, 0.9189965,
0.9341353, 0.9468843, 0.9575066, 0.9662679, 0.9734255, 0.9792199, 0.9838699, 0.9875704,
0.9904913, 0.9927784, 0.9945554, 0.9959252, 0.996973, 0.9977683, 0.9983672, 0.9988147,
0.9991463, 0.9993902, 0.9995679, 0.9996965, 0.9997886, 0.9998541, 0.9999002, 0.9999324
];
pub const FORMANT_HC: [f32; 128] = [
0.4325754, 0.4395185, 0.4466762, 0.4540543, 0.4616587, 0.4694951, 0.4775694, 0.4858871,
0.4944538, 0.5032748, 0.5123553, 0.5217001, 0.5313137, 0.5412003, 0.5513635, 0.5618063,
0.5725313, 0.5835401, 0.5948334, 0.6064112, 0.6182721, 0.6304136, 0.6428315, 0.6555204,
0.6684727, 0.6816791, 0.6951279, 0.7088049, 0.7226933, 0.7367733, 0.7510218, 0.7654122,
0.7799138, 0.7944921, 0.8091079, 0.823717, 0.8382702, 0.852713, 0.8669847, 0.8810188,
0.8947426, 0.9080764, 0.9209341, 0.9332228, 0.9448427, 0.9556872, 0.9656433, 0.9745918,
0.9824076, 0.9889608, 0.9941173, 0.99774, 0.9996901, 0.9998287, 0.9980189, 0.9941276,
0.9880283, 0.9796035, 0.9687482, 0.9553725, 0.9394053, 0.9207977, 0.8995265, 0.8755981,
0.8490507, 0.8199581, 0.788432, 0.7546231, 0.7187231, 0.6809634, 0.6416151, 0.6009855,
0.5594146, 0.5172699, 0.4749399, 0.4328253, 0.3913308, 0.3508544, 0.3117774, 0.2744532,
0.2391982, 0.2062811, 0.1759161, 0.1482573, 0.1233939, 0.1013508, 0.08208965, 0.06551354,
0.05147445, 0.03978209, 0.03021448, 0.02252942, 0.01647579, 0.01180412, 0.008275946, 0.005671251,
0.003793757, 0.00247408, 0.001570748, 0.0009694174, 0.0005807057, 0.0003370842, 0.0001892847, 0.0001026384,
5.364142e-05, 2.69666e-05, 1.301307e-05, 6.014644e-06, 2.656526e-06, 1.118507e-06, 4.477936e-07, 1.700075e-07,
6.103549e-08, 2.066045e-08, 6.573316e-09, 1.959285e-09, 5.452353e-10, 1.411491e-10, 3.386309e-11, 7.498906e-12,
1.526406e-12, 2.843332e-13, 4.824596e-14, 7.420919e-15, 1.029419e-15, 1.280961e-16, 1.421787e-17, 1.399303e-18
];
