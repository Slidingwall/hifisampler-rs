use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, atomic::{AtomicBool, Ordering}},
};
use tokio::sync::Semaphore;
use tracing::{info, warn, error};
use crate::resample::Resampler;
#[derive(Clone)]
pub struct AppState {
    server_ready: Arc<AtomicBool>,
    concurrency_semaphore: Arc<Semaphore>,
}
pub fn split_arguments(input: &str) -> Vec<String> {
    let tokens: Vec<&str> = input.split(' ').collect();
    let prefix = tokens[..tokens.len()-11].join(" ");
    let split_idx = prefix.find(".wav ").unwrap();
    let (in_file, out_file) = prefix.split_at(split_idx + 4);
    let mut args = vec![
        in_file.to_string(),
        out_file.trim_start_matches(' ').to_string()
    ];
    args.extend(tokens[tokens.len()-11..].iter().map(|s| s.to_string()));
    args
}
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let ready = state.server_ready.load(Ordering::SeqCst);
    let (status, msg) = if ready {
        (StatusCode::OK, "Server Ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Server Initializing")
    };
    info!("{}", msg);
    (status, msg.to_string())
}
async fn handle_post(State(state): State<AppState>, body: String) -> (StatusCode, String) {
    if !state.server_ready.load(Ordering::SeqCst) {
        warn!("POST arrived but server not ready.");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Server initializing, please retry.".to_string(),
        );
    }
    let _permit = state.concurrency_semaphore.acquire().await.unwrap();
    info!("post_data_string: {}", body);
    let args = split_arguments(&body);
    let note_info = format!(
        "'{}' -> '{}'",
        PathBuf::from(&args[0]).file_stem().unwrap().to_str().unwrap(),
        PathBuf::from(&args[1]).file_name().unwrap().to_str().unwrap()
    );
    info!("Queued {} ...", note_info);
    match tokio::task::spawn_blocking(move || Resampler::new(args)).await.unwrap() {
        Ok(()) => {
            info!("Processing {} successful.", note_info);
            (StatusCode::OK, format!("Success: {}", note_info))
        }
        Err(e) => {
            error!("Processing {} failed: {}", note_info, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error processing: Internal error.".to_string()
            )
        }
    }
}
pub async fn run(port: u16, max_workers: usize) {
    info!("Starting server (max_workers={})...", max_workers);
    let app_state = AppState {
        server_ready: Arc::new(AtomicBool::new(false)),
        concurrency_semaphore: Arc::new(Semaphore::new(max_workers)),
    };
    let app = Router::new()
        .route("/", get(health_check).post(handle_post))
        .with_state(app_state.clone());
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    app_state.server_ready.store(true, Ordering::SeqCst);
    info!(
        "Listening on {}; axum + inference-thread={}",
        listener.local_addr().unwrap(),
        max_workers
    );
    axum::serve(listener, app).await.unwrap();
}
#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use crate::{
        server::split_arguments,
        utils::parser::{pitch_parser, tempo_parser}
    };
    #[test]
    fn test_basic_arguments() {
        let input = "input.wav output.wav C4 1.0 \"\" 0.0 1000.0 0.0 0.0 100.0 0.0 !120 AA";
        let args = split_arguments(input);
        assert_eq!(args[0], "input.wav");
        assert_eq!(args[1], "output.wav");
        let pitch = pitch_parser(&args[2]).unwrap();
        assert_eq!(pitch, 60);
        let tempo = tempo_parser(&args[11]).unwrap();
        assert_eq!(tempo, 120.0);
    }
    #[test]
    fn test_paths_with_spaces() {
        let input = "my audio file.wav output dir/result.wav A4 0.8 \"flag\" 1.5 2000.0 0.5 0.3 90.0 2.0 !90 B7CPCV";
        let args = split_arguments(input);
        assert_eq!(args[0], "my audio file.wav");
        assert_eq!(args[1], "output dir/result.wav");
        let pitch = pitch_parser(&args[2]).unwrap();
        assert_eq!(pitch, 69);
    }
    #[test]
    fn test_minimum_tokens() {
        let input = "a.wav b.wav 60 0.0 x 0.0 0.0 0.0 0.0 0.0 0.0 !100 zz";
        let args = split_arguments(input);
        assert_eq!(args.len(), 13);
        assert_eq!(args[0], "a.wav");
        assert_eq!(args[1], "b.wav");
    }
    #[test]
    fn test_parameter_types() {
        let input = "in.wav out.wav C5 1.5 \"fe+10\" -2.3 500.5 3.0 -0.5 80.0 -1.0 !150 AB#14#CD";
        let args = split_arguments(input);
        let pitch = pitch_parser(&args[2]).unwrap();
        assert_eq!(pitch, 72);
        let tempo = tempo_parser(&args[11]).unwrap();
        assert_eq!(tempo, 150.0);
        let offset: f64 = args[5].parse().unwrap();
        assert_eq!(offset, -2.3);
    }
    #[test]
    fn test_path_compatibility() {
        let input = "test data/input.wav output_dir/out.wav D4 1.0 \"\" 0.0 500.0 0.0 0.0 80.0 0.0 !100 C5CC";
        let args = split_arguments(input);
        let in_path = PathBuf::from(&args[0]);
        assert!(in_path.ends_with("input.wav"));
        let out_path = PathBuf::from(&args[1]);
        assert!(out_path.ends_with("out.wav"));
        assert!(out_path.starts_with("output_dir"));
    }
}