use anyhow::Result;
use axum::{ extract::State, http::StatusCode, response::IntoResponse, routing::get, Router };
use std::{ collections::HashMap, net::SocketAddr, path::PathBuf, sync::{Arc, atomic::{AtomicBool, Ordering}} };
use tokio::sync::Semaphore;
use tracing::{info, warn, error};
use crate::{
    resample::resample,
    utils::parser::{flag_parser, pitch_parser, pitch_string_to_cents, tempo_parser}
};
#[derive(Clone)]
pub struct AppState {
    server_ready: Arc<AtomicBool>,
    concurrency_semaphore: Arc<Semaphore>,
}
#[derive(Debug)]
pub struct Arguments {
    pub in_file: PathBuf,
    pub out_file: PathBuf,
    pub pitch: f32,
    pub velocity: f32,
    pub flags: HashMap<String, Option<f32>>,
    pub offset: f32,
    pub length: f32,
    pub consonant: f32,
    pub cutoff: f32,
    pub volume: f32,
    pub modulation: f32,
    pub tempo: f32,
    pub pitchbend: Vec<f32>,
}
pub fn split_arguments(input: &str) -> Result<Arguments> {
    let tokens: Vec<&str> = input.split(' ').collect();
    let prefix = tokens[..tokens.len() - 11].join(" ");
    let split_idx = prefix.find(".wav ").ok_or_else(|| anyhow::anyhow!("Missing .wav in input"))?;
    let (in_file, out_file) = prefix.split_at(split_idx + 4);
    let len = tokens.len();
    Ok(Arguments {
        in_file: PathBuf::from(in_file),
        out_file: PathBuf::from(out_file.trim_start_matches(' ')),
        pitch: pitch_parser(tokens[len - 11])? as f32,
        velocity: tokens[len - 10].parse::<f32>()? * 0.01,
        flags: flag_parser(tokens[len - 9])?,
        offset: tokens[len - 8].parse::<f32>()? * 0.001,
        length: tokens[len - 7].parse::<f32>()? * 0.001,
        consonant: tokens[len - 6].parse::<f32>()? * 0.001,
        cutoff: tokens[len - 5].parse::<f32>()? * 0.001,
        volume: tokens[len - 4].parse::<f32>()? * 0.01,
        modulation: tokens[len - 3].parse::<f32>()? * 0.01,
        tempo: tempo_parser(tokens[len - 2])?,
        pitchbend: pitch_string_to_cents(tokens[len - 1])?,
    })
}
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let ready = state.server_ready.load(Ordering::Relaxed);
    let (status, msg) = if ready {
        (StatusCode::OK, "Server Ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Server Initializing")
    };
    info!("{}", msg);
    (status, msg.to_string())
}
async fn handle_post(State(state): State<AppState>, body: String) -> (StatusCode, String) {
    if !state.server_ready.load(Ordering::Relaxed) {
        warn!("POST arrived but server not ready.");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Server initializing, please retry.".to_string(),
        );
    }
    info!("post_data_string: {}", body);
    let args = match split_arguments(&body) {
        Ok(a) => a,
        Err(e) => {
            error!("Failed to parse arguments: {}", e);
            return (StatusCode::BAD_REQUEST, format!("Invalid arguments: {}", e));
        }
    };
    let note_info = format!(
        "'{}' -> '{}'",
        args.in_file.file_stem().unwrap().to_str().unwrap(),
        args.out_file.file_name().unwrap().to_str().unwrap()
    );
    info!("Queued {} ...", note_info);
    let permit = state.concurrency_semaphore.acquire_owned().await.unwrap();
    let task_result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        resample(args)
    }).await.unwrap();
    if let Ok(()) = task_result {
        info!("Processing {} successful.", note_info);
        (StatusCode::OK, format!("Success: {}", note_info))
    } else {
        error!("Processing {} failed: {}", note_info, task_result.unwrap_err());
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Error processing: Internal error.".to_string()
        )
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