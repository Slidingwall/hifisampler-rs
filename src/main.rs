mod audio;
mod consts;
mod resample;
mod utils;
mod model;
mod server;
use anyhow::Result;
use tokio;
use tracing_subscriber::{fmt, prelude::*};
use crate::consts::HIFI_CONFIG;
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
fn init_logging() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing::level_filters::LevelFilter::INFO)
        .with(fmt::layer()
            .without_time() 
            .with_target(false) 
            .with_thread_names(false)) 
        .init();
    Ok(())
}
#[tokio::main]
async fn main() -> Result<()> {
    init_logging()?;
    model::initialize_models();
    tracing::info!("starting_server_on_0.0.0.0:{}",8572);
    server::run(8572, HIFI_CONFIG.max_workers).await;
    Ok(())
}