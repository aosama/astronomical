#![forbid(unsafe_code)]

#[tokio::main]
async fn main() {
    std::future::pending::<()>().await;
}
