use std::future::Future;

use tokio::time::Interval;

pub(crate) async fn await_generation_advance_with_live_progress<GenerationAdvanceOutput>(
    generation_advance_future: impl Future<Output = GenerationAdvanceOutput>,
    progress_interval: &mut Interval,
    mut report_progress: impl FnMut(),
) -> GenerationAdvanceOutput {
    tokio::pin!(generation_advance_future);
    loop {
        tokio::select! {
            generation_advance_output = &mut generation_advance_future => return generation_advance_output,
            _ = progress_interval.tick() => report_progress(),
        }
    }
}
