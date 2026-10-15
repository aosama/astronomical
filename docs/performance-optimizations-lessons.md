# Performance Optimization Lessons

Durable engineering lessons for LLM (Large Language Model), VLM (Vision Language Model), and MLX (Machine Learning framework for Apple silicon) APIs. Never raw metrics from one machine; always the transferable principle, with the measurement that established it.

## Expert streaming throughput is concurrency times per-call latency

The bounded SafeTensors (safe tensor serialization format) positional reader's achievable throughput is exactly `concurrent read permits x bytes per read / per-read latency`. A process-wide permit pool of four capped unpressured expert streaming near 14 GB/s (decimal SI: 1 GB = 1,000,000,000 bytes) on Apple silicon; raising the pool to sixteen reached roughly 23 GB/s because physical SSD (Solid-State Drive) queue depth far exceeds four. Tune the permit count, not the read size, when the disk is not the limit.

## macOS active-memory pressure stalls positional reads regardless of concurrency

Holding roughly 27 GB of materialized MLX expert arrays inflated per-positional-read latency more than four-fold (tens of milliseconds to over a hundred for ~30-50 MB reads). Raising read concurrency from four to sixteen did not restore throughput: the stall lives in the operating system's memory-management path, not the storage device. Consequence: a paging policy that seats most experts and then streams the remainder pays a collapsed-rate tail on every later chunk. Extra permits still help wall time by overlapping the stalled reads, but the byte rate stays memory-bound. When a served model's complete payload is close to the machine's wired limit, either seat everything that will be re-read or minimize pressured-phase streaming volume; the middle ground is the slowest configuration.

## Attribute paged prefill per chunk before blaming the disk

A model-level probe that replays the engine's request-open sequence (residency-plan publication, then fixed-size prefill chunks, each with its own performance attribution) separates three failure layers that end-to-end tokens-per-second numbers conflate: storage read rate, retention policy behavior, and engine scheduling. Measured this way, a 2,048-token chunk of a sparse MoE (Mixture of Experts) model routes essentially every expert of every layer, so paged prefill streams one complete expert payload per chunk unless the retained cache persists across chunks. Verify the retained-expert ceiling is actually published (non-zero) before concluding a retention policy is broken: a probe that skips the engine's plan publication observes a zero ceiling and zero seating, which looks like a policy bug but is a harness bug.
