# How decision-model training works

Engineering notes for anyone reading or changing the training code
(`crates/engine/model-serving/src/decision_models`, `crates/tooling/decision-training`).
How to use training is in the Astronomical skill,
`skill/astronomical/references/decision-models.md`.

Facts that hold across releases, for anyone reading or changing the training
code in Astronomical (`crates/engine/model-serving/src/decision_models`,
`crates/tooling/decision-training`):

- **The Gated DeltaNet recurrence in Qwen3.5 models (Kev, Clef) has its own
  backward pass.** MLX's fused kernel cannot be differentiated, so training
  keeps it for the forward pass and attaches a custom backward pass that
  differentiates MLX's plain-operation recurrence: fast forward, exact
  gradients.
- **Attention runs as plain operations while training.** MLX's fused
  attention cannot be differentiated either; serving keeps the fused kernel.
- **Many questions share one pass, each starting empty.** Training sequences
  all begin at their first token, so they need none of serving's stored
  state: a zero convolution window, a zero recurrent state and causal
  attention, with rows padded at the end. Every operation is causal, so
  padding never changes a real position. One question per pass spends most
  of its time on fixed per-kernel cost.
- **Pass size comes from measured memory.** Each model has a measured cost
  per padded token while a pass runs backwards, and passes are sized to the
  memory limit. A fixed size that fits a small model runs a large one out of
  memory.
- **Add-ons sit on the frozen quantized layers** and are cast to each
  layer's number format, so a 32-bit add-on never widens a 16-bit layer.
- **The `mlx_buffer_cache` is reclaimed after every pass**, or freed buffers
  pile up over a long training.
- **A text too long to keep is trained layer by layer.** When one pass would
  hold more than half the memory limit, each decoder layer runs through
  MLX's checkpoint transform and is recomputed during the backward pass.
  MLX differentiates a checkpointed function only through its explicit
  inputs, so each layer takes its own add-on weights as inputs. Same
  weights, a fraction of the memory, slower.
- **Trained weights cross processes through a file.** Messages between the
  worker, the daemon and the CLI are capped well below a 9B model's add-ons,
  so the worker writes them to a private temporary file and the message
  carries only its name. Prove anything that crosses those processes
  through the real CLI, not only by calling the engine directly.
- **Scores that decide anything come from the serving path.** Gradient
  passes may batch differently, but the comparison that gives the verdict
  runs the same forward pass that will serve the trained folder.
