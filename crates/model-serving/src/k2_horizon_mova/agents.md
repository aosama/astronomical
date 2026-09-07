# K2 Horizon MoVA family constitution

This package is the native Rust/MLX-C owner for the `k2_horizon_mova` family. It does not execute checkpoint Python.

## Memory

`crates/model-serving/src/memory` is the sole owner of admission, residency, reclamation, recovery, live-ceiling, and `Resident` / `Hybrid` / `Paged` classification.

Family code measures byte facts, calls those decision types, and enacts the result. It must not re-derive that arithmetic.

- Do not add `k2_horizon_mova/memory/`.
- Do not introduce K2 or MoVA types inside the centralized memory package.
- Do not import `laguna` or `qwen3_5` internals. Family-neutral reuse is `memory`, `expert_paging`, `sparse_experts`, `decoder_cache`, `persistent_cache`, and `performance_attribution`.
- Persistent prompt cache is append-only attention KV through `PersistentPromptCacheDiskStore`. This family maps tensors; it does not invent a second store.
- Performance attribution is the serving-settings log plus `PerformanceAttribution::enabled` on load and each generation. Flush reports on load success and generation end; do not leave the recorder disabled on the serving path.
- The prompt always opens `<ifm|think>`. Honor `thinking_budget` when the request sets it; when it is omitted, apply a family default that leaves room for `</ifm|think>` and a visible answer, then inject the close token into decoder history.
- Expert plan slots are FFN then MoVA, built from measured stack bytes. If centralized residency would stream, fail closed until complete-layer streaming is enacted. Do not OOM and do not pretend Resident.
- Shared experts are always-resident non-expert weights, not expert-plan citizens.
- Sparse FFN and MoVA value experts are two routed pools presented as two `ExpertLayerGeometry` entries in one contiguous plan-slot list. Centralized `plan_expert_residency` decides; this family maps slots and enacts pages.

## Family, not one artifact

Production numbers come from `config.json`, the quantization document, and the safetensors index. Do not hardwire layer counts, expert counts, hidden size, vocabulary size, bit width, group size, or a Hugging Face repository name.

The first executable on-disk dialect is stacked MLX affine. Unstacked per-expert tensors stay rejected until a later normalizer exists.

## Serving advertisement

Do not advertise this family on `/v1/models` until an executable path exists. Classification may recognize the family while discovery still returns no requestable model.
