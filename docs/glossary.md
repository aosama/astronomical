# Glossary: the memory areas

Four areas of memory matter when Astronomical runs a model on MLX. Each has one formal name, and the prefix says who owns it: `astro_` is Astronomical's, `mlx_` is MLX's, `metal_` is macOS's. Code, logs, tests, documentation and conversation use these names and no others for these areas.

| Name | What it holds | Owner | Where it lives | Counts against the memory limit |
|---|---|---|---|---|
| `astro_expert_cache` | Experts of a sparse model kept in memory so they are not read from storage again | Astronomical | Memory | Yes |
| `mlx_working_buffers` | Buffers one step of the model is passing through right now | MLX | Memory | Yes |
| `mlx_buffer_cache` | Buffers MLX has finished with but keeps, so the next step reuses them | MLX | Memory | Yes |
| `metal_kernel_cache` | Compiled GPU programs | macOS (Metal) | Storage | No |

## `astro_expert_cache`

A sparse model routes every token to a few experts in every layer. When the whole model does not fit under the memory limit, the experts that are not in memory are read from storage. The `astro_expert_cache` is the set of experts held in memory: one table per layer, with a capacity the memory governor sets.

- In the engine it is the type `AstroExpertCache` (`crates/model-serving/src/qwen3_5_streaming/expert_streaming/astro_expert_cache.rs`).
- Its arithmetic model, replayed against recorded replies without a GPU, is `astro_expert_cache_model` in `crates/astronomical-memory-policy`.
- To the memory ledger it is the kept weights of a sparse model: `kept_weight_capacity_bytes` is the capacity it is given. A dense model keeps whole layers under the same account and has no `astro_expert_cache`.
- Older code calls an expert held in it kept, retained, resident, warm or hot. All five mean the same thing: the expert is in the `astro_expert_cache`.

## `mlx_working_buffers`

The memory one step needs while it runs: activations, intermediate results and the outputs it is producing. MLX counts these as part of its active memory, together with the model's weights.

- Measured, never estimated: the peak MLX reports for one layer step, less what was held when the step began and less the weights the step had in hand (`crates/astronomical-memory-policy/src/ledger/mlx_working_buffers.rs`).
- Reading a prompt needs far more than writing a reply, because a prompt step handles a whole span of tokens and a reply step handles one.
- The memory ledger sets them aside as the account `mlx_working_buffer_bytes`.

## `mlx_buffer_cache`

When a step finishes, MLX does not hand its buffers back to the system. It keeps them and gives them to the next step that asks for the same size, which is far cheaper than allocating again. Those kept, currently unused buffers are the `mlx_buffer_cache`.

- MLX's own name for it is "cache memory": `get_cache_memory`, `set_cache_limit` and `clear_cache` in `mlx/memory.h`, backed by the class `BufferCache` in MLX's Metal allocator.
- It is not counted in MLX's active memory, but it does occupy memory under the limit.
- MLX starts releasing it once active memory plus the cache reaches its collection line, which stock MLX places at 95% of the working set Metal recommends for the Mac. Past that line no buffer is reused and every step allocates afresh; a large sparse model then replies at about a third of its speed.
- The memory ledger has no account for it yet. It lives inside the ledger's 2% margin.

## `metal_kernel_cache`

MLX compiles its GPU programs (kernels) the first time each is used on a Mac. macOS stores the compiled programs on storage and keeps them across restarts, which MLX's documentation calls the Metal kernel cache.

- It is small and it is not memory the limit governs.
- Its only visible effect is that the first use of a kernel after an MLX or macOS update is slower.
- It is not the `mlx_buffer_cache`. "Kernel cache" must never be used for memory MLX holds while a model runs.

## How the limit is divided

For a sparse model being streamed, the memory limit a person sets is divided as:

```
limit = core weights that always stay in memory
      + the conversation's stored state
      + mlx_working_buffers for one step
      + slots for weights being read
      + margin (where the mlx_buffer_cache lives today)
      + astro_expert_cache
```

The `astro_expert_cache` is always the remainder. Whatever the other accounts need is taken first, and it gets what is left.

## On screen

Screens do not show these names; they use plain words. On Observatory's Overview, where the limit is divided:

| On screen | Formal name |
|---|---|
| "Room for the rest of the model" | The capacity of the `astro_expert_cache` (kept layers on a dense model) |
| "Room to work" | `mlx_working_buffers`, plus any lease |
| "Safety margin" | The margin, where the `mlx_buffer_cache` lives today |
