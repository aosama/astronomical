# Glossary: memory areas and caches

Every cache and memory area in Astronomical has one formal name, and the prefix says who owns it: `astro_` is Astronomical's, `mlx_` is MLX's, `metal_`, `macos_` and `coreml_` are Apple's, and the rest name the outside project that owns it. Code, logs, tests, documentation and conversation use these names. The bare word "cache" is never used on its own: every mention says which cache it means.

## The four memory areas under the limit

| Name | What it holds | Owner | Where it lives | Counts against the memory limit |
|---|---|---|---|---|
| `astro_expert_cache` | Experts of a sparse model kept in memory so they are not read from storage again | Astronomical | Memory | Yes |
| `mlx_working_buffers` | Buffers one step of the model is passing through right now | MLX | Memory | Yes |
| `mlx_buffer_cache` | Buffers MLX has finished with but keeps, so the next step reuses them | MLX | Memory | Yes |
| `metal_kernel_cache` | Compiled GPU programs | macOS (Metal) | Storage | No |

## `astro_expert_cache`

A sparse model routes every token to a few experts in every layer. When the whole model does not fit under the memory limit, the experts that are not in memory are read from storage. The `astro_expert_cache` is the set of experts held in memory: one table per layer, with a capacity the memory governor sets.

- In the engine it is the type `AstroExpertCache` (`crates/families/text/qwen35/src/qwen3_5_streaming/expert_streaming/astro_expert_cache.rs`).
- Its arithmetic model, replayed against recorded replies without a GPU, is `astro_expert_cache_model` in `crates/foundation/memory-policy`.
- To the memory ledger it is the kept weights of a sparse model: `kept_weight_capacity_bytes` is the capacity it is given. A dense model keeps whole layers under the same account and has no `astro_expert_cache`.
- Older code calls an expert held in it kept, retained, resident, warm or hot. All five mean the same thing: the expert is in the `astro_expert_cache`.

## `mlx_working_buffers`

The memory one step needs while it runs: activations, intermediate results and the outputs it is producing. MLX counts these as part of its active memory, together with the model's weights.

- Measured, never estimated: the peak MLX reports for one layer step, less what was held when the step began and less the weights the step had in hand (`crates/foundation/memory-policy/src/ledger/mlx_working_buffers.rs`).
- Reading a prompt needs far more than writing a reply, because a prompt step handles a whole span of tokens and a reply step handles one.
- The memory ledger sets them aside as the account `mlx_working_buffer_bytes`.

## `mlx_buffer_cache`

When a step finishes, MLX does not hand its buffers back to the system. It keeps them and gives them to the next step that asks for the same size, which is far cheaper than allocating again. Those kept, currently unused buffers are the `mlx_buffer_cache`.

- MLX's own name for it is "cache memory": `get_cache_memory`, `set_cache_limit` and `clear_cache` in `mlx/memory.h`, backed by the class `BufferCache` in MLX's Metal allocator.
- It is not counted in MLX's active memory, but it does occupy memory under the limit.
- MLX starts releasing it once active memory plus the cache reaches its collection line, which stock MLX places at 95% of the working set Metal recommends for the Mac. Past that line no buffer is reused and every step allocates afresh; a large sparse model then replies at about a third of its speed.
- While a reply is written, the memory ledger keeps room for it in the account `mlx_buffer_cache_bytes`: one reply step's measured working buffers, taken from what would otherwise be kept experts, and never more than fits. Reading a prompt keeps no such room yet.
- Its limit starts equal to the memory limit. Once a streamed model has measured one prompt step's `mlx_working_buffers` at load, the limit drops to that step, at least 1 GiB (`measured_mlx_buffer_cache_limit_bytes`). Without that bound, reading a long prompt kept every freed step buffer beside a full `astro_expert_cache`, MLX held the whole limit wired, and macOS swapped (issue #2533).

## `metal_kernel_cache`

MLX compiles its GPU programs (kernels) the first time each is used on a Mac. macOS stores the compiled programs on storage and keeps them across restarts, which MLX's documentation calls the Metal kernel cache.

- It is small and it is not memory the limit governs.
- Its only visible effect is that the first use of a kernel after an MLX or macOS update is slower.
- It is not the `mlx_buffer_cache`. "Kernel cache" must never be used for memory MLX holds while a model runs.

## The n-gram table of Qwen3.8-Flash-Next

Qwen3.8-Flash-Next adds to its second layer rows looked up in a table of about 30 GB, chosen by hashing the last two and three tokens. The table is model weights, not a cache, and it has no room under the memory limit.

- Each token's rows are read from storage when the token is processed (`crates/families/text/qwen35/src/qwen4_exp/ngram_table.rs`); only those rows become MLX arrays, for that step.
- It is never held in the `astro_expert_cache`. Rows read again come from the `macos_file_cache`, which macOS sizes and empties on its own.

## How the limit is divided

For a sparse model being streamed, the memory limit a person sets is divided as:

```
limit = core weights that always stay in memory
      + the conversation's stored state
      + mlx_working_buffers for one step
      + room for the mlx_buffer_cache (while a reply is written)
      + room left unused for the system's wired ceiling
      + slots for weights being read
      + margin
      + astro_expert_cache
```

The `astro_expert_cache` is always the remainder. Whatever the other accounts need is taken first, and it gets what is left.

## Every other cache

| Name | What it holds | Owner | Where it lives |
|---|---|---|---|
| `astro_ssd_prompt_cache` | Blocks of processed prompts saved on the SSD, so a later request that starts the same way skips that work | Astronomical | Storage (`~/.astronomical/cache`) |
| `astro_decoder_state_cache` | One request's attention keys and values and recurrent state, carried from token to token | Astronomical | Memory |
| `astro_neural_engine_program_cache` | Compiled Neural Engine programs kept between loads | Astronomical | Storage |
| `astro_image_prefix_key_value_cache` | Qwen-Image's prefix keys and values reused across denoising steps | Astronomical | Memory |
| `astro_native_build_cache` | Downloaded and built native sources such as MLX | Astronomical | Storage (`~/Library/Caches/Astronomical`) |
| `coreml_model_cache` | Core ML's own compiled copy of a model | Core ML | Storage |
| `huggingface_hub_cache` | Downloaded model files in Hugging Face's folder layout | Hugging Face | Storage |
| `macos_file_cache` | File pages macOS keeps in memory after a read | macOS | Memory outside the limit |
| `github_actions_cache` | Saved CI state restored by later CI runs | GitHub | GitHub's servers |
| `sccache` | The compiler cache tool of that name, and what it stores | Mozilla's sccache | Storage and CI |

- **Small in-process memos** need no table entry, but their name says what they hold, such as `vision_padding_zero_cache` or `safetensors_header_cache`. A memo may also be called a memo.
- **Names other projects define** keep their spelling: MLX's `set_cache_limit`, `clear_cache` and `get_cache_memory`; OpenAI's `cached_tokens` and its request fields `prompt_cache_key`, `prompt_cache_options` and `prompt_cache_retention`; Anthropic's `cache_read_input_tokens`, `cache_creation_input_tokens` and `cache_control`; HTTP's `Cache-Control`; OptiQ's and diffusers' `kv_cache`. Where Astronomical's code calls them, it wraps or imports them under the formal name.
- **config.json** leaves out the `astro_` prefix, because the whole file is Astronomical's: the `astro_ssd_prompt_cache` settings are the `ssd_prompt_cache` section. Everything else, including IPC, Astronomical's own REST fields and logs, carries the full name; a field of the OpenAI or Anthropic request formats is never renamed, because clients that send it would be refused.
- **Folders on disk** keep the names they were created with, so an update never orphans what a person already has.
- **"Cached"** as an adjective is replaced by what it means: "held in the `astro_expert_cache`", "restored from the `astro_ssd_prompt_cache`", or "memoized".

## The system's wired ceiling

macOS wires (pins in memory) what the graphics processor works on, and its kernel publishes how much it will keep wired (`vm.global_user_wire_limit`). Once the memory wired across the whole system passes that limit, macOS no longer keeps all of MLX's memory wired and every piece of graphics-processor work slows. That total includes what the kernel and other applications have wired, so it changes as a person opens and closes applications.

- After every measured step the engine reads the system's wired total and moves the ledger account `system_wired_reserve_bytes` by how far the total is over or under the line. The reserve comes out of what would otherwise be kept experts, shrinks to what fits, and is zero on a Mac with room to spare.
- The memory limit a person sets keeps its full range. Near its top the engine uses a little less than the limit allows, and Observatory shows how much as "Left free for macOS and other apps".
- The MLX-only probe `cargo measure-memory-fill-speed` measures the line without any Astronomical engine code.

## On screen

Screens do not show these names; they use plain words, such as "Prompt cache" in Settings or "Restoring prompt cache" in the menu bar. On Observatory's Overview, where the limit is divided:

| On screen | Formal name |
|---|---|
| "Room for the rest of the model" | The capacity of the `astro_expert_cache` (kept layers on a dense model) |
| "Room to work" | `mlx_working_buffers`, the room kept for the `mlx_buffer_cache` while a reply is written, and any lease |
| "Safety margin" | The margin |
| "Left free for macOS and other apps" | `system_wired_reserve_bytes`, the room kept under the system's wired ceiling |
