# Qwen 3.8 Flash (`qwen4_exp`) architecture evidence

Measured facts about the Qwen 3.8 Flash family and the rulings Astronomical applies to it. Every number here was read from published configuration documents, weight maps, and safetensors headers, or from the public reference implementations named in the licensing section. No weight payload was downloaded to write this document.

## Identity and naming

- Wire identity: `qwen4_exp` everywhere — discovery, catalog, status, routing, and attribution. The text configuration reports `qwen4_exp_text`.
- Display name: "Qwen 3.8 Flash". Marketing text never appears in code, module names, or wire fields.
- Precedent: `k2_horizon_mova` and `modernbert` match `model_type` rather than marketing text, and this family follows the same rule.
- The checkpoint architecture string is `Qwen4ExpForConditionalGeneration`. Upstream calls the architecture a Qwen4 preview; the published checkpoints brand it Qwen 3.8 Flash Next.

## Measured geometry

Shared by every published artifact (the last row of the variant matrix below):

| property | value |
| --- | --- |
| decoder layers | 48: 36 `linear_attention`, 12 `full_attention`, `full_attention_interval` 4 |
| hidden size | 2560 |
| full attention | head dimension 256, 24 query heads, 2 key-value heads, gated output (`output_gate_type: sigmoid`), partial rotary factor 0.25 |
| rotary | interleaved M-RoPE, sections 11/11/10, theta 10,000,000 |
| linear attention | 16 key heads of 128, 48 value heads of 128, convolution kernel 4, `A_log` and `dt_bias` per value head |
| mixture of experts | 512 routed experts (288 in the REAP-pruned artifact), top 10, one shared expert, intermediate 640 |
| hyper-connections | 4 streams, low-rank 320, one attention and one MLP connection per layer |
| sparse attention indexer | budget 2048, compression ratio 4, 4 heads of 128, 1 index key-value head |
| n-gram embedding table | 320,001,536 rows of width 160, about 51 billion parameters, one table at decoder layer index 1 |
| vocabulary and context | 248,320 tokens, 262,144 positions |
| vision | 27-block tower, hidden 1152, patch 16, temporal patch 2, spatial merge 2, output 2560 |

## The five net-new subsystems

1. **Hybrid gated-delta layers.** The linear-attention tensors (`A_log[48]`, `dt_bias[48]`, `conv1d[10240, 4, 1]`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) match the gated-delta structure Astronomical already executes for Qwen 3.5. Reuse, not new math.
2. **Qwen Sparse Attention.** Twelve full-attention layers carry `self_attn.indexer.index_qk_proj`, `indexer.q_layernorm`, and `indexer.k_layernorm`. The indexer scores keys and attends to at most `indexer_budget` (2048) of them, and the serving path keeps an auxiliary index-key cache beside the ordinary key-value cache.
3. **N-gram embedding table (PLE).** One hashed lookup table of 320,001,536 rows consulted at decoder layer index 1 (`ple_layer_ids: [2]`, one-based). Sixteen rows per token: eight hash heads over bigrams and eight over trigrams. About 32 GB of the inspected 4-bit artifact — roughly 44 percent of its bytes.
4. **Hyper-connections.** Every layer reads and writes four residual streams through low-rank gated mixing and learned injection, replacing a plain residual add.
5. **Stacked expert bank.** `switch_mlp.gate_proj` and `switch_mlp.up_proj` as `U32[288, 640, 320]`, `switch_mlp.down_proj` as `U32[288, 2560, 80]`, with a BF16 router `mlp.gate.weight[288, 2560]`, one shared expert, and `shared_expert_gate`.

## Cross-variant matrix

Eight published artifacts share `model_type: qwen4_exp`. Measured differences:

| axis | published values |
| --- | --- |
| expert count | 512, and 288 in one REAP-pruned artifact |
| quantization | native BF16 upstream, 2-bit group 32, 3-bit group 32, 4-bit group 64, 4-bit group 32 MXFP4 |
| per-tensor override entries | 0, 128, 228, 272, 746, 760 |
| packed columns for a 2560-wide projection | 160 at 2-bit, 240 at 3-bit, 320 at 4-bit |
| indexer projection storage | packed unsigned 32-bit in three artifacts, plain BF16 in two |
| convolution weight axes | `[10240, 4, 1]` in three artifacts, `[10240, 1, 4]` in one |
| lookup table naming | `ngram_embedding.shard_N.` in five artifacts, `ngram_embedding.shards.N.` in two, plus one that also publishes `ple-store.json` byte ranges |
| lookup table fields | weight, scales, and biases, or weight and scales only |
| multi-token prediction | tensors present in three artifacts and absent in five, including one that declares one layer |
| shards and payload | 14 to 131 shards, about 67 GB to 99 GB converted and 360 GB upstream |
| shared structure | everything in the geometry table above |

The last row is the family. Every other row is packaging, and no packaging detail may become an engine assumption.

## The n-gram lookup rule

Public reference implementations document the rule completely. It is restated here because the whole streaming design depends on it:

- Sixteen hash heads: heads 0–7 hash bigrams, heads 8–15 hash trigrams. Each head owns one prime-sized table, and `head_dim` is `ple_embed_dim` divided by 16, which is 160.
- Per-head table sizes are successive primes strictly greater than `ngram_vocab_size_base - 1` (20,000,000): the head's global ordinal selects which prime. For this family the ordinals are 0 through 15, and the sixteen primes sum to 320,001,446, which pads by 90 rows to the published 320,001,536 — exactly 128 shards of 2,500,012 rows. The arithmetic closes against the published manifest.
- Hash multipliers come from a seed (1234 when configuration omits one) plus 10,007 times the PLE layer ordinal, mixed through splitmix64, reduced into a range that keeps `token × multiplier` inside signed 63 bits, then forced odd.
- A row index for one head is `(mixed mod prime_size) + head_offset`, where `mixed` is the XOR of each window token multiplied by its position multiplier, computed with wrapping signed 64-bit arithmetic and non-negative remainders.
- The window never crosses an end-of-sequence boundary: tokens before a boundary read as the end-of-sequence identifier, and the context resets after one.
- The consuming layer gates the looked-up values against the hidden state and adds a short depthwise convolution term; layers after it never consult the table again.
- The public serving implementation prefetches layer-two rows while layer one computes. Reproducing that overlap is a design requirement for the row reader, not an optional optimization.

## Hyper-connection algebra

The published checkpoints use the gated-residual variant of the Hyper-Connections scheme (arXiv 2409.19606), with per-branch normalization:

- State between layers is four streams of 2560, stored stream-major as 10240.
- Normalization is a grouped RMSNorm over each 2560-wide stream computed in float32, with a per-element affine of `1 + weight` over all 10240 elements — which is why `hc_norm.weight` is `[10240]`.
- Mixing into a block input: normalize, project down 10240 → 320, apply SiLU scaled by one over the stream count, project up 320 → 10240, apply sigmoid, multiply into the normalized streams, and average over the four streams.
- Combining a block output: an injection weight of `2 · sigmoid(project(normed) / stream_count)` per stream scales the block output before it is added to each stream.
- The tensor shapes confirm the wiring: `input_mix_weight_down` `[320, 10240]`, `input_mix_weight_up` `[10240, 320]`, `block_inject_weight` `[4, 10240]`, all packed 4-bit in the inspected artifact.
- An average-pooling variant (mean over streams on the way in, broadcast add on the way out) exists in the reference and must remain expressible, selected by the weight inventory rather than assumed.

## Reference implementations and licensing

| source | license | role |
| --- | --- | --- |
| `mlx_vlm` `qwen4_exp` model package | MIT | executable MLX semantics, row-addressable lookup storage, sparse-attention kernel |
| `vllm` `qwen4_exp` model package | Apache-2.0 | independent semantics cross-check, n-gram hashing reference, hyper-connection reference |
| MLX language-model library | MIT | no support for this architecture; Astronomical would be the first Rust engine for it |
| training framework `engram` documentation | public docs | independent statement of the hashing rule and per-head tables |
| serving engine day-zero write-up | public article | prefetch pattern and access-pattern analysis |

Ruling: consulting these sources for semantics is expected and recorded. Astronomical's implementation is original Rust expressing the same mathematics; if any source is ever copied rather than re-derived, the copy carries that source's license notice and attribution. Weight payloads are never redistributed or committed.

## Community conversion policy

The catalog already lists community-published conversions beside upstream artifacts, with provider identity and immutable revision provenance recorded at publication. That precedent extends here: community REAP-pruned and requantized conversions may be advertised once the family executes, with their provenance recorded, and the upstream checkpoint — which ships HF-format tensors rather than MLX layout — is treated as a conversion source rather than a directly loadable artifact.

## Mechanism owner placement

The indexer, the lookup row store, and the residual stream mixer are mechanisms, not family property: a reference server drives this architecture and a different vendor's architecture through one path, and that vendor's artifacts carry the same indexer and hyper-connections at different numbers. Decision: the three owners live under the family directory for now, behind configuration-agnostic interfaces that never read a family configuration type, and they are promoted to shared owners the moment a second consumer lands. Creating shared structure for a single consumer today would add scaffolding without removing code from the active path.

## Performance evidence

Measured against the published lookup table on one Apple Silicon machine with NVMe storage, reading real manifest-addressed ranges with no model loaded:

| access pattern | median per row |
| --- | --- |
| cold, three piece ranges, one reader | about 217 microseconds |
| cold, one 16 KiB page read | about 75 microseconds |
| cold, three piece ranges, eight concurrent readers | about 26 microseconds effective |
| warm in the page cache | about 2 microseconds |

Bandwidth is not the constraint; operation count and per-operation latency are. The artifact's own documentation reports about 68 GB resident with the table in memory and about 39 GB with the table streamed from NVMe, so streaming the table roughly halves the footprint on a machine twice the size of this artifact. See `docs/performance-optimizations-lessons.md` for the full measurement record.

## Open questions

- The hyper-connection algebra is pinned from the reference implementation and the scheme's paper; the upstream technical report's own text has not been diffed against it because automated text extraction of the report failed. The serving oracle issue verifies arithmetic against checkpoint behavior, which is the stronger check.
- Multi-token prediction variants, image input, and video input are planned separately and carry their own evidence work.
- Whether the MXFP4 profile can execute on this runtime is an open binding question, recorded in the quantization issue.
