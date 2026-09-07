# Experimental Aligned Expert Packs

This package preserves research into sparse-expert on-disk layouts and direct Metal input and output loading.

It is not part of Astronomical production serving, production configuration, public status, the macOS application, or default workspace builds. Astronomical production paging reads standard safetensors.

The package can prepare local research artifacts and exercise isolated data-plane measurements. Those measurements do not establish user-visible serving performance. Any proposal to return this capability to production requires a new representative end-to-end acceptance and explicit approval.

The package intentionally publishes no benchmark result or performance claim.

## Layouts

- **Layer-major packs** (`format_version` 2): one file per expert layer. All experts of each tensor occupy one aligned segment. Kept for A/B measurement against the per-expert layout.
- **Per-expert packs** (`format_version` 3, issue #430): one file per `(layer, expert)` pair. Issue #430 called this generation "v2"; the on-disk version is 3 because the layer-major pack already occupies version 2. Payload bytes are an exact copy of one expert's source slice.

A `--streaming-model` conversion publishes a self-sufficient directory:

```
<output-directory>/
├── manifest.json
├── config.json, tokenizer files, and source weight shards
└── layers/<layer_index>/<expert_id>.apack
```

The converted public identity is `<source-model-id>-expert-streaming`. The experiment target is `Ornith-1.5-35B-A3B-OptiQ-4bit`, producing `Ornith-1.5-35B-A3B-OptiQ-4bit-expert-streaming`. The directory copies the source artifact so existing Qwen discovery can load it independently; the `.apack` files travel with it so the revision is complete without the original source directory.

```
astronomical-experimental-aligned-expert-pack-preparer \
  --model-directory PATH \
  --streaming-model \
  --output-directory PATH/Ornith-1.5-35B-A3B-OptiQ-4bit-expert-streaming \
  --yes
```
