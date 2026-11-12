# Configuration

Most people never need this page: Observatory's **Settings** and **Tuning** pages change every setting described here. This page is for people who want to know what those pages write, or who edit the file by hand.

Each instance keeps its settings in `config.json` in its state folder, which is `~/.astronomical` for the installed app.

## The configuration file

On first launch each instance creates its own versioned `config.json` and adjacent local JSON Schema. Open Models in Observatory to install a curated model under that instance's automatic models directory. A model installed that way can be removed again with **Remove** on its line under On this Mac: its folder moves to the Trash, and copies you placed in other folders are never offered for removal. When an existing unversioned configuration can be migrated without changing its behavior, Astronomical preserves its exact bytes once as `config.legacy-v0.json` before atomically replacing `config.json`. An unsafe migration leaves the original configuration unchanged and creates no backup. Additional existing model locations remain available as recursive scan roots:

    {
      "$schema": "./astronomical-config.schema.json",
      "schema_version": 1,
      "runtime": {
        "model_directories": ["/path/to/models"],
        "maximum_mlx_memory_gb": 16
      },
      "prompt_cache": {
        "enabled": true,
        "maximum_size_gb": 50
      }
    }

The memory value uses the gigabyte Apple uses to market the Mac: 1 GB = 1,073,741,824 bytes (2³⁰). Remove `maximum_mlx_memory_gb` to use the Mac-reported MLX ceiling. Per-model entries are keyed by the model folder name or by its full Hugging Face ID, such as `organization/model`; models installed through Observatory's Models page or `astronomical models download` record their full ID, so an entry for a different organisation's model of the same name does not apply to them, and an entry keyed by the folder name wins over one keyed by the full ID. Per-model entries can set a lower context ceiling, generation defaults, chunking overrides, and multi-token-prediction depth. Explicit request values override generation defaults. Overrides for temporarily absent models remain dormant and are reported as unmatched rather than preventing other models from serving. `astronomical validate config` says what each entry does against the models installed now: which model it is used for, why it is not used, or that it is waiting for a model you have not downloaded yet. Settings only change chat models. Global chunking controls remain available under `chunking`, and each model may override individual values. See the [version 1 configuration schema](https://aosama.github.io/astronomical/schemas/config/v1/astronomical-config.schema.json) for the complete contract.

## Edit the configuration

Observatory's **Settings** and **Tuning** pages edit every setting in that instance's `config.json`. Settings holds what applies to the whole instance. Tuning lists your models with **All models** first, which holds the execution defaults every model starts from, and shows the chosen model's generation defaults, context limit and advanced execution settings. Click the (i) beside a setting or a section for a plain explanation, its `config.json` key and when a change takes effect; a value that differs from its default shows an arrow that returns it to the default. A Save bar appears while either page holds unsaved edits. Saving validates the whole document, refuses to overwrite a file that changed since the page loaded it, and reports whether the change applied, is waiting for a running request, or needs a restart. The ⋯ menu previews the file, applies the saved file, or resets everything; reset keeps the old file beside the new one as `config.previous.json`. The pages are generated from the version 1 schema, whose `x-astronomical-layout` member arranges them. **Maximum model RAM** on Settings has its own Apply button, applies the memory ceiling at once and writes `runtime.maximum_mlx_memory_gb`; the generic row for that setting is left out so there is one place to change it.

## Metal residency refresh

macOS can release the Metal residency of wired memory while the GPU is idle, so model weights may have to be made resident again before the next prompt. Set `metal_residency_refresh_enabled` under `runtime`, or use the switch on Observatory's **Settings** page, to have the inference worker re-request residency once per second. It is off by default, changing it restarts the worker, and the renewal can keep that memory wired through long idle periods.

## Idle model unload

A loaded model is unloaded after it has gone unused for `idle_model_unload_seconds` under `runtime` (60 by default), which frees its memory for the rest of the Mac; the next request loads it again. Set it to 0 to keep a model loaded until another one is requested. The change applies at once from Observatory's **Settings** page.

The experimental Neural Engine setting has [its own page](neural-engine-prefill.md).
