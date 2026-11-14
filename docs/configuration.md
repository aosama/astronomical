# Configuration

Most people never need this page. Observatory's **Settings** and **Tuning** pages change every setting described here, and they are the easier way to do it. This page is for people who want to know what those pages write, or who edit the file by hand.

## Where the settings live

Astronomical keeps its settings in one file, `config.json`, in its state folder. For the installed app that folder is `~/.astronomical`.

The app creates the file the first time it starts, with a second file beside it, `astronomical-config.schema.json`, that describes every setting. Editors that understand JSON Schema use it to check your edits as you type.

A small file looks like this. Keep the `$schema` line: the app refuses a file that states its `schema_version` without it, and does not start.

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

## The settings most people change

| Setting | What it does | If you leave it out |
| --- | --- | --- |
| `runtime.maximum_mlx_memory_gb` | The most memory models may use, in GB. | The limit your Mac reports. |
| `runtime.model_directories` | Extra folders to search for models you already have. Subfolders are searched too. | No extra folders. |
| `runtime.default_model` | The model used when a command or request does not name one. | The built-in default, Qwen3.5-2B-4bit. |
| `runtime.idle_model_unload_seconds` | How long an unused model stays loaded. | 60 seconds. |
| `runtime.metal_residency_refresh_enabled` | Keeps a model's memory ready for the graphics chip while idle. | Off. |
| `prompt_cache.enabled` | Saves work on prompts to the SSD so repeated text is not read again. | On. |
| `prompt_cache.maximum_size_gb` | The most disk space that saved work may use. | 50 GB. |

One GB here is the gigabyte Apple uses when it describes a Mac's memory: 1,073,741,824 bytes (2³⁰).

The [version 1 schema](https://aosama.github.io/astronomical/schemas/config/v1/astronomical-config.schema.json) lists every setting.

## Where models are kept

- **Models you download in Observatory** go into a models folder that Astronomical manages for you. You can remove one with **Remove** on its line under **On this Mac**; its folder moves to the Trash.
- **Models you already have elsewhere** stay where they are. Add their folder to `model_directories` and Astronomical finds them. It never offers to remove models in those folders.

## Settings for one model

You can give a single model its own settings under `models`:

    "models": {
      "organization/model": {
        "limits": { "maximum_context_tokens": 32768 },
        "generation_defaults": { "temperature": 0.7 }
      }
    }

**What you can set for a model:**

- a lower limit on the length of a conversation,
- defaults for how it writes, such as temperature,
- how a long prompt is split into pieces (chunking),
- how many tokens a small helper model may draft at once, when one is installed beside the model.

**How a model is matched:**

- Name the entry after the model's folder, or after its full Hugging Face ID, such as `organization/model`.
- Models downloaded through Observatory or `astronomical models download` record their full ID. An entry for another organization's model of the same name does not apply to them.
- If one entry uses the folder name and another uses the full ID, the folder-name entry wins.

**Good to know:**

- These settings only change chat models.
- A value sent with a request overrides the model's default for that request.
- An entry for a model you have not downloaded does nothing and causes no error. It waits until that model appears.
- Settings under `chunking` at the top level apply to every model. A model's own entry can override any of them.

To see what each entry does with the models you have now, run:

    astronomical validate config

It says which model each entry is used for, why an entry is not used, or that it is waiting for a model you have not downloaded yet.

## Editing in Observatory

**Settings** holds what applies to the whole app. **Tuning** holds what applies to models: **All models** comes first, with the defaults every model starts from, followed by each of your models with its writing defaults, conversation limit and advanced settings.

- **Click the (i)** beside a setting or section for a plain explanation, its `config.json` key, and when a change takes effect.
- **An arrow** appears beside a value that differs from its default. Click it to return to the default.
- **A Save bar** appears while either page has unsaved edits.
- **Saving checks the whole file first.** It will not overwrite a file that changed after the page loaded it. It then tells you whether the change applied, is waiting for a running request to finish, or needs a restart.
- **The ⋯ menu** previews the file, applies the saved file, or resets everything. A reset keeps your old file beside the new one as `config.previous.json`.
- **Maximum model RAM** on Settings has its own **Apply** button. It changes the memory limit at once and writes `runtime.maximum_mlx_memory_gb`. It is the only place on the page that changes this setting.

Both pages are generated from the version 1 schema; its `x-astronomical-layout` member arranges them.

The Mac App Store version runs inside the macOS App Sandbox, which lets it open only its own storage. It keeps models, downloads and compiled programs there, so its **Settings** page has no **Locations** section and the three folder settings have no effect in that version.

## Unloading an idle model

A loaded model is unloaded after it has gone unused for `idle_model_unload_seconds` (60 by default). That frees its memory for the rest of your Mac, and the next request loads it again. Set it to 0 to keep a model loaded until you ask for a different one. A change made on the **Settings** page applies at once.

## Keeping memory ready while idle

macOS can stop holding a model's memory ready for the graphics chip while the chip is idle. The model then has to be made ready again before the next prompt, which takes time.

Set `metal_residency_refresh_enabled` under `runtime`, or use the switch on the **Settings** page, to have Astronomical ask macOS once per second to keep that memory ready. It is off by default. Changing it restarts the part of the app that runs the model. While it is on, that memory can stay reserved through long idle periods.

## Updating from an older file

When Astronomical finds a `config.json` from before settings files had a version number, it updates the file if it can do so without changing how the app behaves. It first saves an exact copy of the old file, once, as `config.legacy-v0.json`. If the update is not safe, it leaves your file as it is and makes no copy.

## Related

The experimental Neural Engine setting has [its own page](neural-engine-prefill.md).
