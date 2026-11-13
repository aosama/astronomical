# How Astronomical works

This page explains, in plain terms, how Astronomical runs big models in a small amount of memory, why it stays fast, and what it supports today. To install and start using it, see the [README](../README.md).

## The short version

- You set one number: how much memory models may use.
- If a model fits in that memory, Astronomical keeps all of it there.
- If it does not fit, Astronomical keeps the rest on your SSD and reads the parts it needs, when it needs them.
- It never lowers a model's quality to make it fit.
- Everything stays on your Mac.

## Running a model bigger than your memory

Many large models are built as a *mixture of experts*. The model holds a large number of small specialist parts, called experts, but uses only a few of them for each word it writes. Most of the model is idle at any moment.

Astronomical uses that. It does not need the whole model in memory at once, only the parts in use.

- **You choose one memory limit.** On a 32 GB Mac, for example, a limit of 16 GB leaves room for macOS and your other apps.
- **Experts are read from the SSD when needed.** When the whole model does not fit under your limit, Astronomical loads only the experts the model picks for the next word.
- **A model that fits stays fully in memory.** That is the fastest way to run it, so Astronomical does it whenever it is safe.
- **It switches by itself.** A long conversation needs more memory. If that pushes a model over your limit, Astronomical moves the experts to SSD reading. When the model is idle and fits again, it moves them back. You never pick a mode.
- **You can see what it is doing.** The menu-bar item shows your memory limit, whether the model is in memory or being read from the SSD, how memory is being used, how much room is left, how busy the graphics chip is, how far along your prompt is, and how much earlier work was reused.

The switch is all or nothing: a model's experts are either all in memory or all read from the SSD, never a mix of the two.

### What still has to fit

Some things must always fit under your limit:

- the part of the model that every word uses,
- your conversation so far,
- the experts in use right now,
- working space for the calculation.

So not every model fits every Mac. Which models work, and how fast, depends on the model, the length of the conversation, the speed of your SSD and the memory you allow. Astronomical adapts as far as it can; it does not promise that everything fits.

## Why it is fast

The part of Astronomical that runs the model is written specifically for Apple silicon, with speed-ups made for each supported model family.

| What Astronomical does | What you get |
| --- | --- |
| Keeps a model fully in memory when it fits, and reads from the SSD when it does not | The best speed your memory limit allows, with no setting to tune. |
| Works out how much memory the next step needs before starting it | Memory is checked before it is used, and only as much is freed as the next step needs. |
| Reads a long prompt in pieces of a set size, made smaller only when memory requires it | Predictable memory use, and no time spent on trial runs while you wait. |
| Saves its work on a prompt to the SSD | When a new request starts with text it has seen before, it picks up from there. This also covers images it has already looked at. The space used is capped. |
| Chooses each next word on the graphics chip | Less waiting between the graphics chip and the rest of the Mac while it writes. |
| Groups work by expert | Less scattered work for the graphics chip on larger prompts. |
| Reuses prepared calculations, with custom graphics-chip routines where they measured faster | Less repeated setup for every word. |
| Clears finished work after each piece of a prompt | Memory does not creep up over a long prompt. |
| Loads a model only when it is asked for | No model is loaded at startup. Asking for a different model swaps it in. |

## Your model's quality is not reduced

Running in less memory must not quietly make a model worse.

- Astronomical never lowers a model's precision when memory is tight.
- Experts read from the SSD are exactly as the model's author stored them.
- Saved prompt work restores the model to the same state it was in.
- If a model's design or number format is not supported, Astronomical says so and refuses to run it. It does not guess from the file name.

## Private, and easy to watch

Astronomical includes a control page called Observatory, and a menu-bar item. Together they show:

- progress while a prompt is being read and while a reply is being written,
- memory in use now and the highest it has been,
- whether the model is in memory or being read from the SSD,
- how busy the graphics chip is, and the speed of the session,
- how often saved prompt work was reused,
- your memory limit, which you can change while a model is running.

Your prompts, replies, models and saved prompt work stay on your Mac. The app keeps a small record for troubleshooting. That record holds a limited amount of technical detail and fingerprints of requests. It does not hold what you typed, what the model wrote, the details of tool calls, or request headers.

## What it supports today

Astronomical is experimental and deliberately narrow.

- **Macs:** any Mac with Apple silicon (an M-series chip).
- **macOS:** 14 (Sonoma) or later. Support across that range is best effort: Astronomical checks what each Mac can do and uses a standard path where a faster one is not available. Combinations nobody has tested are also best effort.
- **Models:** chat and coding models built on Qwen3.5 and Qwen3.6, FLUX.2 Klein and Qwen-Image for images, Kokoro and Phonon for speech, ModernBERT for embeddings, and small decision models, in the builds this repository has validated.
- **People:** one person on one Mac, with one reply being written at a time.
- **Connections:** only from your own Mac. Other apps can talk to it in the same way they talk to OpenAI or Anthropic services: chat, responses, embeddings, image generation, speech, listing models, and replies that arrive word by word.

A supported model runs on every Apple silicon Mac with a new enough macOS. Graphics chips that can run Astronomical's custom routines use them, because they measured fastest. Where a chip cannot run one, the same request still finishes through Apple's standard routines, at the precision the model specifies. Speed depends on the chip, your memory limit, the model and your SSD.

Astronomical does not include any models. Observatory downloads models from Astronomical's hand-picked list of public Hugging Face models straight to your Mac, and checks each download before offering it. Each model has its own license, which must allow your intended use.

## Connecting other apps

Astronomical listens at one address on your Mac, for both other apps and Observatory:

- the installed app: [http://127.0.0.1:6732/](http://127.0.0.1:6732/)
- a development build: [http://127.0.0.1:6733/](http://127.0.0.1:6733/)

It refuses to listen on any address that other computers could reach.

Point an app that speaks the OpenAI or Anthropic API at that address and choose one of the models it lists. Astronomical loads that model, or swaps to it, when the first request arrives.

To start a coding agent already pointed at the running app for one session:

    astronomical launch claude
    astronomical launch codex
    astronomical launch opencode

Settings for other apps are on the [local API page](https://getastronomical.com/local-api.html).

## The same things, in engineering terms

For readers who want the precise vocabulary:

- The serving path is written in Rust and C++ and runs on Apple MLX through MLX-C.
- "Read from the SSD" is sparse-expert paging: router-selected expert pages move into MLX memory on demand. "Fully in memory" is expert residency in contiguous arrays. The mode always describes the complete sparse payload.
- "Works out how much memory the next step needs" is adaptive memory admission: it projects context growth and observed transient work before allocating.
- "Pieces of a set size" is deterministic prompt chunking, with exact terminal remainders and a bounded reduction when memory capacity requires it.
- "Saved prompt work" is the persistent prompt store: validated prompt state, recurrent snapshots and projected image embeddings in a bounded SSD store.
- "Chooses each next word on the graphics chip" is GPU-native sampling, submitting one token ahead to avoid host synchronization.
- "Groups work by expert" is sorted mixture-of-experts execution: larger routing batches are grouped by expert, and one ordering is reused across projections.
- "Prepared calculations" are compiled MLX graphs, with dedicated Metal kernels for recurrent state, activation, gating and expert combination.
- "Clears finished work" is chunk-level allocator cleanup: it materializes the required decoder state and releases completed intermediates.
- Expert paging preserves the artifact's declared data types and quantization parameters. No quantization is added under memory pressure.
- Troubleshooting records keep bounded metadata and payload digests.
- The API surfaces are OpenAI-compatible chat completions, responses, embeddings, image generation, speech and model discovery, Anthropic-compatible Messages, and server-sent event streaming, over loopback only.
