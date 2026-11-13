# Neural Engine prefill (experimental)

Every Apple silicon Mac has a Neural Engine, a part of the chip built for AI work. Astronomical normally runs models on the graphics chip alone. With this setting on, it shares part of the work of reading a long prompt between the graphics chip and the Neural Engine, which can make long prompts faster on some models.

It is off by default, and it has real costs in memory and disk space. Read the costs below before turning it on.

## The short version

- **What gets faster:** reading long prompts (called *prefill*). Writing the reply does not change.
- **How much:** about 1.1× to 1.3× on a 9B model. Smaller models gain less or nothing.
- **Which models:** dense models only. Mixture-of-experts models do not benefit, because their experts stay on the graphics chip.
- **What it costs:** extra memory (1.8 GB to 15 GB depending on the model) and about 10 GB of disk per model.
- **You do not have to guess:** Astronomical measures each model on your Mac and turns the feature off for a model that does not gain.

## Turn it on

Set `experimental_neural_engine_prefill_enabled` under `chunking` in `config.json`, for every model or for one model. In Observatory it is the switch in the **Experimental** section of the **Tuning** page. It applies the next time the model loads. See [Configuration](configuration.md) for where the file is and how per-model settings work.

When the feature cannot run, the model keeps running on the graphics chip, and `/v1/status` and the Overview's Neural Engine panel say why.

## What it does

A long prompt is read in pieces of 2,048 tokens. For each full piece, the Neural Engine takes 47 percent of the MLP work and the output gates of the DeltaNet layers, while the graphics chip does the rest.

Only full 2,048-token pieces use the Neural Engine, so short prompts see little or no gain.

## What was measured

On Ornith-1.5-9B, on one M5-class Mac, with 10,000-token Shakespeare prompts sent through the app with the prompt cache on:

| | Graphics chip alone | With the Neural Engine | Gain |
| --- | --- | --- | --- |
| Reading the prompt | about 1,207 tokens per second | 1,362 to 1,444 tokens per second | 1.13× to 1.2× |
| Direct model runs, without the cache's work | | | about 1.27× |
| Writing the reply | no change | no change | none |

The first long prompt after a model loads sees less of the gain.

## The costs

### Memory

The Neural Engine's programs need memory of their own. The amount depends on the model and is measured on your Mac the first time the programs run:

| Model size | Extra memory |
| --- | --- |
| 0.8B | 1.8 GB |
| 2B | 3.6 GB |
| 9B | 15 GB |

- **It counts against your memory limit.** A piece of the prompt uses the Neural Engine only when the memory the model is expected to reach, plus this cost, fits under your limit. A long prompt can therefore start on both chips and finish on the graphics chip.
- **A model is refused only when its programs could not fit even for one piece.** On the 9B model that means a limit below about 20 GB.
- **Leave a few GB of room.** With a limit just above that point, `/v1/status` says `active` while the pieces actually run on the graphics chip. The worker log records each change of chip.
- **The memory is given back** 10 to 20 seconds after work stops.

### Disk space and preparation time

- **Disk:** a model keeps about 10 GB of compiled programs in the `neural-engine-programs` folder of the app's state folder. macOS keeps its own copies in the app's cache folder as well.
- **You are not kept waiting.** A model answers on the graphics chip as soon as it loads, while its programs are prepared in the background. The menu-bar app shows a banner while they prepare and another when they start being used.
- **Preparation gets quicker.** On a 9B model with its compiled programs already on disk, preparation took 94 seconds on the first load after the app itself changed, then 43, 13 and about 3 seconds on the next loads, as macOS rebuilt its own compiled copies.
- **The very first preparation does more.** It writes and compiles the programs, then measures them once the graphics chip has been idle for 30 seconds. Measuring takes about 3 seconds and delays no request. A request restarts that 30-second wait, so a steady stream of requests holds the programs back until there is a pause, and 1.2 GB of memory is held for them meanwhile.
- **Measuring happens on the first two loads.** Once the two readings agree, the result is kept until macOS is updated or the Mac changes.
- **A request that arrives during measuring** cancels the measurement and starts at once. The programs' memory can stay in use beside it for up to 20 seconds, which can go past a tight memory limit.

### After the Mac has been idle

The Neural Engine's programs go to sleep 10 to 20 seconds after their last use, and waking them takes time.

- A prompt sent after 10 seconds of idle starts with one piece on the graphics chip alone. This is because a first piece on both chips took 380 to 600 ms longer than a warm one, and the same programs waking on the second piece took only 140 to 180 ms longer.
- Over a whole 20,000-token prompt sent after 25 seconds of idle, a 9B model still ran 1.11× the speed of the graphics chip alone. A 2B model ran 0.94×, against 1.11× for prompts sent in quick succession.

So a small model gains only when its long prompts arrive close together.

### Slightly different output

Part of each prompt is calculated in a lower-precision number format (fp16), so a reply can differ slightly from one made on the graphics chip alone. In six exact-answer tasks over 13,000-token prompts, every answer stayed correct, though on one coding task the answer ignored "output only the code".

## How Astronomical decides whether a model gains

Astronomical finds this out for each model on your Mac.

1. **It compares.** During a model's first long prompts, full pieces take turns between the graphics chip alone and both chips. This costs those prompts a few percent.
2. **It needs enough evidence:** at least three prompts and 15 comparisons, which is eight 10,000-token prompts or three 25,000-token ones.
3. **It decides.** A model must gain at least 2 percent to keep its programs. Otherwise they are released, the status says why, and later loads skip them until a macOS update.

To have a model measured again sooner, remove `program-speed.txt` from that model's folder under `neural-engine-programs`.

Results through the app:

| Model size | Measured speed | Outcome |
| --- | --- | --- |
| 9B | 1.28× | Kept |
| 2B | 1.03× to 1.11× | Kept |
| 0.8B | 0.90× to 0.98× | Turned off |

A 2B model whose first prompts all arrive after idle spells can read too low and be turned off. That costs only the few percent it would have gained on prompts sent in quick succession.

## Where to see it

- `/v1/status` reports the measured figure as `measured_speed_percent`.
- The menu-bar item, the **Tuning** page and the Overview's Neural Engine panel show the same figure.
- The Overview panel also shows the share of prompt tokens the Neural Engine actually ran, and an estimate of the time saved since the app started.

The time-saved estimate counts only prompts that began with the programs already awake, because a prompt that has to wake them can run slower overall. It can therefore undercount for a large model, and it never claims time that a small model lost.

Performance attribution, a diagnostic setting, removes the speed gain while it is on, because it adds synchronization.
