# One request, start to finish

This page follows one chat request through Astronomical's memory management, for a contributor who wants to know which code decides what. It describes the streaming engine first, which serves a model that does not fit under the memory limit, and then says how the in-memory engine differs. For what this means to a person using the app, see [How Astronomical works](how-it-works.md).

## Who decides what

- **The memory governor** (`crates/astronomical-memory-policy/src/ledger/governor.rs`) owns the limit. It keeps a ledger with one account per use of memory: the model's fixed weights, the conversation's stored state, the MLX working buffers one step needs, the slots for weights being read, any leases, and a margin of 2%. What is left is the **capacity**: how many bytes of weights may be kept in memory. Nothing sets the capacity. It is always the remainder.
- **The engine** (`crates/qwen35/src/qwen3_5_streaming/model/memory_governance.rs`) tells the governor what happens and applies the capacity each answer carries, in one function, `apply_kept_weight_capacity`. For a model with experts that function sets the astro expert cache's one capacity. For a dense model it gives trailing kept layers back.
- **The astro expert cache** keeps experts up to its capacity and gives back, in its own order, whatever no longer fits. Nothing else limits or empties it.

## When the model loads

1. The model's fixed weights are loaded.
2. One layer is run once over a span of the size the engine will serve, and one token is run through one layer. The peaks are the first MLX working-buffer measurements for the two phases: reading a prompt and writing a reply. A dense model measures one streamed layer the same way.
3. The governor opens with the limit and those measurements, and the idle capacity is applied. A dense model reads in the leading layers there is room for, up to what a prompt would leave.

A measurement the limit cannot hold does not fail the load. The phase stays unmeasured, its capacity is zero, and everything is read from the SSD until a real step has been measured.

## A request arrives

4. The request is validated: token counts and context length. It is not checked against memory and is never refused for memory.
5. `govern_request_start` sets aside the stored state the whole prompt will occupy and enters the reading phase. The capacity drops by the prompt's stored state, the reading MLX working buffers and the layer slots. The astro expert cache, or the dense layers, give back what that displaces, once.
6. If part of the prompt is in the astro SSD prompt cache, it is restored. The blocks need room while they load, so the governor leases it for the restore and takes it back afterwards.

## Reading the prompt

7. The prompt is cut into spans of the configured size. The size was fixed before the request and is never reduced.
8. For each layer of the model, in order:
   - If the layer's experts are not all kept, its page is read from the SSD once and held for every span. The next layer's page is read ahead while this one computes, so at most two pages are in hand.
   - Every span runs through the layer. Each span remembers what it found when it began: its input, the layer's state, and the output it waits for.
   - The engine waits for the layer. Which experts the tokens chose is recorded. The step is measured: what MLX held before and after, and how high it peaked.
   - The governor is told. It records the MLX working buffers, compares its accounts with what MLX reports, and answers with the capacity. The page that was read is kept by the astro expert cache if there is capacity for it, and dropped otherwise.
   - The engine reports "N of M layers done" to the status page.
9. **If a span cannot fit**, MLX reports it at the next wait. The layer is put back to its oldest unconfirmed span. The governor leases as much room again as the step was given, or the shortfall MLX reported if that is more, and the astro expert cache gives back only what the lease displaces. The span runs again. After four attempts the request ends with MLX's error.
10. After the last layer the prompt's tokens count as read, the astro SSD prompt cache blocks are published under a lease for their workspace, and the request moves on.

## Writing the reply

11. `govern_reply_start` enters the reply phase. One token needs far fewer MLX working buffers than a span, so the capacity rises.
12. One plan is made against that capacity: each layer is **kept** (all its experts in memory), **hot** (the experts tokens choose are kept in a table) or **absent**. Layers the plan keeps that are not yet whole are read in before the first reply token.
13. Each token runs through every layer. Kept layers are used directly. For the others only the experts the token chose are read, and hot ones are kept. After each token the step is measured and the governor is told, as in step 8.

## When the request ends

14. `govern_request_end` returns any lease or unsettled retry and enters the idle phase. The streaming engine keeps no stored state between requests, so the capacity rises again. The astro expert cache keeps what it has. A dense model reads back the layers it gave up.

## When the limit changes

15. The governor is told the new limit and answers with the capacity. A lower limit makes the astro expert cache or the dense layers give back at once. A higher one lets a dense model read layers back while idle and lets the astro expert cache keep more as it reads.

## What is measured and what is checked

- MLX working buffers are the peak of a step above what was held around it, less the weights the step had in hand. It is measured at load and at every step and is never estimated from earlier requests.
- At every layer boundary the ledger's accounts are compared with what MLX reports. On the models measured, the largest difference is 64 MB, 0.2% of the limit.
- The status page's memory breakdown is composed from the same accounts.

## The in-memory engine

A model that fits is served by the in-memory engine (`crates/qwen35/src/qwen3_5_resident`), which keeps every weight. Its governor's ledger has every weight as fixed and nothing to stream, so the capacity plays no part and there is nothing to give back.

- At load it runs one prompt chunk of the serving size and one reply token and records what each needed.
- Before each forward it asks the governor one question: do the weights, the stored state after this forward, the forward's MLX working buffers and the margin fit the limit? (`inference_execution/memory_governance/forward_memory.rs`)
- A forward that does not fit is never made smaller. The request moves to the streaming engine, which serves it as described above.
- After each forward it reports what MLX measured, so its ledger is compared with MLX too.

## Where the rules are tested

- The ledger is plain arithmetic, so whole requests are replayed against it without a model or a GPU in `crates/astronomical-memory-policy/src/simulation`. Each real memory incident is one named scenario there, and they run on every commit.
- Streamed and in-memory serving are compared token for token by `cargo test-paged-resident-parity`, including a question where one span is failed on purpose and run again.
- The journeys under `verification/acceptance/tests/expert_paging/journeys` each assert one outcome a person would notice, on real models.
