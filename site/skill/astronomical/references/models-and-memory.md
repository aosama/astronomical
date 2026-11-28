# Models and memory

## Finding out

- `astronomical models list`: what this Mac has.
- `astronomical models supported`: what this release can download, and its size.
- `astronomical status`: the default model, the loaded one, and whether it
  drafts with a draft model (`drafter:`), which makes replies quicker.
- The address's `/v1/models` list (see `references/local-api.md`) also says
  what each chat model accepts, such as pictures.

Model names name one specific build. Copy them from these listings.

## Getting, changing and removing

`astronomical --help` shows how to download a model and how to change the
default. Observatory's models page does both with a button, and removes
models. Before a download, tell the person its size, check there is that much
free disk space, and wait for their yes. Leave removal to the person.

## The memory limit

The person sets one number, the memory limit: the most memory models may use.
It is in Observatory's settings, where the person can see and change it. A
lower limit leaves more room for the rest of the Mac.

- **A model that fits under the limit stays fully in memory.** That is
  fastest.
- **A model larger than the limit still runs.** Astronomical keeps as much of
  it in memory as the limit allows and reads the rest from the SSD when
  needed. That is slower, never refused, and never lower in quality. It works
  best with large models that use only a small part of themselves for each
  word, which the model's description usually says.
- **It switches by itself** as a conversation grows or shrinks.
- **Some things must always fit:** the parts every word uses, the
  conversation so far, and working space. A very small limit can be too small
  for a very long conversation.

## Draft models

Some models carry a draft model: a small helper that guesses the next few
words, which the main model checks all at once. Replies arrive sooner, most
of all for code, and every word is still the main model's own. Their names
contain `DFLASH` or `DFLASH2`.

- **It needs an M5-family Mac.** A Mac without an M5 chip runs the model
  without the helper, at its normal speed.
- **A model read partly from the SSD drafts too,** when its memory limit
  leaves room for the draft model beside it. Drafting there pauses on its own
  when guesses stop paying off, and resumes later. If the limit leaves no
  room, the model runs at its normal speed and `astronomical status` says so.
- **Some models predict ahead by themselves.** A model with its own
  prediction head, such as Qwen3.8-Flash-Next, guesses one word ahead by
  default, with no separate draft model. Ternary Bonsai 2 9B ships a head
  too, but it is off by default because it does not speed that model up;
  setting `chunking.multi_token_prediction_draft_tokens` turns it on.
- **Some requests never use it:** a picture in the prompt, structured output
  and a repetition penalty all decode one word at a time.
- **`astronomical status` shows what it is doing.** The `drafter:` line says
  whether the loaded model drafts or why not, and the line under it says how
  many guesses the last reply kept. Most kept means most gained.

## Helping someone choose

The welcome and Observatory suggest a model that suits this Mac; start from
that suggestion. Then ask what they want to do and compare the model sizes in
the listings with their memory limit.

- Fits under the limit: fastest; best for everyday use.
- Larger than the limit: works, more slowly. Say so plainly.
- Drawing and speech use their own models, which `astronomical image` and
  `astronomical speak` pick by default.

Give one recommendation with its reason, and let the person decide.

A model that sits unused is unloaded after a while to free memory, and the
next request loads it again. A setting in Observatory controls how long.
`astronomical unload` frees the memory now, when the person needs it for
something else; it refuses while a request is running.
