# Models and memory

## Finding out

- `astronomical models list`: what this Mac has.
- `astronomical models supported`: what this release can download, and its size.
- `astronomical status`: the default model and the loaded one.
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
