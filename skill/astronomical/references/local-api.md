# The local API, for programs

Use this when writing a program. From a terminal, the `astronomical` command
is simpler.

- **The address** is `http://127.0.0.1:6732` for the installed app and
  `http://127.0.0.1:6733` for a build from source; Observatory's connect page
  confirms it. It answers only on this Mac and checks no key.
- **It speaks the OpenAI and Anthropic APIs:** chat (with streaming, tools,
  pictures and structured output), embeddings, image generation, speech, and
  model listing. Decision models have their own endpoint, listed under
  each decision model's supported endpoints in `/v1/models`;
  `references/decision-models.md` explains the questions and answers.
- **`/v1/models` is the map.** It lists every model served, with its exact
  name, context window, the endpoints it serves and what input it accepts.
  Read it rather than hard-coding names.

## Habits that work

- **Expect a slow first request;** it loads the model. Set long timeouts.
- **Send one request at a time.** One reply is written at a time.
- **Read error messages.** Unsupported options and unknown model names are
  refused with a JSON error that names the problem.
- **Keep a fixed beginning.** Astronomical reuses its work on a prompt that
  starts like an earlier one, so put the fixed instructions first and what
  changes last.
- **Give memory back when you are done.** A loaded model holds many gigabytes
  until Astronomical unloads it after a quiet spell. `POST /v1/control/unload-model`
  frees it now, and the next request loads it again. It refuses with `busy`
  while a reply is being written, and says `nothing_loaded` when no model is
  loaded. Do it only when the person asks or a job of yours is finished.

The site's local API page, https://getastronomical.com/local-api.html, has
settings for specific clients.
