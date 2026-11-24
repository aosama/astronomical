# Using local models in coding tools and editors

Any tool that speaks the OpenAI or Anthropic API can use the chat models on
this Mac.

## The quickest way

`astronomical launch` starts a supported coding tool pointed at this Mac for
one session, without writing the tool's settings. `astronomical --help` lists
which tools it supports.

For OpenCode there is also a plugin that keeps every local model in its model
picker: `opencode plugin add opencode-astronomical`.

## Any other tool or editor

Editors such as Zed and Visual Studio Code, and tools `launch` does not
support, connect by hand through their "OpenAI-compatible" or "custom
provider" option. Four things are needed. The site's local API page,
https://getastronomical.com/local-api.html, has checked recipes for some
tools.

1. **The address.** Observatory's connect page shows the exact address of the
   running app. OpenAI-protocol tools take it with `/v1` at the end;
   Anthropic-protocol tools take it without.
2. **The model name.** Copy an `id` from the address's `/v1/models` list.
   Each entry also gives the model's context window; set the tool's limit to
   it.
3. **The key.** None is checked, but many tools refuse an empty one. Any
   placeholder text works; it is not a secret.
4. **Patience on the first request.** It loads the model, which can take a
   while. Give the tool a long timeout.

Editing another tool's settings file changes the person's setup: show them
the change, keep their other settings, and wait for their yes.

## Things that trip tools up

- The address answers only on this Mac.
- A model's listing says which protocols it serves; pick one that matches the
  tool.
- One reply is written at a time; parallel requests wait.
- An option the app does not accept is refused with a JSON error that names
  it. The local API page lists the settings that avoid this for the tools
  it covers.

## Zed

Zed's settings file location depends on the operating system. Do not guess a
path or search another person's home folder. Open Zed's command palette and
run `zed: open settings file`; use `agent: open settings` to manage language
model providers in the settings UI. Zed's [OpenAI-compatible provider
guide](https://zed.dev/docs/ai/use-api-access) has its current configuration
format.

Before changing the person's Zed settings, ask for permission. Then:

1. Get the active Astronomical address from Observatory's connect page. In
   Zed, use its OpenAI-compatible provider with that address and `/v1`.
2. Read `astronomical models list` for models on this Mac and
   `astronomical models supported` for models this app can provide. Use the
   running app's `/v1/models` response for each model's exact ID, context
   window, supported endpoints and input capabilities; do not infer IDs from
   folder names.
3. Add models that advertise OpenAI-compatible chat completions. Set the
   context window and capabilities from the model metadata and Zed's current
   guide. Only enable picture or tool support when the model advertises it.
   Image generation, speech, embeddings and decision models are not chat
   models and do not belong in Zed's chat model list.
4. Preserve the person's other settings and existing models. Do not change
   their default model or favorites unless they ask. Reopen Zed's provider
   settings afterward to confirm the new entries appear.
