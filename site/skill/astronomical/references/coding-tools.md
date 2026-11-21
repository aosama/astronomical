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
