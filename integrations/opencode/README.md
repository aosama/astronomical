# Astronomical for OpenCode

This plugin puts the models of [Astronomical](https://getastronomical.com/) in OpenCode's model picker. Astronomical runs AI models privately on your Mac; with the plugin installed, every chat model the app serves shows up in OpenCode with its real context window, and the list stays current as you add or remove models. You never type a model name into a settings file.

## What you need

- Astronomical, running on an Apple silicon Mac. The menu-bar item shows it is running.
- At least one chat model downloaded in Astronomical.
- OpenCode 2.3 or later.

## Install

```bash
opencode plugin add opencode-astronomical
```

To try a change from this repository instead, add this folder to `plugins` in `~/.config/opencode/opencode.json`, using the folder's full path. OpenCode loads the plugin's TypeScript sources directly, so there is nothing to build:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugins": ["/path/to/astronomical/integrations/opencode"],
}
```

## Use it

Start Astronomical, then OpenCode. Open the model picker with `/models`: your Astronomical models are listed under **Astronomical**. To use one from the command line, name it as `astronomical/<model id>`:

```bash
opencode run -m "astronomical/<model id>" "Say hello in one short sentence."
```

`/astronomical-status` in OpenCode's terminal shows whether Astronomical is reachable, its version, and how many chat models it serves.

To switch the plugin off without uninstalling it, add `"-astronomical"` after it in `plugins`.

`astronomical launch opencode` works with or without the plugin: both use the provider name `astronomical`, and OpenCode lists each model once.

## What it does

- Every 30 seconds the plugin asks the running app which models it serves. Models you add or remove in Astronomical appear or disappear without restarting OpenCode.
- Only chat models are listed. Picture, speech, embedding and decision models are left out, because OpenCode cannot talk to them.
- Each model's context window, reply limit, picture input, reasoning output and tool calls come from the app, not from a fixed list.
- If Astronomical is not running, its models are simply absent. Nothing old is shown in their place.
- The first reply from a model that is not yet loaded can take a while. The plugin allows up to ten minutes for it.

## Settings

Settings go in the plugin's entry in `plugins`. Both are optional.

```jsonc
{
  "plugins": [
    {
      "package": "opencode-astronomical",
      "options": { "channels": ["stable", "development"] },
    },
  ],
}
```

- **`channels`**: which Astronomical instances to list models from. The published app is the Stable instance, on port 6732, and that is the default. People building Astronomical from source also have a Development instance on port 6733; add `"development"` to list it too. Each instance gets its own provider, **Astronomical** and **Astronomical Development**, so a request always goes to the instance that serves the model.
- **`baseUrl`**: one address to use instead, such as a proxy, for example `"http://127.0.0.1:6732/v1"`. A missing `/v1` at the end is added. Its models are listed under **Astronomical**.

A misspelled setting stops the plugin, and OpenCode's log names the setting and what it accepts.

## Privacy

The plugin talks only to the Astronomical address it is set to, `127.0.0.1` unless you set `baseUrl`. It sends nothing anywhere else. It writes one file, `astronomical-plugin.log`, in OpenCode's state folder (`~/.local/state/opencode`, or `$XDG_STATE_HOME/opencode`): a record of what the plugin did that never grows past 1 MB.

## Develop

The plugin is a standalone npm package in this folder, with its own lockfile. It ships its TypeScript sources; OpenCode and Node 23.6 or later run them by removing the types, so the sources use only syntax that can be removed that way (`erasableSyntaxOnly`).

```bash
npm ci            # install the pinned toolchain
npm run typecheck # check the types
npm test          # run the tests
npm run verify    # formatting, types and tests
```

`solid-js` is a development dependency only because OpenCode's terminal plugin SDK loads it; inside OpenCode the host supplies its own copy. `effect` is a peer dependency, as OpenCode's plugin packaging notes ask: OpenCode resolves every plugin's `effect` imports, including those of the SDK it loads for this plugin, to its own single copy, so the package never bundles a second one.

The tests use stand-ins for OpenCode and for the app, so they need neither running. `tests/unit` covers single modules; `tests/component` runs the whole plugin against a stand-in OpenCode. Both read the app's model listing from `crates/contracts/rest-contract/fixtures/openai_model_list.json`, which a Rust test keeps identical to what the app serves.

## License

Apache License 2.0, like the rest of the Astronomical repository.
