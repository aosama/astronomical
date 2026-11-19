# Astronomical for OpenCode

This plugin puts the models of [Astronomical](https://getastronomical.com/) in OpenCode's model picker. Astronomical runs AI models privately on your Mac; with the plugin installed, every chat model the app serves shows up in OpenCode with its real context window, and the list stays current as you add or remove models. You never type a model name into a settings file.

## What you need

- Astronomical, running on an Apple silicon Mac. The menu-bar item shows it is running.
- At least one chat model downloaded in Astronomical.
- OpenCode 2.

## Install

The plugin is not on npm yet. Until it is, build it from this repository and point OpenCode at the folder:

```bash
cd integrations/opencode
npm ci
npm run build
```

Then add the folder to `plugins` in `~/.config/opencode/opencode.json`, using the folder's full path:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugins": ["/path/to/astronomical/integrations/opencode"],
}
```

Once it is published, one command will replace both steps:

```bash
opencode plugin add opencode-astronomical
```

## Use it

Start Astronomical, then OpenCode. Open the model picker with `/models`: your Astronomical models are listed under **Astronomical**. To use one from the command line, name it as `astronomical/<model id>`:

```bash
opencode run -m "astronomical/<model id>" "Say hello in one short sentence."
```

Two commands are added to OpenCode's terminal:

- `/astronomical-status` shows whether Astronomical is reachable, its version, and how many chat models it serves.
- `/astronomical-toggle` hides Astronomical's models from the picker, or shows them again.

## What it does

- Every 30 seconds the plugin asks the running app which models it serves. Models you add or remove in Astronomical appear or disappear without restarting OpenCode.
- Only chat models are listed. Picture, speech, embedding and decision models are left out, because OpenCode cannot talk to them.
- Each model's context window, reply limit, picture input, reasoning output and tool calls come from the app, not from a fixed list.
- If Astronomical is not running, its models are simply absent. Nothing old is shown in their place.
- The first reply from a model that is not yet loaded can take a while. The plugin allows up to ten minutes for it.

## Stable and Development

Astronomical's published app is the Stable instance, on port 6732. People building Astronomical from source also have a Development instance on port 6733. When both are running, the plugin lists them separately, as **Astronomical** and **Astronomical Development**, so a request always goes to the instance that serves the model.

Two settings change which instance the plugin uses. They are read from the environment of the OpenCode process:

- `ASTRONOMICAL_INSTANCE=stable` or `dev` uses only that instance.
- `ASTRONOMICAL_BASE_URL` points the plugin at one specific address instead, such as a proxy. A missing `/v1` at the end is added.

## Privacy

The plugin talks only to Astronomical on your own Mac, at `127.0.0.1`. It sends nothing anywhere else. It writes two small files in OpenCode's state folder (`~/.local/state/opencode`, or `$XDG_STATE_HOME/opencode`): `astronomical-plugin-state.json`, which remembers the toggle, and `astronomical-plugin.log`, a record of what the plugin did that never grows past 1 MB.

## Develop

The plugin is a standalone npm package in this folder, with its own lockfile.

```bash
npm ci          # install the pinned toolchain
npm run build   # compile src/ into dist/
npm test        # build, then run the tests
npm run verify  # formatting check, build and tests
```

The tests use stand-ins for OpenCode and for the app, so they need neither running. `tests/unit` covers single modules; `tests/component` runs the whole plugin against a stand-in OpenCode. Both read the app's model listing from `crates/rest-contract/fixtures/openai_model_list.json`, which a Rust test keeps identical to what the app serves.

## License

Apache License 2.0, like the rest of the Astronomical repository.
