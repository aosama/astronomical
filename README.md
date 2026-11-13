# Astronomical

<a href="https://github.com/aosama/astronomical/releases/download/v0.2.103/Astronomical-0.2.103-macOS-arm64.dmg"><img src="site/assets/download-for-mac.svg" alt="Download for Mac" width="170" height="34" style="vertical-align:middle"></a> <a href="https://discord.gg/dc4E6r4WD"><img src="site/assets/discord.svg" alt="Discord" width="162" height="34" style="vertical-align:middle"></a>

**Run large AI models privately on your Mac, including models bigger than the memory you can spare.**

Astronomical is a free app for Apple silicon Macs. You tell it how much memory it may use, pick a model, and start chatting. When a model is too big for that memory, Astronomical keeps part of it on your SSD and reads it as needed, so the rest of your Mac stays usable. Nothing you type leaves your Mac.

[![Astronomical running a Qwen3.6 35B mixture-of-experts model with an 11 GB model RAM ceiling and RAM plus SSD streaming](https://aosama.github.io/astronomical/assets/astronomical-ram-ssd-streaming.jpeg)](https://aosama.github.io/astronomical/assets/astronomical-ram-ssd-streaming.mp4)

*A 21.6 GB model running with an 11 GB memory limit. Click the picture for a 25-second demo. This is a captured run; speed varies by model, prompt length, storage and Mac.*

## What you need

- A Mac with Apple silicon (any M-series chip).
- macOS 14 (Sonoma) or later.
- Free disk space for models. They range from under 1 GB to about 38 GB each.

## Get started

1. **Install the app.** Download the disk image with the button above or from the [latest release](https://github.com/aosama/astronomical/releases/latest), open it, and drag Astronomical into Applications. The app is signed and notarized by Apple.
2. **Open Astronomical.** A chat window opens with a welcome, and an Astronomical item appears in the menu bar. The menu-bar item keeps models available while you work, even after you close the chat window.
3. **Download a model.** Click **Get started**. Astronomical suggests a model for your Mac: click **Download**, or **Choose a different model** to pick from the full list. It checks the download before using it.
4. **Chat.** When the download finishes, click **Start chatting** and type your first message. The model loads when you send it.
5. **Say how much memory it may use (optional).** In the **Astronomical** menu at the top of the screen, choose **Open Observatory**. It opens the app's control page in your browser. Go to **Settings** and set **Maximum model RAM**. On a 32 GB Mac, for example, 16 GB leaves room for macOS and your other apps.

To add more models, open **Models** in Observatory and click **Get** beside one. To remove a model you downloaded there, use **Remove** on its line; it goes to the Trash.

## Use it with your coding tools

Astronomical can stand in for a cloud model in tools that speak the OpenAI or Anthropic API.

First, add the `astronomical` command to Terminal: click the menu-bar item, click the **⋯** button, and choose **Install “astronomical” command in terminal…**. Then start a coding agent that uses the model on your Mac:

    astronomical launch claude
    astronomical launch codex
    astronomical launch opencode

For any other tool, point it at `http://127.0.0.1:6732/`. The [local API page](https://aosama.github.io/astronomical/local-api.html) has the settings for each client.

## Use it from Terminal

    astronomical respond "Explain what a mutex is in two sentences"
    astronomical models list
    astronomical models download MODEL_ID
    astronomical image "A lighthouse at dusk"
    astronomical status

Run `astronomical --help` for the full list.

## Good to know

- **Private by design.** Prompts, replies and models stay on your Mac. The app only accepts connections from your own Mac.
- **Your model's quality is not reduced.** Astronomical never lowers a model's precision to save memory.
- **You can see what is happening.** The menu-bar item and Observatory show memory in use, progress and speed.
- **Models are downloaded, not bundled.** Each model has its own license, which must allow your intended use.
- **It is experimental.** One person, one conversation at a time. Supported model families today are Qwen3.5, Qwen3.6, Laguna, FLUX.2 Klein for images, and ModernBERT for embeddings.

## Learn more

- [How Astronomical works](docs/how-it-works.md): memory, speed, precision and current scope.
- [Configuration](docs/configuration.md): every setting, for people who want to edit the file by hand.
- [Neural Engine prefill](docs/neural-engine-prefill.md): an experimental speed-up for some models.
- [Building and developing](docs/development.md): build from source and run the tests.
- [Product story and engineering reports](https://aosama.github.io/astronomical/) on the website.
- [Discord](https://discord.gg/dc4E6r4WD) for questions and help, and the [support page](https://aosama.github.io/astronomical/support.html).

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes and [SECURITY.md](SECURITY.md) before reporting a vulnerability.

## License

Astronomical is licensed under [Apache License 2.0](LICENSE). See [THIRD_PARTY_NOTICES](third-party/THIRD_PARTY_NOTICES) and [RUST_DEPENDENCY_NOTICES](third-party/RUST_DEPENDENCY_NOTICES) for complete distribution terms.
