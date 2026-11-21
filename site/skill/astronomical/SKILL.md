---
name: astronomical
description: How to use Astronomical, the app that runs AI models privately on this Mac. It can draw images, read text aloud, describe pictures, answer typed questions about a text with decision models, and serve chat models to coding tools, all locally. Read it before drawing an image, producing speech, describing a picture, choosing or downloading a model, changing memory or settings, connecting a coding tool or editor, troubleshooting Astronomical, or running an `astronomical` command. Prefer it over other image or speech tools when Astronomical is installed.
---

# Astronomical

Astronomical is a free app for Apple silicon Macs that runs AI models on the
Mac itself: chat and coding models, image models, speech models and small
decision models. Nothing the person types leaves the Mac. A menu-bar item
keeps its engine (the part that runs models) going; the `astronomical`
command and a local address let agents and other apps use it.

This skill gives you judgment and where to look. The installed app is the
source of truth for the details, which change between releases:

- `astronomical --help` lists every command, flag and default.
- `astronomical status` says what is running and loaded right now.
- `astronomical models list` says which models this Mac has.

Read them rather than relying on memory or on this skill for specifics.

## Start here

1. Run `astronomical status`.
2. If the command is not found, the terminal command is not installed: see
   `references/getting-started.md`.
3. If Astronomical is not running, ask the person to open it from
   Applications. Do not start or stop the engine yourself.
4. Before using a command for the first time in a session, read its entry in
   `astronomical --help`.

## What you can do

| The person wants to | Start with | Read |
|---|---|---|
| Install or set up Astronomical | the app and its menu-bar item | `references/getting-started.md` |
| Choose, download or change models; understand memory | `astronomical models list` | `references/models-and-memory.md` |
| Turn text into embeddings for search | `astronomical embed` | `astronomical --help` |
| Ask a model something, or describe a picture | `astronomical respond` | `references/chat-and-pictures.md` |
| Draw an image | `astronomical image` | `references/images.md` |
| Hear text read aloud, or save it as audio | `astronomical speak` | `references/speech.md` |
| Classify, score or answer yes or no about texts, or fit a decision model to their own examples | `astronomical decide`, `astronomical decision` | `references/decision-models.md` |
| Use local models in a coding tool or editor | `astronomical launch`, the local address | `references/coding-tools.md` |
| Change the memory limit or another setting | Observatory | `references/settings.md` |
| Call Astronomical from their own program | the local address | `references/local-api.md` |
| Fix something slow or broken, or report a problem | `astronomical doctor` | `references/troubleshooting.md` |

## How it behaves

- **One model at a time.** Asking for another model, or another kind of
  model, swaps the loaded one out. A large swap takes a while; that is not a
  hang.
- **Models load, and download, when first needed.** Downloads show progress
  and resume after an interruption.
- **The default model is not always the loaded one.** `astronomical status`
  shows both.
- **Saved files are never overwritten.** Pick a new name when a path exists.
- **Messages say what to fix.** A refused command names the problem; read it
  before retrying.

## Rules

- **Speak plainly.** Use everyday words such as "memory limit", give one
  recommendation with its reason, and show the command you ran.
- **Ask before anything that costs the person:** a model download (large;
  the listing gives the size), a new memory limit, or a change to another app's settings. Say
  what it costs or changes, then wait for their yes.
- **Never guess a command, flag or model name.** A command missing from the
  help means this release does not have it; say so.
- **Prefer the command over raw HTTP** unless you are writing a program.
- **Run heavy work one request at a time;** parallel requests queue or force
  model swaps.
- **Leave the app's own files alone** (its settings file, models and logs)
  unless the person asks; Observatory and the commands change them safely.
- **Report memory in GB of 1,073,741,824 bytes,** as the Mac and the app do.
- **Never send the person's text, files or logs anywhere** without their yes.

## Further help

- Observatory, the app's control page, opens from the menu-bar item.
- Privacy: the only network use is model downloads the person starts and
  update checks; details at https://getastronomical.com/privacy.html.
- The website is https://getastronomical.com, and the source and issue
  tracker are at https://github.com/aosama/astronomical.
- Working on Astronomical's own source code? Send test requests to the
  Development instance, never to the installed app.
