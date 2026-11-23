# Troubleshooting and reporting a problem

Start with what the app says about itself: `astronomical status` and
`astronomical doctor`.

## The command is not found

The terminal command is not installed; see `references/getting-started.md`.

## Astronomical is not running

Ask the person to open Astronomical from Applications, or check that its
menu-bar item is showing.

## It is slow

Work through these in order and tell the person what you find:

1. **The model was unloaded while idle,** and the first request after a
   pause loads it again. A setting in Observatory controls how long a model
   stays loaded.
2. **A model is loading or swapping.** `astronomical status` shows it.
   Switching between a chat model and an image model swaps each time.
3. **A long prompt is being read.** The menu-bar item and Observatory show
   its progress. A later prompt that starts the same way is quicker.
4. **The model is larger than the memory limit,** so part of it is read from
   the SSD. Options for the person: a smaller model, or a higher limit if the
   Mac has room. Let them choose.
5. **Something else is busy.** Other apps using the graphics chip or most of
   the memory slow every model.

Do not restart or stop the engine to fix slowness; it throws away the loaded
model and its progress. If nothing answers at all, the menu-bar item has an
option to restart the engine; that is the person's action, not yours.

## A model copied in by hand does not show up

Astronomical notices a model folder added to, moved within or removed from
its models folder within a few seconds, with no restart. If one still does
not appear:

- Its weight files must be real files. A shortcut (symlink) to a weight file
  is refused; a copy, or a symlink to the whole model folder, works.
- `astronomical models list` lists only models this release can run. A folder
  of another kind, or an incomplete one, is skipped.

## A model fails to load

Usually there is not enough free memory at that moment. Something else may
hold a large model: another app, or a second copy of Astronomical such as a
build from source. Ask the person to close or unload it there
(`astronomical unload` frees Astronomical's), then try again. A lower memory
limit also lets the model load, reading part of it from the SSD.

## A download stalls or fails

Run the same download again; it resumes. Check free disk space. A model the
app cannot verify is refused rather than used.

## Logs

The app keeps logs in the `logs` folder of its state folder, which
`astronomical validate config` reports. They hold technical detail, not
what the person typed or what the model wrote. Read them only to diagnose,
and never send them anywhere without the person's yes.

## Reporting a problem

1. Collect, without guessing: `astronomical --version`, `astronomical status`,
   `astronomical doctor`, the macOS version (`sw_vers`), and the Mac's chip and
   memory.
2. Ask only for what is missing: what they did, what happened, what they
   expected, and whether it happens every time.
3. Remove anything personal: names, paths with their user name, the contents
   of prompts or documents.
4. Draft the report and show it to the person.
5. Send it only after they say yes, through
   https://github.com/aosama/astronomical/issues or the support page,
   https://getastronomical.com/support.html. If you cannot file it, give them
   the finished text.
