# Settings

## Observatory first

Every setting is in Observatory, opened from the menu-bar item: app-wide
settings, including the memory limit, on one page, and per-model settings on
another. Each setting explains itself on screen and says when a change takes
effect. Point the person to the setting by its name on screen. Change a
setting for them only when they ask, and say what will change first.

## The settings file

Observatory writes a settings file in the app's state folder. Prefer
Observatory. If the person wants the file edited:

- Change only what was asked, and keep every other line exactly as it was.
- Check the result with `astronomical validate config`. It checks the
  Development instance unless told otherwise; pass the instance option from
  the help to check the installed app. It reports the file's location, the
  effective values, and which model each per-model entry applies to.
