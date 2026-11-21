# The Astronomical skill

`astronomical/` is the skill shipped with the app: `SKILL.md`, read whenever an
agent uses the skill, and `references/`, read only when a task needs one of
them. It follows the Agent Skills format, so the folder's name and the skill's
`name` are both `astronomical`.

The engine tells every model it serves where the skill is, so agents in any
harness can read it without the person installing anything. The plan, the
evidence behind it and the remaining work are in issue #2397.

## Writing it

- **Two readers:** an agent helping a developer, and an agent helping a person
  who is not technical. Plain words, exact commands.
- **Judgment and pointers, not a manual.** The skill does not replace
  `astronomical --help` or the running app: agents discover commands, flags,
  defaults, model names and sizes there. The skill says which command fits a
  task, what to ask the person first, and what to watch out for.
- **Nothing that changes from release to release:** no option lists, default
  values, timings, sizes, menu paths that move, or field-by-field formats.
  Name the command and send the agent to the help.
- **Not the other extreme either.** Concepts that stay true (one model at a
  time, the memory limit and SSD streaming, the training levels, the examples
  file) belong here, because the help cannot teach them.
- **The description names the abilities people ask for** (drawing, reading
  aloud, describing pictures, decisions, coding tools). An evidence test showed
  agents skip a skill whose description names only the product.
- **The index stays under 150 lines;** detail goes in a reference file that the
  index's table points to.
- **Everything here ships to people.** No developer paths, names, or anything
  true only on one machine.

## Keeping it true

`cargo test -p astronomical-cli --test integration_tests astronomical_skill`
checks the skill against the CLI. It fails when the skill names a command,
subcommand or flag the help does not have, when a reference file is missing
from the index or the index points to a missing one, when the index grows past
its line budget, and when the description stops naming the abilities.

The engine compiles the skill in and writes it to the app's state folder at
every start (`apps/daemon/src/astronomical_skill.rs`), unless a newer skill
downloaded from the website is already there. A reference file added
or removed here must also be added to or removed from that file's
`EMBEDDED_SKILL_FILES`; `cargo test -p astronomical-daemon --lib
astronomical_skill` fails until the two match.

## Publishing an update

Installed apps look for a newer skill on the website about once a day, so the
skill can improve without an app release.

1. Edit `skill/astronomical`.
2. Raise `skill_version` in `skill/release.json`. Without a higher number no
   app takes the update.
3. Set `minimum_app_version` there to the oldest app version the skill suits.
   The site is published from `main`, so a skill that mentions something not
   released yet must name the upcoming version, or released apps would be
   told about a command they do not have.
4. Run `cargo xtask skill-site`, which copies the skill to `site/skill` and
   writes its manifest. `cargo xtask skill-site --check` fails in CI when the
   site's copy is out of date.

A test cannot notice a new ability the skill never mentions. Whenever a
change adds, renames or removes a command or flag, changes how a coding tool or
editor connects, adds or withdraws a kind of model, or changes a setting or
what a person should do when something fails, update the skill in the same
change.
