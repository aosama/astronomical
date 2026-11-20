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

A test cannot notice a new ability the skill never mentions. Whenever a
change adds, renames or removes a command or flag, changes how a coding tool or
editor connects, adds or withdraws a kind of model, or changes a setting or
what a person should do when something fails, update the skill in the same
change.
