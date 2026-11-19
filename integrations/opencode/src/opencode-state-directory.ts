// Where OpenCode keeps per-user state. The plugin's log and toggle file live
// beside OpenCode's own files there, so removing OpenCode's state removes the
// plugin's too.

import { homedir } from "node:os";
import { join } from "node:path";

/** OpenCode's state folder: $XDG_STATE_HOME/opencode, else ~/.local/state/opencode. */
export function resolveOpenCodeStateDirectory(
  environment: NodeJS.ProcessEnv = process.env,
  homeDirectory: string = homedir(),
): string {
  const stateHome: string =
    environment.XDG_STATE_HOME?.trim() ||
    join(homeDirectory, ".local", "state");
  return join(stateHome, "opencode");
}
