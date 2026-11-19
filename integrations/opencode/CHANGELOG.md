# Changelog

Each version lists the OpenCode releases it was checked against. Versions follow OpenCode's plugin interface, not the Astronomical app's releases.

## 0.1.0

First published version. Checked against OpenCode 2.3.0 with `@opencode/plugin` 2.3.2.

- Lists the chat models of a running Astronomical in OpenCode's model picker, with each model's context window, reply limit, picture input, reasoning output and tool calls, and keeps the list current every 30 seconds.
- Lists the Stable instance by default. The `channels` option adds the Development instance, and `baseUrl` points the plugin at one address instead.
- `/astronomical-status` shows which instances answer, their versions and their chat model counts.
- Models are registered under the provider name `astronomical`, the same name `astronomical launch opencode` uses.
- Talks only to the configured Astronomical address and writes one log file in OpenCode's state folder.
