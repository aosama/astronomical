# instructions for the Scripts Folder

- There should not be scripts that test other scripts in this folder.
- Do not create or retain scripts that run tests through Cargo commands; use the Cargo commands themselves.
- Proactively record the needed cargo command in <repo-root>/cargo-commands-for-testing.md and delete existing test-command wrappers when their commands can run directly.
