# Notes for App Review

The "Notes" field of the version on the app record. Written so a reviewer can run the app without help. Keep it true to the build being reviewed: a new entitlement, a new process or a new kind of download needs its sentence here before submission.

The field cannot be saved until the review contact is filled in. The contact is the account holder's; ask them for it and write it nowhere else.

## The text

**What the app is.** Astronomical runs open AI models on the Mac itself. The person picks a model on the first screen; it downloads once (the first suggestion is about 5.6 GB) and then chat works offline. No account or sign-in exists, so there are no demo credentials.

**To review quickly.** Open the app, press Get started, then Download. When it finishes, type a question. To try a smaller model first, press "Choose a different model", which opens the Models page.

**Model files are data, not code.** Models are weight files in the safetensors format, which holds numbers only, read by the app's own bundled engine. Nothing downloaded is a script or a program of ours, and nothing downloaded can change the app's features. One experimental setting, off by default, lets the app write Core ML model packages on the Mac from weights it already has; macOS itself compiles and runs them through Apple's Core ML. They are made locally and never downloaded.

**Two processes.** The window is the chat app. A small menu-bar helper inside the bundle starts the engine, so chats keep working when the window is closed. Both are sandboxed and share one app group container, where settings, logs and downloaded models are kept. The engine and its worker are inside the bundle and inherit the helper's sandbox.

**Entitlements.**

- App Sandbox: on for both applications; the engine and worker inherit it.
- Outgoing network connections: to download the model files the person chooses, over HTTPS, from public model repositories.
- Incoming network connections: the engine listens on a loopback address only (127.0.0.1), which is how the chat window and local developer tools reach it. Nothing outside the Mac can connect.
- App group: the storage the two applications share.
- User-selected files, read and write: the person can attach a photo or a text file through the standard Open dialog, and save a generated picture or an exported conversation through the standard Save dialog. The app reaches only what they pick.

**Privacy.** Prompts, files, pictures and replies never leave the Mac. The app has no analytics and no advertising.

**Generated content.** Replies and pictures come from third-party open models chosen by the person.

## State on the record

Since the resubmission of 2026-11-14 (build 2313), the notes on the record match the text above, with two small differences in form: the record uses capital-letter headings in place of bold ones, and its "To review quickly" paragraph adds that the smallest chat model on the Models page is under 1 GB. Before editing, read the notes from the record and change that text, so nothing saved there is lost.

## Questions a reviewer may raise

- **Objectionable generated content.** The models are third-party and not filtered by the app. The age rating answers were given with that in mind. If Apple asks for a way to report or filter content, that is a product decision for the account holder, not a wording fix.
- **Downloading code.** The answer is the "Model files are data" paragraph. Keep it accurate when a new kind of model is added.
- **A slow connection.** The Models page lists smaller models than the first suggestion.
