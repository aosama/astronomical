# Observatory chat render stack

This directory holds the chat answer render stack the Observatory chat uses:

- markdown rendering (marked)
- inline math (KaTeX)
- diagrams (Mermaid)
- code highlighting (highlight.js)
- the model-output sanitiser (canvas-trust.js + DOMPurify)

The five `canvas-*.js` modules are **owned by the console**. They used to be
symlinks into Thin Talk's canvas shell, but Thin Talk's client now lives in the
TypeScript sources under `apps/thin-talk/web/src/` and ships as the generated
`bundle.js`; the plain-JS modules here are no longer Thin Talk's living source,
so the console keeps its own copies and evolves them independently. They stay
bridge-agnostic: they depend only on the `window.__thintalk*` namespace, the
vendor libraries, and the DOM.

`vendor/` remains a symlink into
`../../../thin-talk/Sources/ThinTalkCanvas/Resources/web/vendor` — the vendored
libraries are byte-identical third-party releases consumed by both surfaces, so
they keep one shared source. To bump a vendored library, update it under Thin
Talk (see `scripts/vendor-thin-talk-canvas-assets.sh`); the console picks it up
through this link.
