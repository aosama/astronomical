# Observatory chat render stack (shared source)

This directory holds the chat answer render stack the Observatory chat uses:

- markdown rendering (marked)
- inline math (KaTeX)
- diagrams (Mermaid)
- code highlighting (highlight.js)
- the model-output sanitiser (canvas-trust.js + DOMPurify)

**It is not a copy.** Every entry here is a symlink into Thin Talk's canvas shell,
which is the single source of truth:

```
../../../thin-talk/Sources/ThinTalkCanvas/Resources/web/
```

The modules are bridge-agnostic: they depend only on the `window.__thintalk*`
namespace, the vendor libraries, and the DOM, so the same files serve both the
Thin Talk native canvas and the Observatory console.

Keeping one source means a sanitizer or rendering fix lands in both places at
once instead of drifting apart. **Edit the files under Thin Talk, not the symlinks
here.** If you need a new render capability, add it to Thin Talk's web dir and it
appears in the Observatory automatically.
