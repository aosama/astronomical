# Thin Talk conversation canvas

This record explains how the right-hand conversation area of Thin Talk renders answers, why it is a WebKit surface rather than a stack of native views, and which boundaries keep it safe, offline, and fast. It implements the rendering side of the journey in issue 647.

## Decision

The conversation canvas is one `WKWebView` occupying the right-hand pane. The window chrome, sidebar, composer dock, model selector, and failure banners stay SwiftUI. The web surface renders the transcript; it owns no chat state.

Three facts drive that split.

- The application's floor is macOS 14 (issue 647, and `LSMinimumSystemVersion` in `apps/thin-talk/run-thin-talk.sh`). SwiftUI's `WebView` and `WebPage` require macOS 26, so the canvas must bridge `WKWebView` through `NSViewRepresentable`.
- Rich answers cannot be expressed natively. `AttributedString(markdown:)` renders no tables, no fenced code with syntax highlighting, and no embedded HTML; the Swift ecosystem's richer options are either in maintenance mode or force architectural compromises. `gonzalezreal/MarkdownUI`, the most established choice, is [explicitly in maintenance mode](https://github.com/gonzalezreal/swift-markdown-ui), with new work moved to Textual.
- One web view beats many. Combining multiple vertical web views inside a native scroll view is unreliable, and each instance costs a web content process; a single view also restores text selection across the whole transcript, which separate native text views cannot do. The [SuperSwiftMarkup rationale](https://github.com/SuperSwiftMarkup/SuperSwiftMarkdownPrototype) documents both limitations, and Craft's experience with WKWebView in a native shell is a practical account of the same trade.

The cost of this decision is honest: per-message affordances are rendered as HTML controls that report back to Swift, and reading order depends on WebKit's accessibility tree rather than SwiftUI's.

## Rendering pipeline

Markdown is parsed, sanitised, then patched. Every stage is pinned and offline.

| Stage | Implementation | Reason |
| --- | --- | --- |
| Parse | `marked` 15.0.7 (GitHub Flavored Markdown) | Maintained, GFM-complete, small, and no network use |
| Sanitise | `DOMPurify` 3.2.4 with an allowlist | Model output is untrusted markup; sanitising after parsing is the only correct order |
| Patch | `morphdom` 2.7.7 over one message subtree | Only the growing message is re-rendered, so finished code blocks, diagrams, and expanded details keep their DOM identity |
| Highlight | `highlight.js` 11.11.1 plus one theme | Code blocks are the most common rich answer a local model emits |

Streaming sends one message, not the transcript: the planner emits a message-level command while only the last message's text changes, so the work per delta is proportional to the growing block rather than the conversation. The naive alternative, re-assigning `innerHTML` from a re-parsed buffer, costs O(n²) parse work and throws away scroll position and selection; measured comparisons of that pattern are available in the [Generative DOM](https://github.com/generative-dom/generative-dom) and [Flowdown](https://github.com/Atomics-hub/flowdown) benchmarks.

## Content the canvas renders

- Assistant answers as full-width blocks with spaced headings, lists, tables, block quotes, links, and fenced code with syntax highlighting.
- User asks as compact tinted blocks, kept distinct without becoming large bubbles.
- Model reasoning in a collapsed `details` region, never merged into the visible answer.
- Attachments and app-generated images as a bounded strip with an "N images" badge, so a multi-image answer reads as one block.
- App-owned rich HTML cards, rendered from typed data the application already controls.
- Per-answer actions (copy, regenerate) that post to Swift, which keeps every behaviour out of the page.

## Security model

Rendered answers are attacker-influenced markup: a model can emit an image whose URL carries conversation content, an event-handler attribute, or a `javascript:` link. Four independent controls apply, and each one is tested.

1. **No raw model HTML reaches the DOM.** `DOMPurify` runs on the parsed output with an allowlist for tags and attributes, forbidding `script`, `style`, `iframe`, `form`, `input`, `button`, `svg`, `math`, `object`, `embed`, `link`, and `meta`.
2. **URLs are restricted by scheme.** Only `thintalk-asset:` and `data:image/` survive sanitisation, which removes both `javascript:` payloads and remote images, closing the zero-click markdown-image exfiltration channel.
3. **The page cannot reach the network.** A `Content-Security-Policy` sets `default-src 'none'` with `connect-src 'none'`, `font-src 'none'`, `frame-src 'none'`, and sources limited to the private scheme.
4. **The web view cannot navigate.** The navigation delegate cancels anything that is not the shell document, and hands `http`, `https`, and `mailto` links to the system browser instead.

Attachments are never addressed by file path. Swift registers a file and hands the page an opaque `thintalk-asset://attachment/<token>` URL, and the scheme handler serves only registered tokens, so the page cannot read arbitrary files even if a payload fabricates a URL.

## Trade-offs and deliberate omissions

- The transcript scrolls inside WebKit. This is what removes per-message height measurement, and with it the layout-feedback loops that dominate WKWebView-in-SwiftUI problems.
- Math, diagrams, and charts are not part of this slice. `KaTeX` needs inlined web fonts and Mermaid is a multi-megabyte bundle with a history of rendering advisories; both belong behind a later decision with their own evidence.
- Native Markdown was rejected, not deferred: it cannot render the tables, code, or embedded HTML that issue 647 requires.
## Math and diagram modalities (decision, 2026-09-19)

Question: the conversation canvas should carry the modalities open-webui's chat
surface carries — for the local-chat subset, maths and diagrams — without
weakening the "model output is untrusted" rule.

Evidence from open-webui's shipped pipeline (deepwiki 5.2/5.3 + source reading):
marked + `markedKatexExtension` ($, $$, \(, \[, \pu, \ce), highlight.js for code,
mermaid for `mermaid` fenced blocks with Vega alongside, Pyodide/Jupyter for code
execution, citations + artifacts as separate typed surfaces.

### What ships now

| Modality | Decision | Mechanism |
| --- | --- | --- |
| LaTeX maths ($…$, $$…$$, \(…\), \[…\]) | In | KaTeX 0.16.22, output grafted AFTER sanitisation |
| Fenced `mermaid` diagrams | In | mermaid 11.12.2, `securityLevel: "strict"`, renders once per finished message |
| Code highlighting | In (before) | highlight.js 11.11.1, unchanged |
| Code execution (Pyodide/Jupyter) | Deferred | Requires a sandboxed process surface; wholly different trust story than a rendering pane |
| Citations/RAG chips | Deferred | No retrieval backend yet; needs typed card data from Swift, not model text |
| Artifacts (model HTML/SVG panes) | Deferred | Rendering model-authored HTML whole would need a separate sandboxed pane + policy |
| Audio/Video/TTS/STT | Deferred | Merits its own decision (native speechSynthesis vs model audio) |
| Vega plots | Deferred | Mermaid covers the common diagram need at 1/4 the bundle cost |

### Security reasoning

- KaTeX output legitimately contains inline SVG and MathML, and the main
  sanitiser's allowlist exists to strip hostile SVG — widening it would admit
  model-authored SVG. Instead maths is extracted BEFORE markdown parsing
  (`extractMathSpans`), the surviving prose is sanitised as normal, and KaTeX
  output is grafted into the sanitised HTML per span. Tokens ("%%MATHn%%") are
  rebuilt per call, so a literal token a model emits can only splice its own
  maths, never cross a message boundary.
- Inline maths needs a TeX signature (commensurate with `$5-$10` staying money):
  single letter, Greek, or a command/brace/exponent character.
- Mermaid renders with `securityLevel: "strict"` (no clickable links) plus BOTH
  `htmlLabels: false` and `flowchart.htmlLabels: false`. Mermaid otherwise wraps
  node labels in `<foreignObject>`, which the SVG sanitiser forbids, leaving empty
  boxes. Measured in this shell: the flowchart-only key still produced
  `foreignObject`, while both keys together produced native SVG `<text>` labels
  that survive the strict list. Its SVG then passes a dedicated DOMPurify
  SVG-profile sanitisation that additionally forbids `foreignObject`, `script`,
  `iframe`, and `use`; the mermaid theme `<style>` block is allowed so labels keep
  their colour, and its selectors are scoped to that figure's own id. A diagram
  that fails to parse keeps its fenced source visible, labelled "(diagram could
  not render)".
- CSP now allows `font-src 'self' thintalk-asset:` — fonts are the only
  cross-file fetch KaTeX needs, and both remain inside the private scheme.

### Vendoring

`scripts/vendor-thin-talk-canvas-assets.sh` pins all of it by digest: katex.min.js,
katex.min.css, 20 woff2 fonts (served from `vendor/katex/fonts/` so the upstream
stylesheet's relative URLs stay untouched), mermaid.min.js, and every MIT or
BSD license file. `--verify-only` checks the committed bytes without network access.
The `marked-katex-extension` UMD bundle was evaluated and then removed: it grafts
KaTeX output during markdown parsing, which puts it in front of the sanitiser, so
the extraction is done in house instead.
