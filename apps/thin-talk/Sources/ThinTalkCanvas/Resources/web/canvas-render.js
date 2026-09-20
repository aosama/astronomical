/*
 * Answer composition for the conversation canvas.
 *
 * This file answers one question: given a message payload, what HTML may the
 * transcript insert? It composes that HTML from four collaborators — the trust
 * boundary (canvas-trust.js), the maths pipeline (canvas-math.js), the diagram
 * pipeline (canvas-diagrams.js), and the content keys that let finished blocks
 * survive diffing (canvas-content-key.js) — and owns the expensive-block rules for
 * code. It never parses conversation state or talks to the Swift bridge.
 */
(function () {
  "use strict";

  var trust = window.__thintalkTrust;
  var math = window.__thintalkMath;
  var contentKey = window.__thintalkContentKey;

  var VISIBLE_STATE_LABELS = {
    streaming: "generating",
    stopped: "stopped",
    failed: "failed"
  };

  marked.setOptions({ gfm: true, breaks: false });

  // MARK: - Message HTML

  function stateLabel(state) {
    if (state === "streaming") {
      return 'generating<span class="message__streaming-caret"></span>';
    }
    return VISIBLE_STATE_LABELS[state] || "";
  }

  function attachmentStripHtml(attachments) {
    if (!attachments.length) {
      return "";
    }
    var items = attachments
      .map(function (attachment) {
        return (
          '<span class="attachment-strip__item" title="' +
          trust.escapeHtml(attachment.label || "") +
          '">' +
          '<img alt="' +
          trust.escapeHtml(attachment.label || "attachment") +
          '" src="' +
          trust.escapeHtml(attachment.assetURL || "") +
          '">' +
          "</span>"
        );
      })
      .join("");
    var badge =
      attachments.length > 1
        ? '<span class="attachment-strip__badge" data-testid="attachment-badge">' +
          attachments.length +
          " images</span>"
        : "";
    return '<div class="attachment-strip" data-testid="attachment-strip">' + items + badge + "</div>";
  }

  /**
   * Rich cards are app-authored HTML produced by Swift from typed data. They are
   * sanitised with the same allowlist as model output rather than trusted, because
   * one rendering path with one rule is easier to keep correct than two.
   */
  function cardHtml(card) {
    return '<section class="rich-card">' + trust.sanitizeHtml(card.html || "") + "</section>";
  }

  function actionsHtml(message) {
    if (message.state !== "complete" && message.state !== "stopped") {
      return "";
    }
    // Regenerate belongs to what the model produced; offering it under the
    // user's own words made every turn look like another compose surface.
    var actions = ['<button class="message__action" data-testid="action-copy" data-action="copy">Copy</button>'];
    if (message.role !== "user") {
      actions.push('<button class="message__action" data-testid="action-regenerate" data-action="regenerate">Regenerate</button>');
    }
    return '<div class="message__actions">' + actions.join("") + "</div>";
  }

  /**
   * One rule set for streaming and snapshots alike: maths is carved out before
   * marked sees the text, the surviving prose is sanitised, and the rendered maths
   * is grafted back into that sanitised result.
   */
  function renderMarkdown(markdown) {
    var source = typeof markdown === "string" ? markdown : "";
    if (!source) {
      return "";
    }
    var collectedMathSpans = [];
    try {
      var tokenedSource = math.tokenize(source, collectedMathSpans);
      var sanitisedHtml = trust.sanitizeHtml(marked.parse(tokenedSource));
      return math.graft(sanitisedHtml, collectedMathSpans);
    } catch (error) {
      // A partial answer can defeat the parser mid-stream. Showing the raw text
      // keeps the conversation readable instead of dropping the message body.
      return trust.sanitizeHtml("<p>" + trust.escapeHtml(source) + "</p>");
    }
  }

  function messageHtml(message) {
    var reasoning = message.reasoning
      ? '<details class="reasoning"><summary>Reasoning</summary><div class="reasoning__body">' +
        trust.escapeHtml(message.reasoning) +
        "</div></details>"
      : "";
    var answerBody = renderMarkdown(message.markdown);
    var attachments = attachmentStripHtml(message.attachments || []);
    var cards = (message.cards || []).map(cardHtml).join("");
    if (message.role === "user") {
      // The sender label lives inside the user bubble so stacked user turns
      // read as sent messages instead of extra compose fields.
      return (
        '<div class="message__answer message__answer--user">' +
        '<span class="message__role">You</span>' +
        '<div class="markdown-body">' + answerBody + '</div>' +
        attachments +
        '</div>' +
        actionsHtml(message)
      );
    }
    return (
      '<div class="message__meta">' +
      '<span class="message__role">Assistant</span>' +
      '<span class="message__state">' + stateLabel(message.state) + '</span>' +
      '</div>' +
      '<div class="message__answer">' +
      reasoning +
      '<div class="markdown-body">' + answerBody + '</div>' +
      cards +
      attachments +
      '</div>' +
      actionsHtml(message)
    );
  }

  // MARK: - Expensive blocks

  function highlightCodeBlocks(root) {
    root.querySelectorAll("pre > code").forEach(function (codeBlock) {
      if (codeBlock.dataset.highlighted === "true" || codeBlock.dataset.highlighted === "skipped") {
        return;
      }
      var language = (codeBlock.className.match(/language-([\w+#-]+)/) || [])[1];
      if (!language || !window.hljs.getLanguage(language)) {
        codeBlock.dataset.highlighted = "skipped";
        return;
      }
      codeBlock.innerHTML = window.hljs.highlight(codeBlock.textContent, { language: language }).value;
      codeBlock.dataset.highlighted = "true";
      codeBlock.classList.add("hljs");
    });
  }

  function labelCodeBlocks(root) {
    root.querySelectorAll("pre > code").forEach(function (codeBlock) {
      var pre = codeBlock.parentElement;
      if (
        !pre ||
        !pre.parentElement ||
        pre.parentElement.classList.contains("code-block") ||
        pre.parentElement.dataset.mdKey
      ) {
        return;
      }
      var language = (codeBlock.className.match(/language-([\w+#-]+)/) || [])[1];
      if (!language) {
        return;
      }
      var codeBlockWrapper = document.createElement("div");
      codeBlockWrapper.className = "code-block";
      var languageLabel = document.createElement("span");
      languageLabel.className = "code-block__label";
      languageLabel.textContent = language;
      pre.parentElement.insertBefore(codeBlockWrapper, pre);
      codeBlockWrapper.appendChild(pre);
      codeBlockWrapper.appendChild(languageLabel);
    });
  }

  /**
   * Expensive blocks are stamped with a stable key derived from their content, so
   * the transcript's diffing keeps the existing node when nothing about that block
   * changed. That is what lets a streamed answer stay rendered: highlighting,
   * expanded details, and installed diagrams are not recomputed while unrelated
   * prose updates around them.
   */
  function stampExpensiveKeys(root) {
    root.querySelectorAll("div.code-block").forEach(function (codeBlockWrapper) {
      var codeBlock = codeBlockWrapper.querySelector("code");
      if (codeBlock && codeBlock.dataset.mdKey) {
        codeBlockWrapper.dataset.mdKey = codeBlock.dataset.mdKey;
        return;
      }
      codeBlockWrapper.dataset.mdKey =
        "code-" + contentKey.hash(codeBlock ? codeBlock.textContent : "");
    });
  }

  function nodeKey(node) {
    return (node.dataset && node.dataset.mdKey) || node.id || "";
  }

  /**
   * Reuses the existing subtree when a keyed block is textually identical, which
   * preserves its rendered form instead of re-highlighting or collapsing it.
   */
  function beforeElUpdated(fromElement, toElement) {
    var fromKey = nodeKey(fromElement);
    var toKey = nodeKey(toElement);
    if (fromKey && fromKey === toKey && fromElement.textContent === toElement.textContent) {
      return false;
    }
    return true;
  }

  /** Diagrams render once per finished message; the pipeline owns the policy. */
  function renderDiagrams(root) {
    window.__thintalkDiagrams.render(root);
  }

  window.__thintalkRenderer = {
    messageHtml: messageHtml,
    labelCodeBlocks: labelCodeBlocks,
    stampExpensiveKeys: stampExpensiveKeys,
    highlightCodeBlocks: highlightCodeBlocks,
    renderDiagrams: renderDiagrams,
    nodeKey: nodeKey,
    beforeElUpdated: beforeElUpdated
  };
})();