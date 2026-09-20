/*
 * Conversation canvas transcript layer.
 *
 * Swift owns all chat state and pushes commands here; this file turns each
 * message into a DOM article, reports interactions back, and owns nothing else.
 * The rendering rules themselves live in canvas-render.js.
 *
 * Two properties matter for a streaming conversation:
 *
 * 1. Only the message that changed is patched, so a growing answer never
 *    re-renders the conversation and finished blocks keep their DOM identity.
 * 2. Failures are reported rather than swallowed, because a canvas that renders
 *    nothing while throwing nothing is indistinguishable from a broken model.
 */
(function () {
  "use strict";

  var RENDERER = window.__thintalkRenderer;
  var TRANSCRIPT = document.getElementById("transcript");
  var EMPTY_STATE = document.getElementById("empty-state");
  var EMPTY_STATE_NOTICE = document.getElementById("empty-state-notice");

  var INNER = document.createElement("div");
  INNER.className = "transcript__inner";
  TRANSCRIPT.appendChild(INNER);

  var articlesByMessageID = Object.create(null);
  var pendingMessages = [];
  var flushRequested = false;
  var followScroll = true;

  function decodeBase64Utf8(encoded) {
    var binary = window.atob(encoded);
    var bytes = new Uint8Array(binary.length);
    for (var index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new TextDecoder("utf-8").decode(bytes);
  }

  // MARK: - Scheduling

  /**
   * Batches pending messages into one flush.
   *
   * A frame is the natural unit of visible work, but a frame is not guaranteed: an
   * occluded window and an offscreen web view both pause `requestAnimationFrame`,
   * which would leave a streamed answer unrendered until the window came back. The
   * timer is the floor that keeps progress moving, and whichever wins performs the
   * single flush.
   */
  function scheduleFlush() {
    if (flushRequested) {
      return;
    }
    flushRequested = true;
    window.requestAnimationFrame(flushPending);
    window.setTimeout(flushPending, 16);
  }

  function flushPending() {
    if (!flushRequested) {
      return;
    }
    flushRequested = false;
    var batch = pendingMessages;
    pendingMessages = [];
    var wasFollowing = followScroll;
    batch.forEach(function (message) {
      try {
        applyMessage(message);
      } catch (error) {
        postError("render: " + (error && error.message ? error.message : String(error)));
        renderPlainTextFallback(message);
      }
    });
    if (wasFollowing) {
      scrollToBottom();
    }
  }

  function shouldFollowScroll() {
    return TRANSCRIPT.scrollHeight - TRANSCRIPT.scrollTop - TRANSCRIPT.clientHeight < 120;
  }

  function scrollToBottom() {
    TRANSCRIPT.scrollTop = TRANSCRIPT.scrollHeight;
  }

  // MARK: - Messages

  function applyMessage(message) {
    if (!message || !message.id) {
      return;
    }
    var article = articlesByMessageID[message.id];
    if (!article) {
      article = document.createElement("article");
      article.dataset.messageId = message.id;
      INNER.appendChild(article);
      articlesByMessageID[message.id] = article;
    }
    article.className = "message message--" + (message.role === "user" ? "user" : "assistant");
    var probe = document.createElement("div");
    probe.className = "message__probe";
    probe.innerHTML = RENDERER.messageHtml(message);
    RENDERER.labelCodeBlocks(probe);
    RENDERER.stampExpensiveKeys(probe);
    morphdom(article, probe, {
      childrenOnly: true,
      getNodeKey: RENDERER.nodeKey,
      onBeforeElUpdated: RENDERER.beforeElUpdated
    });
    RENDERER.highlightCodeBlocks(article);
    // Diagrams render once for finished messages; streamed ones show code until
    // the fence closes, exactly like code highlighting does.
    if (message.state === "complete" || message.state === "stopped") {
      RENDERER.renderDiagrams(article);
    }
  }

  /**
   * The one thing this layer can always do without the rendering pipeline: show
   * the answer as plain text so a reader is never left with a blank pane.
   */
  function renderPlainTextFallback(message) {
    if (!message || !message.id || articlesByMessageID[message.id]) {
      return;
    }
    var article = document.createElement("article");
    article.className = "message message--" + (message.role === "user" ? "user" : "assistant");
    article.dataset.messageId = message.id;
    var body = document.createElement("div");
    body.className = "message__answer markdown-body";
    body.textContent = message.markdown || "";
    article.appendChild(body);
    INNER.appendChild(article);
    articlesByMessageID[message.id] = article;
  }

  function renderSnapshot(command) {
    var incoming = {};
    (command.messages || []).forEach(function (entry) {
      incoming[entry.id] = true;
    });
    Object.keys(articlesByMessageID).forEach(function (id) {
      if (!incoming[id]) {
        INNER.removeChild(articlesByMessageID[id]);
        delete articlesByMessageID[id];
      }
    });
    (command.messages || []).forEach(function (entry) {
      pendingMessages.push(entry);
    });
    updateEmptyState(command);
    scheduleFlush();
  }

  function pushSingleMessage(message) {
    if (!message) {
      return;
    }
    followScroll = shouldFollowScroll();
    pendingMessages.push(message);
    scheduleFlush();
  }

  function updateEmptyState(command) {
    var hasMessages = (command.messages || []).length > 0;
    EMPTY_STATE.hidden = hasMessages;
    if (EMPTY_STATE_NOTICE) {
      EMPTY_STATE_NOTICE.textContent = hasMessages ? "" : command.notice || "";
    }
  }

  function applyAppearance(isDark) {
    document.documentElement.dataset.appearance = isDark ? "dark" : "light";
  }

  // MARK: - Bridge

  function postToSwift(payload) {
    if (!window.webkit || !window.webkit.messageHandlers || !window.webkit.messageHandlers.thintalk) {
      return;
    }
    window.webkit.messageHandlers.thintalk.postMessage(payload);
  }

  function postAction(action, messageId, detail) {
    postToSwift({
      kind: "action",
      action: action,
      messageId: messageId || "",
      detail: detail || ""
    });
  }

  function postError(detail) {
    postToSwift({ kind: "error", detail: String(detail) });
  }

  /**
   * Reports which renderer libraries and canvas modules arrived. A partially
   * loaded shell renders nothing while throwing nothing, so the check runs at
   * startup and the result goes back to Swift — where the missing name is the
   * difference between a broken build and a broken model.
   */
  function requireRendererLibraries() {
    var missing = [];
    if (typeof RENDERER !== "object") {
      missing.push("renderer");
    }
    if (typeof window.marked === "undefined") {
      missing.push("marked");
    }
    if (typeof window.DOMPurify === "undefined") {
      missing.push("dompurify");
    }
    if (typeof window.morphdom === "undefined") {
      missing.push("morphdom");
    }
    if (typeof window.hljs === "undefined") {
      missing.push("highlight.js");
    }
    [
      ["canvas-trust", window.__thintalkTrust],
      ["canvas-math", window.__thintalkMath],
      ["canvas-diagrams", window.__thintalkDiagrams],
      ["canvas-content-key", window.__thintalkContentKey]
    ].forEach(function (moduleEntry) {
      if (typeof moduleEntry[1] === "undefined") {
        missing.push(moduleEntry[0]);
      }
    });
    if (missing.length) {
      postError("missing libraries: " + missing.join(", "));
      return false;
    }
    return true;
  }

  window.addEventListener("error", function (event) {
    postError("uncaught: " + (event.message || "unknown error"));
  });

  INNER.addEventListener("click", function (event) {
    var actionButton = event.target.closest("button[data-action]");
    if (actionButton) {
      var article = actionButton.closest("[data-message-id]");
      postAction(actionButton.dataset.action, article ? article.dataset.messageId : "");
      return;
    }
    var anchor = event.target.closest("a[href]");
    if (anchor) {
      event.preventDefault();
      postAction("openExternal", "", anchor.getAttribute("href"));
    }
  });

  TRANSCRIPT.addEventListener("scroll", function () {
    followScroll = shouldFollowScroll();
  });

  // MARK: - Entry point

  window.__thintalk = {
    receive: function (encoded) {
      var command;
      try {
        command = JSON.parse(decodeBase64Utf8(encoded));
      } catch (error) {
        return false;
      }
      if (!command || command.v !== 1) {
        return false;
      }
      switch (command.kind) {
        case "snapshot":
          renderSnapshot(command);
          break;
        case "message":
          pushSingleMessage(command.message);
          break;
        case "appearance":
          applyAppearance(command.dark === true);
          break;
        default:
          return false;
      }
      return true;
    },
    messageCount: function () {
      return Object.keys(articlesByMessageID).length;
    },
    renderedText: function (messageId) {
      var article = articlesByMessageID[messageId];
      return article ? article.textContent : "";
    },
    diagnostics: function () {
      return {
        readyState: document.readyState,
        entryPoint: typeof window.__thintalk,
        libraries: {
          renderer: typeof window.__thintalkRenderer,
          marked: typeof window.marked,
          dompurify: typeof window.DOMPurify,
          morphdom: typeof window.morphdom,
          highlight: typeof window.hljs,
          katex: typeof window.katex,
          mermaid: typeof window.mermaid
        },
        modules: {
          trust: typeof window.__thintalkTrust,
          math: typeof window.__thintalkMath,
          diagrams: typeof window.__thintalkDiagrams,
          contentKey: typeof window.__thintalkContentKey
        },
        scripts: Array.prototype.map.call(document.scripts, function (script) {
          return script.getAttribute("src") || "";
        }),
        messages: Object.keys(articlesByMessageID).length,
        transcriptChildren: INNER.children.length
      };
    }
  };

  requireRendererLibraries();
  postToSwift({ kind: "ready" });
})();