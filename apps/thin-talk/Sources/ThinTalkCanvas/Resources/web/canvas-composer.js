/*
 * Conversation canvas composer layer.
 *
 * The composer, the thinking-effort pill, and the failure banner are app-owned
 * chrome that happens to live on the same web surface as the transcript, so
 * every control shares one font, one zoom, and one design language with the
 * answers. This module renders the state Swift pushes and reports every intent
 * back; it owns no chat behaviour and holds no chat state beyond the field's
 * in-progress text, the same way a native field owns its caret.
 */
(function () {
  "use strict";

  var FIELD = document.getElementById("composer-input");
  var SEND = document.querySelector('[data-testid="composer-send"]');
  var STOP = document.querySelector('[data-testid="composer-stop"]');
  var EFFORT_PILL = document.querySelector('[data-testid="composer-effort"]');
  var EFFORT_LABEL = document.querySelector('[data-testid="composer-effort-label"]');
  var EFFORT_MENU = document.querySelector('[data-testid="composer-effort-menu"]');
  var BANNER = document.querySelector('[data-testid="composer-banner"]');
  var BANNER_MESSAGE = document.querySelector('[data-testid="banner-message"]');
  var BANNER_NEXT = document.querySelector('[data-testid="banner-next-action"]');
  var RETRY = document.querySelector('[data-testid="composer-retry"]');

  var state = {
    isStreaming: false,
    acceptsInput: true,
    isReady: false
  };

  // MARK: - Reporting to Swift

  function postToSwift(payload) {
    if (!window.webkit || !window.webkit.messageHandlers || !window.webkit.messageHandlers.thintalk) {
      return;
    }
    window.webkit.messageHandlers.thintalk.postMessage(payload);
  }

  function postAction(action, detail) {
    postToSwift({
      kind: "action",
      action: action,
      messageId: "",
      detail: detail || ""
    });
  }

  // MARK: - Send rules

  function trimmedDraft() {
    return (FIELD.value || "").trim();
  }

  function canSend() {
    return state.isReady && !state.isStreaming && trimmedDraft().length > 0;
  }

  function refreshControls() {
    var streaming = state.isStreaming;
    SEND.hidden = streaming;
    STOP.hidden = !streaming;
    STOP.disabled = !streaming;
    FIELD.disabled = !state.acceptsInput;
    // The placeholder keeps the field discoverable while disabled; the send rule
    // needs the draft, so it is evaluated from the live value.
    SEND.disabled = !canSend();
    growField();
  }

  function growField() {
    FIELD.style.height = "auto";
    var grownHeight = Math.min(FIELD.scrollHeight, 160);
    FIELD.style.height = grownHeight + "px";
    // The field scrolls only once the growth ceiling clamps it; a field that
    // still fits must never paint a scrollbar over the ask.
    FIELD.style.overflowY = FIELD.scrollHeight > grownHeight ? "auto" : "hidden";
  }

  function reportSend() {
    if (!canSend()) {
      return;
    }
    var ask = trimmedDraft();
    FIELD.value = "";
    refreshControls();
    postAction("send", ask);
  }

  // MARK: - Effort menu

  function closeEffortMenu() {
    EFFORT_MENU.hidden = true;
    EFFORT_PILL.setAttribute("aria-expanded", "false");
  }

  function toggleEffortMenu() {
    var opening = EFFORT_MENU.hidden;
    EFFORT_MENU.hidden = !opening;
    EFFORT_PILL.setAttribute("aria-expanded", opening ? "true" : "false");
  }

  function buildEffortMenu(options) {
    EFFORT_MENU.textContent = "";
    (options || []).forEach(function (option) {
      var item = document.createElement("li");
      item.setAttribute("role", "none");
      var choice = document.createElement("button");
      choice.type = "button";
      choice.setAttribute("role", "menuitemradio");
      choice.setAttribute("aria-checked", option.value === state.effortValue ? "true" : "false");
      choice.dataset.testid = "composer-effort-" + option.value;
      choice.dataset.effortValue = option.value;
      choice.textContent = option.budgetSummary;
      item.appendChild(choice);
      EFFORT_MENU.appendChild(item);
    });
  }

  // MARK: - State from Swift

  function applyState(command) {
    state.isStreaming = command.streaming === true;
    state.acceptsInput = command.acceptsInput !== false;
    state.isReady = command.ready === true;

    if (command.effort) {
      state.effortValue = command.effort.value;
      EFFORT_LABEL.textContent = command.effort.displayName;
      EFFORT_PILL.title = command.effort.budgetSummary;
    }
    buildEffortMenu(command.effortOptions);

    if (command.failure) {
      BANNER_MESSAGE.textContent = command.failure.message || "";
      BANNER_NEXT.textContent = command.failure.nextAction || "";
      BANNER.hidden = false;
    } else {
      BANNER.hidden = true;
    }

    if (!state.isReady) {
      closeEffortMenu();
    }
    refreshControls();
  }

  // MARK: - Events

  FIELD.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }
    event.preventDefault();
    reportSend();
  });

  FIELD.addEventListener("input", refreshControls);

  SEND.addEventListener("click", reportSend);

  STOP.addEventListener("click", function () {
    postAction("stop");
  });

  RETRY.addEventListener("click", function () {
    postAction("retry");
  });

  EFFORT_PILL.addEventListener("click", toggleEffortMenu);

  EFFORT_MENU.addEventListener("click", function (event) {
    var choice = event.target.closest("[data-effort-value]");
    if (!choice) {
      return;
    }
    closeEffortMenu();
    postAction("setEffort", choice.dataset.effortValue);
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closeEffortMenu();
    }
  });

  document.addEventListener("click", function (event) {
    if (!EFFORT_MENU.hidden && !event.target.closest(".composer__menu-anchor")) {
      closeEffortMenu();
    }
  });

  window.__thintalkComposer = {
    applyState: applyState
  };
})();
