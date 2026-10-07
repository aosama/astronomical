// Astronomical Observatory chat playground.
//
// This owns the console chat: the request it sends, the effort pill that maps to
// the model's thinking budget, the text-size control, and the transcript it
// renders. Rendering is delegated to the shared Thin Talk render stack
// (window.__thintalkRenderer) so the console and the canvas answer the same
// question — "what HTML may this message insert?" — with one pipeline. The
// console keeps no copy of the markdown / KaTeX / Mermaid / sanitiser logic.

const CHAT_URL = "/v1/chat/completions";
const MAX_CHAT_REQUEST_BODY_BYTES = 32 * 1024 * 1024;
const MAX_IMAGE_BYTES_BEFORE_BASE64_EXPANSION = 24 * 1024 * 1024;

// Effort levels the console exposes. Each maps to a thinking budget (in tokens)
// the backend reads as thinking_budget. Temperature, top-p and token caps are
// intentionally absent: effort is the only model-control the console offers.
const EFFORT_LEVELS = [
    { value: "quick", label: "Quick", thinkingBudget: 256 },
    { value: "balanced", label: "Balanced", thinkingBudget: 512 },
    { value: "high", label: "High", thinkingBudget: 1024 },
];
const DEFAULT_EFFORT = "quick";
const EFFORT_STORAGE_KEY = "observatory:chat:effort";
const FONT_SIZE_STORAGE_KEY = "observatory:chat:fontSize";
const MIN_FONT_SIZE = 12;
const MAX_FONT_SIZE = 24;
const DEFAULT_FONT_SIZE = 14;

let currentChatAbortController = null;
let pendingImageDataUri = null;
const transcriptHistory = [];
let currentEffort = DEFAULT_EFFORT;
let currentFontSize = DEFAULT_FONT_SIZE;

function wirePlayground() {
    const sendButton = document.getElementById("chat-send");
    const stopButton = document.getElementById("chat-stop");
    const inputTextarea = document.getElementById("chat-input");
    const imageInput = document.getElementById("chat-image");
    const imageClearButton = document.getElementById("chat-image-clear");
    sendButton.addEventListener("click", sendChat);
    inputTextarea.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendChat();
        }
    });
    stopButton.addEventListener("click", stopChat);
    imageInput.addEventListener("change", handleImageSelected);
    imageClearButton.addEventListener("click", clearAttachedImage);
    wireEffortPill();
    wireFontSizeControls();
    document.getElementById("chat-transcript").addEventListener("click", handleTranscriptCommand);
    applyFontSize();
    updateEffortLabel();
}

function resolveEffortLevel(value) {
    return EFFORT_LEVELS.find((level) => level.value === value) || EFFORT_LEVELS[0];
}

function loadEffort() {
    try {
        const stored = localStorage.getItem(EFFORT_STORAGE_KEY);
        if (stored && resolveEffortLevel(stored).value === stored) {
            return stored;
        }
    } catch (storageError) {
        /* Storage may be unavailable; fall through to the default. */
    }
    return DEFAULT_EFFORT;
}

function wireEffortPill() {
    currentEffort = loadEffort();
    const pill = document.getElementById("chat-effort");
    const menu = document.getElementById("chat-effort-menu");
    pill.addEventListener("click", () => {
        const opening = menu.hidden;
        menu.hidden = !opening;
        pill.setAttribute("aria-expanded", opening ? "true" : "false");
    });
    menu.addEventListener("click", (event) => {
        const choice = event.target.closest("[data-effort]");
        if (!choice) {
            return;
        }
        setEffort(choice.dataset.effort);
        menu.hidden = true;
        pill.setAttribute("aria-expanded", "false");
    });
    document.addEventListener("click", (event) => {
        if (!menu.hidden && !pill.contains(event.target) && !menu.contains(event.target)) {
            menu.hidden = true;
            pill.setAttribute("aria-expanded", "false");
        }
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !menu.hidden) {
            menu.hidden = true;
            pill.setAttribute("aria-expanded", "false");
            pill.focus();
        }
    });
}

function setEffort(value) {
    currentEffort = value;
    try {
        localStorage.setItem(EFFORT_STORAGE_KEY, value);
    } catch (storageError) {
        /* Persistence is best-effort; the in-memory value still applies. */
    }
    updateEffortLabel();
}

function updateEffortLabel() {
    const level = resolveEffortLevel(currentEffort);
    document.getElementById("chat-effort-label").textContent = level.label;
    document.getElementById("chat-effort").title =
        "Thinking effort: " + level.label + " (" + level.thinkingBudget + " tokens)";
    document.querySelectorAll("#chat-effort-menu [data-effort]").forEach((choice) => {
        choice.setAttribute("aria-checked", choice.dataset.effort === currentEffort ? "true" : "false");
    });
}

function loadFontSize() {
    try {
        const stored = Number(localStorage.getItem(FONT_SIZE_STORAGE_KEY));
        if (Number.isInteger(stored) && stored >= MIN_FONT_SIZE && stored <= MAX_FONT_SIZE) {
            return stored;
        }
    } catch (storageError) {
        /* Storage may be unavailable; fall through to the default. */
    }
    return DEFAULT_FONT_SIZE;
}

function wireFontSizeControls() {
    currentFontSize = loadFontSize();
    document.getElementById("font-size-decrease").addEventListener("click", () => {
        setFontSize(Math.max(MIN_FONT_SIZE, currentFontSize - 1));
    });
    document.getElementById("font-size-increase").addEventListener("click", () => {
        setFontSize(Math.min(MAX_FONT_SIZE, currentFontSize + 1));
    });
    applyFontSize();
}

function setFontSize(size) {
    currentFontSize = size;
    try {
        localStorage.setItem(FONT_SIZE_STORAGE_KEY, String(size));
    } catch (storageError) {
        /* Persistence is best-effort; the in-memory value still applies. */
    }
    applyFontSize();
}

function applyFontSize() {
    const transcript = document.getElementById("chat-transcript");
    if (transcript) {
        transcript.style.fontSize = currentFontSize + "px";
    }
    document.getElementById("font-size-value").textContent = String(currentFontSize);
}

function handleImageSelected(event) {
    const selectedImageFile = event.target.files && event.target.files[0];
    if (!selectedImageFile) {
        return;
    }
    if (selectedImageFile.size > MAX_IMAGE_BYTES_BEFORE_BASE64_EXPANSION) {
        showChatError("The image is too large for the server's 32 MiB request limit.");
        event.target.value = "";
        return;
    }
    const imageFileReader = new FileReader();
    imageFileReader.onload = (loadEvent) => {
        pendingImageDataUri = loadEvent.target.result;
        const preview = document.getElementById("chat-image-preview");
        preview.src = pendingImageDataUri;
        preview.hidden = false;
        document.getElementById("chat-image-clear").hidden = false;
    };
    imageFileReader.onerror = () => {
        showChatError("Could not read the selected image.");
    };
    imageFileReader.readAsDataURL(selectedImageFile);
}

function clearAttachedImage() {
    pendingImageDataUri = null;
    const preview = document.getElementById("chat-image-preview");
    preview.src = "";
    preview.hidden = true;
    document.getElementById("chat-image-clear").hidden = true;
    document.getElementById("chat-image").value = "";
}

function collectCurrentMessage() {
    const inputTextarea = document.getElementById("chat-input");
    const text = inputTextarea.value.trim();
    if (!text && !pendingImageDataUri) {
        return null;
    }
    if (!pendingImageDataUri) {
        return { role: "user", content: text };
    }
    const messageContent = [];
    if (text) {
        messageContent.push({ type: "text", text: text });
    }
    messageContent.push({ type: "image_url", image_url: { url: pendingImageDataUri } });
    return { role: "user", content: messageContent };
}

function chatRequestFitsHttpBodyLimit(serializedRequestBody) {
    return new TextEncoder().encode(serializedRequestBody).byteLength <= MAX_CHAT_REQUEST_BODY_BYTES;
}

function visibleUserMessageText(message) {
    if (typeof message.content === "string") {
        return message.content;
    }
    const visibleParts = message.content
        .filter((contentPart) => contentPart.type === "text")
        .map((contentPart) => contentPart.text);
    if (message.content.some((contentPart) => contentPart.type === "image_url")) {
        visibleParts.push("[Image attached]");
    }
    return visibleParts.join("\n");
}

async function sendChat() {
    hideChatError();
    const currentMessage = collectCurrentMessage();
    if (!currentMessage) {
        return;
    }
    if (!selectedModelId) {
        showChatError("No model is available. Check the configured model directories.");
        return;
    }
    const requestBody = {
        model: selectedModelId,
        messages: transcriptHistory.concat([currentMessage]),
        stream: true,
        thinking_budget: resolveEffortLevel(currentEffort).thinkingBudget,
        stream_options: { include_usage: true },
    };
    const serializedRequestBody = JSON.stringify(requestBody);
    if (!chatRequestFitsHttpBodyLimit(serializedRequestBody)) {
        showChatError("This conversation is too large for the server's 32 MiB request limit.");
        return;
    }

    transcriptHistory.push(currentMessage);
    appendTranscriptMessage("user", {
        role: "user",
        markdown: visibleUserMessageText(currentMessage),
        state: "complete",
    });
    document.getElementById("chat-input").value = "";
    clearAttachedImage();
    const assistantHandle = appendTranscriptMessage("assistant", {
        role: "assistant",
        markdown: "",
        reasoning: "",
        state: "streaming",
    });
    const streamedState = { markdown: "", reasoning: "" };
    currentChatAbortController = new AbortController();
    setSendStopState(true);
    try {
        const response = await fetch(CHAT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: serializedRequestBody,
            signal: currentChatAbortController.signal,
        });
        if (!response.ok) {
            throw new Error(parseErrorEnvelope(await response.text(), response.status));
        }
        await streamChatResponse(response, assistantHandle, streamedState);
        if (!streamedState.markdown && !streamedState.reasoning) {
            assistantHandle.renderMessage.markdown = "(no output)";
        }
        assistantHandle.renderMessage.state = "complete";
        renderDisplayMessage(assistantHandle);
    } catch (requestError) {
        if (requestError.name !== "AbortError") {
            showChatError(requestError.message || "The local worker request failed.");
        }
    } finally {
        const assistantMessage = assistantHistoryMessage(streamedState);
        if (assistantMessage) {
            transcriptHistory.push(assistantMessage);
        } else {
            assistantHandle.element.remove();
            transcriptHistory.pop();
        }
        setSendStopState(false);
        currentChatAbortController = null;
    }
}

function assistantHistoryMessage(streamedState) {
    if (!streamedState.markdown && !streamedState.reasoning) {
        return null;
    }
    const assistantMessage = { role: "assistant" };
    if (streamedState.markdown) {
        assistantMessage.content = streamedState.markdown;
    }
    if (streamedState.reasoning) {
        assistantMessage.reasoning_content = streamedState.reasoning;
    }
    return assistantMessage;
}

async function streamChatResponse(response, assistantHandle, streamedState) {
    const responseReader = response.body.getReader();
    const textDecoder = new TextDecoder();
    let pendingEventText = "";
    while (true) {
        const { value: responseBytes, done: responseIsComplete } = await responseReader.read();
        if (responseIsComplete) {
            break;
        }
        pendingEventText += textDecoder.decode(responseBytes, { stream: true });
        const completeEvents = pendingEventText.split("\n\n");
        pendingEventText = completeEvents.pop();
        for (const eventText of completeEvents) {
            applyServerSentEvent(eventText, assistantHandle, streamedState);
        }
    }
}

function applyServerSentEvent(eventText, assistantHandle, streamedState) {
    const dataLine = eventText.split("\n").find((line) => line.startsWith("data:"));
    if (!dataLine) {
        return;
    }
    const payload = dataLine.slice(5).trim();
    if (payload === "[DONE]") {
        return;
    }
    let parsedPayload;
    try {
        parsedPayload = JSON.parse(payload);
    } catch (parseError) {
        return;
    }
    if (parsedPayload.error) {
        throw new Error(parsedPayload.error.message || "The local worker request failed.");
    }
    const delta = parsedPayload.choices && parsedPayload.choices[0] && parsedPayload.choices[0].delta;
    if (!delta) {
        return;
    }
    if (delta.reasoning_content) {
        streamedState.reasoning += delta.reasoning_content;
        assistantHandle.renderMessage.reasoning = streamedState.reasoning;
    }
    if (delta.content) {
        streamedState.markdown += delta.content;
        assistantHandle.renderMessage.markdown = streamedState.markdown;
        renderDisplayMessage(assistantHandle);
    }
}

function renderDisplayMessage(handle) {
    if (typeof __thintalkRenderer === "undefined" || typeof morphdom === "undefined") {
        renderPlainTextFallback(handle);
        return;
    }
    const article = handle.element;
    const probe = document.createElement("div");
    probe.className = "message__probe";
    probe.innerHTML = __thintalkRenderer.messageHtml(handle.renderMessage);
    __thintalkRenderer.labelCodeBlocks(probe);
    __thintalkRenderer.stampExpensiveKeys(probe);
    morphdom(article, probe, {
        childrenOnly: true,
        getNodeKey: __thintalkRenderer.nodeKey,
        onBeforeElUpdated: __thintalkRenderer.beforeElUpdated,
    });
    __thintalkRenderer.highlightCodeBlocks(article);
    if (handle.renderMessage.state === "complete" || handle.renderMessage.state === "stopped") {
        __thintalkRenderer.renderDiagrams(article);
    }
}

function renderPlainTextFallback(handle) {
    handle.element.textContent = "";
    const fallback = document.createElement("div");
    fallback.className = "chat-message-fallback";
    fallback.textContent = handle.renderMessage.markdown || "";
    handle.element.appendChild(fallback);
}

function handleTranscriptCommand(event) {
    const copyButton = event.target.closest('[data-action="copy"]');
    if (!copyButton) {
        return;
    }
    const answer = copyButton.closest(".message__answer") || copyButton.closest(".chat-message");
    const text = answer ? answer.textContent : "";
    navigator.clipboard.writeText(text).then(
        () => {
            copyButton.textContent = "Copied";
            setTimeout(() => {
                copyButton.textContent = "Copy";
            }, 1200);
        },
        () => {
            showChatError("Could not copy to the clipboard.");
        },
    );
}

function appendTranscriptMessage(role, renderMessage) {
    const transcript = document.getElementById("chat-transcript");
    const article = document.createElement("div");
    article.className = "chat-message chat-message-" + role;
    article.setAttribute("data-role", role);
    transcript.appendChild(article);
    const handle = { element: article, renderMessage: renderMessage };
    renderDisplayMessage(handle);
    transcript.scrollTop = transcript.scrollHeight;
    return handle;
}

function stopChat() {
    if (currentChatAbortController) {
        currentChatAbortController.abort();
    }
}

function setSendStopState(streaming) {
    document.getElementById("chat-send").disabled = streaming;
    document.getElementById("chat-stop").disabled = !streaming;
}

function showChatError(message) {
    const banner = document.getElementById("chat-error-banner");
    banner.textContent = message;
    banner.hidden = false;
}

function hideChatError() {
    const banner = document.getElementById("chat-error-banner");
    banner.textContent = "";
    banner.hidden = true;
}

function parseErrorEnvelope(bodyText, statusCode) {
    try {
        const parsedBody = JSON.parse(bodyText);
        if (parsedBody.error && parsedBody.error.message) {
            return parsedBody.error.message;
        }
        if (parsedBody.error && typeof parsedBody.error === "string") {
            return parsedBody.error;
        }
    } catch (parseError) {
        /* Fall through to the bounded status message. */
    }
    return "Request failed (HTTP " + statusCode + ")";
}
