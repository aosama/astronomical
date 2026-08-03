// Astronomical Observatory chat playground.

const CHAT_URL = "/v1/chat/completions";
const MAX_CHAT_REQUEST_BODY_BYTES = 32 * 1024 * 1024;
const MAX_IMAGE_BYTES_BEFORE_BASE64_EXPANSION = 24 * 1024 * 1024;

let currentChatAbortController = null;
let pendingImageDataUri = null;
const transcriptHistory = [];

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
    const temperatureSlider = document.getElementById("sampling-temperature");
    const topPSlider = document.getElementById("sampling-top-p");
    const maxTokensSlider = document.getElementById("sampling-max-tokens");
    temperatureSlider.addEventListener("input", () => {
        document.getElementById("sampling-temperature-value").textContent =
            parseFloat(temperatureSlider.value).toFixed(2);
    });
    topPSlider.addEventListener("input", () => {
        document.getElementById("sampling-top-p-value").textContent =
            parseFloat(topPSlider.value).toFixed(2);
    });
    maxTokensSlider.addEventListener("input", () => {
        document.getElementById("sampling-max-tokens-value").textContent =
            parseInt(maxTokensSlider.value, 10).toLocaleString();
    });
}

function handleImageSelected(event) {
    const selectedImageFile = event.target.files && event.target.files[0];
    if (!selectedImageFile) { return; }
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
    imageFileReader.onerror = () => { showChatError("Could not read the selected image."); };
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
    if (!text && !pendingImageDataUri) { return null; }
    if (!pendingImageDataUri) { return { role: "user", content: text }; }
    const messageContent = [];
    if (text) { messageContent.push({ type: "text", text: text }); }
    messageContent.push({ type: "image_url", image_url: { url: pendingImageDataUri } });
    return { role: "user", content: messageContent };
}

function chatRequestFitsHttpBodyLimit(serializedRequestBody) {
    return new TextEncoder().encode(serializedRequestBody).byteLength
        <= MAX_CHAT_REQUEST_BODY_BYTES;
}

function assistantHistoryMessage(streamedResponse) {
    if (!streamedResponse.assistantText && !streamedResponse.reasoningText) { return null; }
    const assistantMessage = { role: "assistant" };
    if (streamedResponse.assistantText) {
        assistantMessage.content = streamedResponse.assistantText;
    }
    if (streamedResponse.reasoningText) {
        assistantMessage.reasoning_content = streamedResponse.reasoningText;
    }
    return assistantMessage;
}

function visibleUserMessageText(message) {
    if (typeof message.content === "string") { return message.content; }
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
    if (!currentMessage) { return; }
    if (!selectedModelId) {
        showChatError("No model is available. Check the configured model directories.");
        return;
    }
    const requestBody = {
        model: selectedModelId,
        messages: transcriptHistory.concat([currentMessage]),
        stream: true,
        temperature: parseFloat(document.getElementById("sampling-temperature").value),
        top_p: parseFloat(document.getElementById("sampling-top-p").value),
        max_tokens: parseInt(document.getElementById("sampling-max-tokens").value, 10),
        stream_options: { include_usage: true }
    };
    const serializedRequestBody = JSON.stringify(requestBody);
    if (!chatRequestFitsHttpBodyLimit(serializedRequestBody)) {
        showChatError("This conversation is too large for the server's 32 MiB request limit.");
        return;
    }

    transcriptHistory.push(currentMessage);
    appendTranscriptMessage("user", currentMessage);
    document.getElementById("chat-input").value = "";
    clearAttachedImage();
    const assistantBubble = appendTranscriptMessage("assistant", null);
    const reasoningBubble = appendTranscriptMessage("reasoning", null);
    reasoningBubble.element.hidden = true;
    const streamedResponse = { assistantText: "", reasoningText: "" };
    currentChatAbortController = new AbortController();
    setSendStopState(true);
    const rawStream = document.getElementById("raw-stream");
    rawStream.textContent = "";
    try {
        const response = await fetch(CHAT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: serializedRequestBody,
            signal: currentChatAbortController.signal
        });
        if (!response.ok) {
            throw new Error(parseErrorEnvelope(await response.text(), response.status));
        }
        await streamChatResponse(response, assistantBubble, reasoningBubble, rawStream, streamedResponse);
        if (!streamedResponse.assistantText && !streamedResponse.reasoningText) {
            assistantBubble.textNode.textContent = "(no output)";
        }
    } catch (requestError) {
        if (requestError.name !== "AbortError") {
            showChatError(requestError.message || "The local worker request failed.");
        }
    } finally {
        const assistantMessage = assistantHistoryMessage(streamedResponse);
        if (assistantMessage) {
            transcriptHistory.push(assistantMessage);
        } else {
            assistantBubble.element.remove();
            reasoningBubble.element.remove();
            transcriptHistory.pop();
        }
        setSendStopState(false);
        currentChatAbortController = null;
    }
}

async function streamChatResponse(response, assistantBubble, reasoningBubble, rawStream, streamedResponse) {
    const responseReader = response.body.getReader();
    const textDecoder = new TextDecoder();
    let pendingEventText = "";
    while (true) {
        const { value: responseBytes, done: responseIsComplete } = await responseReader.read();
        if (responseIsComplete) { break; }
        pendingEventText += textDecoder.decode(responseBytes, { stream: true });
        const completeEvents = pendingEventText.split("\n\n");
        pendingEventText = completeEvents.pop();
        for (const eventText of completeEvents) {
            applyServerSentEvent(eventText, assistantBubble, reasoningBubble, rawStream, streamedResponse);
        }
    }
}

function applyServerSentEvent(eventText, assistantBubble, reasoningBubble, rawStream, streamedResponse) {
    const dataLine = eventText.split("\n").find((line) => line.startsWith("data:"));
    if (!dataLine) { return; }
    const payload = dataLine.slice(5).trim();
    if (payload === "[DONE]") { return; }
    rawStream.textContent += payload + "\n\n";
    let parsedPayload;
    try { parsedPayload = JSON.parse(payload); } catch (parseError) { return; }
    if (parsedPayload.error) {
        throw new Error(parsedPayload.error.message || "The local worker request failed.");
    }
    const delta = parsedPayload.choices && parsedPayload.choices[0]
        && parsedPayload.choices[0].delta;
    if (!delta) { return; }
    if (delta.reasoning_content) {
        streamedResponse.reasoningText += delta.reasoning_content;
        reasoningBubble.element.hidden = false;
        reasoningBubble.textNode.textContent = streamedResponse.reasoningText;
    }
    if (delta.content) {
        streamedResponse.assistantText += delta.content;
        renderAssistantText(assistantBubble, streamedResponse.assistantText);
    }
}

function renderAssistantText(assistantBubble, fullText) {
    const container = assistantBubble.element;
    container.textContent = "";
    const fenceMarker = "```";
    const segments = [];
    let cursor = 0;
    while (cursor < fullText.length) {
        const fenceStart = fullText.indexOf(fenceMarker, cursor);
        if (fenceStart === -1) {
            segments.push({ kind: "text", text: fullText.slice(cursor) });
            break;
        }
        if (fenceStart > cursor) {
            segments.push({ kind: "text", text: fullText.slice(cursor, fenceStart) });
        }
        const codeStart = fenceStart + fenceMarker.length;
        const fenceEnd = fullText.indexOf(fenceMarker, codeStart);
        if (fenceEnd === -1) {
            segments.push({ kind: "code", text: fullText.slice(codeStart) });
            break;
        }
        segments.push({ kind: "code", text: fullText.slice(codeStart, fenceEnd) });
        cursor = fenceEnd + fenceMarker.length;
    }
    for (const segment of segments) {
        if (segment.kind === "code") {
            const preformattedText = document.createElement("pre");
            preformattedText.textContent = stripLeadingNewline(segment.text);
            container.appendChild(preformattedText);
        } else {
            container.appendChild(document.createTextNode(segment.text));
        }
    }
}

function stripLeadingNewline(text) {
    return text.charAt(0) === "\n" ? text.slice(1) : text;
}

function stopChat() {
    if (currentChatAbortController) { currentChatAbortController.abort(); }
}

function setSendStopState(streaming) {
    document.getElementById("chat-send").disabled = streaming;
    document.getElementById("chat-stop").disabled = !streaming;
}

function appendTranscriptMessage(role, message) {
    const transcript = document.getElementById("chat-transcript");
    const wrapper = document.createElement("div");
    wrapper.className = "chat-message chat-message-" + role;
    wrapper.setAttribute("data-role", role);
    const textNode = document.createTextNode("");
    if (role === "reasoning") {
        const labelNode = document.createElement("span");
        labelNode.textContent = "Reasoning:\n";
        wrapper.appendChild(labelNode);
    } else if (role === "user" && message) {
        textNode.textContent = visibleUserMessageText(message);
    }
    wrapper.appendChild(textNode);
    transcript.appendChild(wrapper);
    transcript.scrollTop = transcript.scrollHeight;
    return { element: wrapper, textNode: textNode };
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
        if (parsedBody.error && parsedBody.error.message) { return parsedBody.error.message; }
        if (parsedBody.error && typeof parsedBody.error === "string") { return parsedBody.error; }
    } catch (parseError) { /* Fall through to the bounded status message. */ }
    return "Request failed (HTTP " + statusCode + ")";
}
