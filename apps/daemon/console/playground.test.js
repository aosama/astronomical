const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ASYNC_TEST_OPTIONS = { timeout: 5_000 };
const playgroundScriptPath = path.join(__dirname, "playground.js");
const playgroundScript = fs.readFileSync(playgroundScriptPath, "utf8");

// The chat view writes through a small set of element properties, so a fixture
// captures the visible contract a user reads instead of asserting on source
// layout. Each test gets a fresh context so module-level chat state (effort,
// font size, transcript history) never leaks between cases.
function createMockElement(tagName) {
    return {
        tagName: String(tagName).toUpperCase(),
        className: "",
        textContent: "",
        innerHTML: "",
        hidden: false,
        value: "",
        src: "",
        title: "",
        disabled: false,
        style: {},
        dataset: {},
        children: [],
        scrollTop: 0,
        scrollHeight: 0,
        setAttribute(attributeName, attributeValue) {
            this.dataset[attributeName] = attributeValue;
        },
        getAttribute(attributeName) {
            return this.dataset[attributeName];
        },
        appendChild(childElement) {
            this.children.push(childElement);
            return childElement;
        },
        remove() {},
        addEventListener() {},
        focus() {},
        contains() {
            return false;
        },
        querySelectorAll() {
            return [];
        },
        querySelector() {
            return null;
        }
    };
}

function createPlaygroundContext() {
    const elements = {
        "chat-send": createMockElement("button"),
        "chat-stop": createMockElement("button"),
        "chat-input": createMockElement("textarea"),
        "chat-image": createMockElement("input"),
        "chat-image-clear": createMockElement("button"),
        "chat-image-preview": createMockElement("img"),
        "chat-transcript": createMockElement("div"),
        "chat-error-banner": createMockElement("div"),
        "chat-effort": createMockElement("button"),
        "chat-effort-label": createMockElement("span"),
        "chat-effort-menu": createMockElement("div"),
        "font-size-decrease": createMockElement("button"),
        "font-size-increase": createMockElement("button"),
        "font-size-value": createMockElement("span")
    };
    const storage = {};
    const capturedRequests = [];
    const document = {
        getElementById(elementId) {
            return elements[elementId] || null;
        },
        createElement(tagName) {
            return createMockElement(tagName);
        },
        querySelectorAll() {
            return [];
        },
        addEventListener() {}
    };
    const localStorage = {
        getItem(key) {
            return Object.prototype.hasOwnProperty.call(storage, key) ? storage[key] : null;
        },
        setItem(key, value) {
            storage[key] = String(value);
        },
        removeItem(key) {
            delete storage[key];
        }
    };
    // A single content delta followed by [DONE] is enough to drive the whole
    // streaming path without a real worker.
    const fetch = async (url, options) => {
        capturedRequests.push({ url, options });
        const sseText =
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n';
        const encoder = new TextEncoder();
        let sent = false;
        return {
            ok: true,
            status: 200,
            text: async () => "",
            body: {
                getReader() {
                    return {
                        read() {
                            if (!sent) {
                                sent = true;
                                return Promise.resolve({ value: encoder.encode(sseText), done: false });
                            }
                            return Promise.resolve({ value: undefined, done: true });
                        }
                    };
                }
            }
        };
    };
    const scriptContext = vm.createContext({
        console: { log() {} },
        document,
        localStorage,
        fetch,
        AbortController,
        TextEncoder,
        TextDecoder,
        setTimeout,
        navigator: { clipboard: { writeText: async () => {} } },
        selectedModelId: "test-model"
    });
    vm.runInContext(playgroundScript, scriptContext, { filename: playgroundScriptPath });
    return { scriptContext, elements, storage, capturedRequests };
}

test("maps each effort level to its thinking budget", () => {
    const { scriptContext } = createPlaygroundContext();
    assert.equal(scriptContext.resolveEffortLevel("quick").thinkingBudget, 256);
    assert.equal(scriptContext.resolveEffortLevel("balanced").thinkingBudget, 512);
    assert.equal(scriptContext.resolveEffortLevel("high").thinkingBudget, 1024);
    // An unknown value falls back to the default level rather than throwing.
    assert.equal(scriptContext.resolveEffortLevel("unknown").value, "quick");
});

test("loads a persisted effort and falls back to the default when absent or invalid", () => {
    const { scriptContext, storage } = createPlaygroundContext();
    assert.equal(scriptContext.loadEffort(), "quick");
    storage["observatory:chat:effort"] = "high";
    assert.equal(scriptContext.loadEffort(), "high");
    storage["observatory:chat:effort"] = "not-a-level";
    assert.equal(scriptContext.loadEffort(), "quick");
});

test("loads a persisted font size and clamps out-of-range or non-numeric values", () => {
    const { scriptContext, storage } = createPlaygroundContext();
    assert.equal(scriptContext.loadFontSize(), 14);
    storage["observatory:chat:fontSize"] = "18";
    assert.equal(scriptContext.loadFontSize(), 18);
    storage["observatory:chat:fontSize"] = "5";
    assert.equal(scriptContext.loadFontSize(), 14);
    storage["observatory:chat:fontSize"] = "99";
    assert.equal(scriptContext.loadFontSize(), 14);
    storage["observatory:chat:fontSize"] = "abc";
    assert.equal(scriptContext.loadFontSize(), 14);
});

test("collects a text-only message and returns null when the input is empty", () => {
    const { scriptContext, elements } = createPlaygroundContext();
    elements["chat-input"].value = "   ";
    assert.equal(scriptContext.collectCurrentMessage(), null);
    elements["chat-input"].value = "  Say hello  ";
    const collectedMessage = scriptContext.collectCurrentMessage();
    assert.equal(collectedMessage.role, "user");
    assert.equal(collectedMessage.content, "Say hello");
});

test("renders visible user text and marks an attached image", () => {
    const { scriptContext } = createPlaygroundContext();
    assert.equal(
        scriptContext.visibleUserMessageText({ role: "user", content: "plain" }),
        "plain"
    );
    assert.equal(
        scriptContext.visibleUserMessageText({
            role: "user",
            content: [
                { type: "text", text: "look" },
                { type: "image_url", image_url: { url: "data:image/png;base64,xx" } }
            ]
        }),
        "look\n[Image attached]"
    );
});

test("parses error envelopes and falls back to a bounded status message", () => {
    const { scriptContext } = createPlaygroundContext();
    assert.equal(
        scriptContext.parseErrorEnvelope(JSON.stringify({ error: { message: "boom" } }), 500),
        "boom"
    );
    assert.equal(
        scriptContext.parseErrorEnvelope(JSON.stringify({ error: "plain error" }), 500),
        "plain error"
    );
    assert.equal(scriptContext.parseErrorEnvelope("not json", 503), "Request failed (HTTP 503)");
});

test("sends the selected effort as thinking_budget and omits sampling controls", async () => {
    const { scriptContext, elements, capturedRequests } = createPlaygroundContext();
    scriptContext.setEffort("high");
    elements["chat-input"].value = "Hello";
    await scriptContext.sendChat();

    assert.equal(capturedRequests.length, 1);
    const sentBody = JSON.parse(capturedRequests[0].options.body);
    assert.equal(sentBody.model, "test-model");
    assert.equal(sentBody.stream, true);
    assert.equal(sentBody.thinking_budget, 1024);
    // The console no longer exposes temperature, top-p, or token caps.
    assert.equal("temperature" in sentBody, false);
    assert.equal("top_p" in sentBody, false);
    assert.equal("max_tokens" in sentBody, false);
}, ASYNC_TEST_OPTIONS);

test("persists the selected effort to storage", () => {
    const { scriptContext, storage } = createPlaygroundContext();
    scriptContext.setEffort("balanced");
    assert.equal(storage["observatory:chat:effort"], "balanced");
});
