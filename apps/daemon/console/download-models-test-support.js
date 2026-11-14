// Shared fixtures for the Observatory Models page behavior tests.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const downloadModelsScriptPath = path.join(__dirname, "download-models.js");
const downloadModelsScript = fs.readFileSync(downloadModelsScriptPath, "utf8");
const downloadModelsRenderScriptPath = path.join(__dirname, "download-models-render.js");
const downloadModelsRenderScript = fs.readFileSync(downloadModelsRenderScriptPath, "utf8");
const downloadModelsGroupsScriptPath = path.join(__dirname, "download-models-groups.js");
const downloadModelsGroupsScript = fs.readFileSync(downloadModelsGroupsScriptPath, "utf8");
const downloadModelsRowScriptPath = path.join(__dirname, "download-models-row.js");
const downloadModelsRowScript = fs.readFileSync(downloadModelsRowScriptPath, "utf8");
const downloadModelsLinkedScriptPath = path.join(__dirname, "download-models-linked.js");
const downloadModelsLinkedScript = fs.readFileSync(downloadModelsLinkedScriptPath, "utf8");
const downloadModelsInstalledScriptPath = path.join(__dirname, "download-models-installed.js");
const downloadModelsInstalledScript = fs.readFileSync(downloadModelsInstalledScriptPath, "utf8");
const downloadModelsRemovalScriptPath = path.join(__dirname, "download-models-removal.js");
const downloadModelsRemovalScript = fs.readFileSync(downloadModelsRemovalScriptPath, "utf8");
const observatoryShellPath = path.join(__dirname, "index.html");
const observatoryShell = fs.readFileSync(observatoryShellPath, "utf8");
const ASYNC_TEST_OPTIONS = { timeout: 5_000 };

function createDownloadModelsContext() {
    const scriptContext = vm.createContext({
        AbortController,
        clearTimeout,
        document: { getElementById() { return null; } },
        setTimeout,
        URLSearchParams
    });
    vm.runInContext(downloadModelsScript, scriptContext, { filename: downloadModelsScriptPath });
    vm.runInContext(downloadModelsGroupsScript, scriptContext, { filename: downloadModelsGroupsScriptPath });
    vm.runInContext(downloadModelsRenderScript, scriptContext, { filename: downloadModelsRenderScriptPath });
    vm.runInContext(downloadModelsRowScript, scriptContext, { filename: downloadModelsRowScriptPath });
    vm.runInContext(downloadModelsLinkedScript, scriptContext, { filename: downloadModelsLinkedScriptPath });
    vm.runInContext(downloadModelsInstalledScript, scriptContext, { filename: downloadModelsInstalledScriptPath });
    vm.runInContext(downloadModelsRemovalScript, scriptContext, { filename: downloadModelsRemovalScriptPath });
    return scriptContext;
}

function createDownloadModelsElement(tagName, onElementRegistered, onElementFocused) {
    return {
        tagName: tagName.toUpperCase(),
        className: "",
        dataset: {},
        attributes: {},
        textContent: "",
        children: [],
        hidden: false,
        style: {},
        value: "",
        classList: { add() {}, remove() {} },
        listeners: {},
        addEventListener(eventType, listener) {
            (this.listeners[eventType] ||= []).push(listener);
        },
        focus(options = {}) {
            if (onElementFocused) onElementFocused(this, options);
        },
        setSelectionRange(startIndex, endIndex) {
            this.selectionStart = startIndex;
            this.selectionEnd = endIndex;
            this.selectionRange = [startIndex, endIndex];
        },
        // Returns the event so tests can assert on defaultPrevented after a handled key.
        dispatch(eventType, eventInit = {}) {
            const dispatchedEvent = {
                type: eventType,
                defaultPrevented: false,
                preventDefault() { this.defaultPrevented = true; },
                ...eventInit
            };
            for (const listener of this.listeners[eventType] || []) listener(dispatchedEvent);
            return dispatchedEvent;
        },
        appendChild(child) {
            this.children.push(child);
            if (onElementRegistered) onElementRegistered(child);
            return child;
        },
        querySelectorAll() { return []; },
        removeAttribute(attributeName) { delete this.attributes[attributeName]; },
        setAttribute(attributeName, attributeValue) {
            this.attributes[attributeName] = attributeValue;
        },
        replaceChildren(...replacementChildren) {
            this.children = replacementChildren;
            if (onElementRegistered) {
                for (const child of replacementChildren) onElementRegistered(child);
            }
        }
    };
}

function createDownloadModelsDocument() {
    const elementByIdRegistry = {};
    const registerElementById = (element) => {
        if (element.id) elementByIdRegistry[element.id] = element;
    };
    const catalogContainer = createDownloadModelsElement("div", registerElementById);
    const catalogStatus = createDownloadModelsElement("p");
    catalogStatus.id = "download-models-catalog-status";
    catalogStatus.setAttribute("role", "status");
    catalogStatus.setAttribute("aria-live", "polite");
    catalogContainer.replaceChildren(catalogStatus);
    const shell = {
        catalogContainer,
        catalogStatus,
        activeElement: null,
        lastFocusOptions: null,
        createElement(tagName) {
            return createDownloadModelsElement(tagName, registerElementById, (element, options) => {
                shell.activeElement = element;
                shell.lastFocusOptions = options;
            });
        },
        createTextNode(text) { return { tagName: "#text", textContent: text, children: [] }; },
        getElementById(elementId) {
            if (elementId === "download-models-catalog") { return catalogContainer; }
            return elementByIdRegistry[elementId] || null;
        }
    };
    return shell;
}

// Depth-first search of the rendered tree, so tests can reach a card or control
// inside a page without depending on how many wrappers the page adds.
function findRenderedElement(rootElement, elementPredicate) {
    const pendingElements = [rootElement];
    while (pendingElements.length > 0) {
        const renderedElement = pendingElements.shift();
        if (elementPredicate(renderedElement)) return renderedElement;
        pendingElements.unshift(...renderedElement.children);
    }
    return undefined;
}

function findRenderedElementsByClass(rootElement, className) {
    const matches = [];
    const pendingElements = [rootElement];
    while (pendingElements.length > 0) {
        const renderedElement = pendingElements.shift();
        // Match one class token, because a card carries modifiers such as
        // download-models-model-card-ready next to its base class.
        if (String(renderedElement.className).split(" ").includes(className)) matches.push(renderedElement);
        pendingElements.unshift(...renderedElement.children);
    }
    return matches;
}

// Schema 5 defaults: a chat model that is not a Get started pick, sized to
// memory the fit tests can compare against explicit ceilings. Its one name, the
// model ID, is the leaf of its Hugging Face identity, as the daemon publishes it,
// unless a test names it outright.
function validCatalogEntry(overrides = {}) {
    const huggingfaceId = overrides.huggingface_id || "astronomical-test/example-qwen";
    return {
        huggingface_id: huggingfaceId,
        model_id: huggingfaceId.split("/").pop(),
        revision: "0123456789abcdef0123456789abcdef01234567",
        family: "qwen3_5",
        approximate_size_bytes: 4_000_000_000,
        approximate_memory_bytes: 6_000_000_000,
        role: "chat",
        get_started: false,
        experimental: false,
        public: true,
        description: "A test model for exercising Models page rendering.",
        capabilities: {
            supports_reasoning: true,
            supports_vision: false,
            supports_tool_calls: true,
            context_window: 32768,
            max_output_tokens: 4096
        },
        quantization_label: "6-bit affine (group 64)",
        upstream_license: "MIT",
        ...overrides
    };
}

// Every text node under a rendered element, depth first, so tests can assert
// what a person would read without depending on the exact element tree.
function renderedTexts(rootElement) {
    const texts = [];
    const pendingElements = [rootElement];
    while (pendingElements.length > 0) {
        const renderedElement = pendingElements.shift();
        texts.push(renderedElement.textContent);
        pendingElements.unshift(...renderedElement.children);
    }
    return texts;
}

module.exports = {
    ASYNC_TEST_OPTIONS,
    assert,
    createDownloadModelsContext,
    createDownloadModelsDocument,
    createDownloadModelsElement,
    findRenderedElement,
    findRenderedElementsByClass,
    fs,
    observatoryShell,
    path,
    renderedTexts,
    test,
    validCatalogEntry,
    vm
};
