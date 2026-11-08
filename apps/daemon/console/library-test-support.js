// Shared fixtures for the Library Observatory behavior tests.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const libraryScriptPath = path.join(__dirname, "library.js");
const libraryScript = fs.readFileSync(libraryScriptPath, "utf8");
const libraryRenderScriptPath = path.join(__dirname, "library-render.js");
const libraryRenderScript = fs.readFileSync(libraryRenderScriptPath, "utf8");
const observatoryShellPath = path.join(__dirname, "index.html");
const observatoryShell = fs.readFileSync(observatoryShellPath, "utf8");
const ASYNC_TEST_OPTIONS = { timeout: 5_000 };

function createLibraryContext() {
    const scriptContext = vm.createContext({
        AbortController,
        clearTimeout,
        document: { getElementById() { return null; } },
        setTimeout
    });
    vm.runInContext(libraryScript, scriptContext, { filename: libraryScriptPath });
    vm.runInContext(libraryRenderScript, scriptContext, { filename: libraryRenderScriptPath });
    return scriptContext;
}

function createLibraryElement(tagName, onElementRegistered, onElementFocused) {
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

function createLibraryDocument() {
    const elementByIdRegistry = {};
    const registerElementById = (element) => {
        if (element.id) elementByIdRegistry[element.id] = element;
    };
    const catalogContainer = createLibraryElement("div", registerElementById);
    const catalogStatus = createLibraryElement("p");
    catalogStatus.id = "library-catalog-status";
    catalogStatus.setAttribute("role", "status");
    catalogStatus.setAttribute("aria-live", "polite");
    catalogContainer.replaceChildren(catalogStatus);
    const shell = {
        catalogContainer,
        catalogStatus,
        activeElement: null,
        lastFocusOptions: null,
        createElement(tagName) {
            return createLibraryElement(tagName, registerElementById, (element, options) => {
                shell.activeElement = element;
                shell.lastFocusOptions = options;
            });
        },
        createTextNode(text) { return { tagName: "#text", textContent: text, children: [] }; },
        getElementById(elementId) {
            if (elementId === "library-catalog") { return catalogContainer; }
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
        // library-model-card-ready next to its base class.
        if (String(renderedElement.className).split(" ").includes(className)) matches.push(renderedElement);
        pendingElements.unshift(...renderedElement.children);
    }
    return matches;
}

// Schema 3 defaults: a chat model that is not a Get started pick, sized to
// memory the fit tests can compare against explicit ceilings.
function validCatalogEntry(overrides = {}) {
    return {
        huggingface_id: "astronomical-test/example-qwen",
        revision: "0123456789abcdef0123456789abcdef01234567",
        display_name: "Example model",
        family: "qwen3_5",
        approximate_size_bytes: 4_000_000_000,
        approximate_memory_bytes: 6_000_000_000,
        role: "chat",
        get_started: false,
        experimental: false,
        public: true,
        description: "A test model for exercising Library rendering.",
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
    createLibraryContext,
    createLibraryDocument,
    createLibraryElement,
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
