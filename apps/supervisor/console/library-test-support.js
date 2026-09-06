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

function createLibraryElement(tagName) {
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
        addEventListener() {},
        appendChild(child) { this.children.push(child); return child; },
        querySelectorAll() { return []; },
        removeAttribute(attributeName) { delete this.attributes[attributeName]; },
        setAttribute(attributeName, attributeValue) {
            this.attributes[attributeName] = attributeValue;
        },
        replaceChildren(...replacementChildren) {
            this.children = replacementChildren;
        }
    };
}

function createLibraryDocument() {
    const catalogContainer = createLibraryElement("div");
    const catalogStatus = createLibraryElement("p");
    catalogStatus.id = "library-catalog-status";
    catalogStatus.setAttribute("role", "status");
    catalogStatus.setAttribute("aria-live", "polite");
    catalogContainer.replaceChildren(catalogStatus);
    return {
        catalogContainer,
        catalogStatus,
        createElement: createLibraryElement,
        createTextNode(text) { return { tagName: "#text", textContent: text, children: [] }; },
        getElementById(elementId) {
            if (elementId === "library-catalog") { return catalogContainer; }
            if (elementId === "library-catalog-status") { return catalogStatus; }
            return null;
        }
    };
}

function validCatalogEntry(overrides = {}) {
    return {
        huggingface_id: "astronomical-test/example-qwen",
        revision: "0123456789abcdef0123456789abcdef01234567",
        display_name: "Example model",
        family: "qwen3_5",
        approximate_size_bytes: 4_000_000_000,
        public: true,
        description: "A test model for exercising Library rendering.",
        capabilities: {
            supports_reasoning: true,
            supports_vision: false,
            supports_tool_calls: true,
            context_window: 32768,
            max_output_tokens: 4096
        },
        quantization_label: "oQ6e (6-bit enhanced)",
        architecture_summary: "Test architecture",
        upstream_license: "MIT",
        ...overrides
    };
}

module.exports = {
    ASYNC_TEST_OPTIONS,
    assert,
    createLibraryContext,
    createLibraryDocument,
    createLibraryElement,
    fs,
    observatoryShell,
    path,
    test,
    validCatalogEntry,
    vm
};
