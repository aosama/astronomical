// Focused Library behavior tests keep the Observatory navigation suite bounded.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const libraryScriptPath = path.join(__dirname, "library.js");
const libraryScript = fs.readFileSync(libraryScriptPath, "utf8");
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
        classList: { add() {}, remove() {} },
        addEventListener() {},
        appendChild(child) { this.children.push(child); return child; },
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
        ...overrides
    };
}

test("renders an empty catalog through the bounded empty state", () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogDocument = { schema_version: 1, entries: [] };

    const renderedState = vm.runInContext(
        "renderLibraryCatalogDocument(catalogDocument)",
        scriptContext
    );

    assert.equal(renderedState, "empty");
    assert.equal(libraryDocument.catalogContainer.dataset.libraryState, "empty");
    assert.equal(libraryDocument.catalogContainer.children.length, 1);
    assert.equal(libraryDocument.catalogContainer.children[0].attributes.role, "status");
});

test("retains catalog order and formats sizes as decimal SI gigabytes", () => {
    const scriptContext = createLibraryContext();
    scriptContext.catalogDocument = {
        schema_version: 1,
        entries: [
            validCatalogEntry({ display_name: "First", approximate_size_bytes: 1_500_000_000 }),
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-laguna",
                display_name: "Second",
                family: "laguna",
                approximate_size_bytes: 2_000_000_000
            })
        ]
    };

    const catalogRows = vm.runInContext(
        "libraryCatalogRowsFromDocument(catalogDocument)",
        scriptContext
    );

    assert.deepEqual(JSON.parse(JSON.stringify(catalogRows)), [
        { displayName: "First", family: "qwen3_5", approximateSize: "1.50 GB" },
        { displayName: "Second", family: "laguna", approximateSize: "2.00 GB" }
    ]);
});

test("accepts the largest catalog size that JavaScript can represent exactly", () => {
    const scriptContext = createLibraryContext();
    scriptContext.catalogEntry = validCatalogEntry({
        display_name: "Largest exact size",
        approximate_size_bytes: Number.MAX_SAFE_INTEGER
    });

    assert.equal(
        vm.runInContext("isRenderableLibraryCatalogEntry(catalogEntry)", scriptContext),
        true
    );
    scriptContext.catalogEntry.approximate_size_bytes = Number.MAX_SAFE_INTEGER + 1;
    assert.equal(
        vm.runInContext("isRenderableLibraryCatalogEntry(catalogEntry)", scriptContext),
        false
    );
});

test("renders future catalog entries as read-only text with safe document operations", () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogDocument = {
        schema_version: 1,
        entries: [
            validCatalogEntry({
                display_name: "<img src=x onerror=alert(1)>",
                approximate_size_bytes: 3_000_000_000
            })
        ]
    };

    const renderedState = vm.runInContext(
        "renderLibraryCatalogDocument(catalogDocument)",
        scriptContext
    );
    const renderedTags = [];
    const pendingElements = [...libraryDocument.catalogContainer.children];
    while (pendingElements.length > 0) {
        const renderedElement = pendingElements.shift();
        renderedTags.push(renderedElement.tagName);
        pendingElements.push(...renderedElement.children);
    }

    assert.equal(renderedState, "ready");
    assert.equal(libraryDocument.catalogStatus.textContent, "Model catalog loaded.");
    assert.equal(renderedTags.includes("BUTTON"), false);
    assert.equal(renderedTags.includes("INPUT"), false);
    assert.equal(renderedTags.includes("A"), false);
    assert.equal(
        libraryDocument.catalogContainer.children[1].children[0].children[0].textContent,
        "<img src=x onerror=alert(1)>"
    );
});

test("rejects an incomplete entry even when its visible fields are renderable", () => {
    const scriptContext = createLibraryContext();
    scriptContext.catalogEntry = {
        display_name: "Incomplete model",
        family: "qwen3_5",
        approximate_size_bytes: 3_000_000_000
    };

    assert.equal(
        vm.runInContext("isRenderableLibraryCatalogEntry(catalogEntry)", scriptContext),
        false
    );
});

test("fetches the immutable catalog once when Library wiring is repeated", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    let catalogFetchCount = 0;
    scriptContext.document = libraryDocument;
    scriptContext.fetch = async () => {
        catalogFetchCount += 1;
        return { ok: true, async json() { return { schema_version: 1, entries: [] }; } };
    };

    await vm.runInContext(
        "Promise.all([wireLibraryCatalog(), wireLibraryCatalog()])",
        scriptContext
    );

    assert.equal(catalogFetchCount, 1);
    assert.equal(libraryDocument.catalogContainer.dataset.libraryState, "empty");
});

test("renders unavailable state for malformed, non-success, and failed catalog responses", ASYNC_TEST_OPTIONS, async () => {
    const malformedContext = createLibraryContext();
    const malformedDocument = createLibraryDocument();
    malformedContext.document = malformedDocument;
    malformedContext.catalogDocument = { schema_version: 1, entries: [{}] };
    assert.equal(
        vm.runInContext("renderLibraryCatalogDocument(catalogDocument)", malformedContext),
        "unavailable"
    );

    const nonSuccessContext = createLibraryContext();
    const nonSuccessDocument = createLibraryDocument();
    nonSuccessContext.document = nonSuccessDocument;
    nonSuccessContext.fetch = async () => ({
        ok: false,
        async json() { return { schema_version: 1, entries: [] }; }
    });
    await vm.runInContext("wireLibraryCatalog()", nonSuccessContext);
    assert.equal(nonSuccessDocument.catalogContainer.dataset.libraryState, "unavailable");

    const failedContext = createLibraryContext();
    const failedDocument = createLibraryDocument();
    failedContext.document = failedDocument;
    failedContext.fetch = async () => { throw new Error("catalog unavailable"); };
    await vm.runInContext("wireLibraryCatalog()", failedContext);

    assert.equal(failedDocument.catalogContainer.dataset.libraryState, "unavailable");
});

test("bounds a catalog request that never completes", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.fetch = (unusedUrl, requestOptions) => new Promise((unusedResolve, reject) => {
        requestOptions.signal.addEventListener("abort", () => {
            reject(new Error("catalog request aborted"));
        }, { once: true });
    });
    scriptContext.catalogContainer = libraryDocument.catalogContainer;

    const renderedState = await vm.runInContext(
        "loadLibraryCatalog(catalogContainer, 1)",
        scriptContext
    );

    assert.equal(renderedState, "unavailable");
    assert.equal(libraryDocument.catalogContainer.dataset.libraryState, "unavailable");
});

test("loads the shipped scripts and resolves the startup Library journey", ASYNC_TEST_OPTIONS, async () => {
    const domReadyHandlers = [];
    const requestedPaths = [];
    const libraryDocument = createLibraryDocument();
    const genericElements = new Map();
    libraryDocument.body = createLibraryElement("body");
    libraryDocument.addEventListener = (eventName, eventHandler) => {
        if (eventName === "DOMContentLoaded") { domReadyHandlers.push(eventHandler); }
    };
    libraryDocument.querySelectorAll = () => [];
    const libraryCatalogContainer = libraryDocument.catalogContainer;
    libraryDocument.getElementById = (elementId) => {
        if (elementId === "library-catalog") { return libraryCatalogContainer; }
        if (elementId === "library-catalog-status") { return libraryDocument.catalogStatus; }
        if (!genericElements.has(elementId)) {
            genericElements.set(elementId, createLibraryElement("div"));
        }
        return genericElements.get(elementId);
    };
    const scriptContext = vm.createContext({
        AbortController,
        clearTimeout,
        console: { log() {} },
        document: libraryDocument,
        fetch: async (requestPath) => {
            requestedPaths.push(requestPath);
            if (requestPath === "/v1/library/catalog") {
                return { ok: true, async json() { return { schema_version: 1, entries: [] }; } };
            }
            return { ok: false, async json() { return {}; } };
        },
        history: { pushState() {}, replaceState() {}, state: null },
        setInterval() {},
        setTimeout,
        TextEncoder,
        window: { location: { pathname: "/library" }, addEventListener() {} }
    });
    const scriptSources = Array.from(
        observatoryShell.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/g),
        (scriptMatch) => scriptMatch[1]
    );
    for (const scriptSource of scriptSources) {
        const shippedScriptPath = path.join(__dirname, path.basename(scriptSource));
        const shippedScript = fs.readFileSync(shippedScriptPath, "utf8");
        vm.runInContext(shippedScript, scriptContext, { filename: shippedScriptPath });
    }

    assert.equal(domReadyHandlers.length, 1);
    domReadyHandlers[0]();
    await vm.runInContext("libraryCatalogLoadPromise", scriptContext);

    assert.equal(
        requestedPaths.filter((requestPath) => requestPath === "/v1/library/catalog").length,
        1
    );
    assert.equal(libraryCatalogContainer.dataset.libraryState, "empty");
});
