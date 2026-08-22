// Focused Library behavior tests keep the Observatory navigation suite bounded.

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
        {
            huggingfaceId: "astronomical-test/example-qwen",
            displayName: "First",
            family: "qwen3_5",
            approximateSize: "1.50 GB",
            approximateSizeBytes: 1_500_000_000,
            readyOnThisMac: false,
            destinationDirectory: null,
            downloadState: null,
            description: "A test model for exercising Library rendering.",
            quantizationLabel: "oQ6e (6-bit enhanced)",
            architectureSummary: "Test architecture",
            upstreamLicense: "MIT",
            requestableModelId: null,
            supportsReasoning: true,
            supportsVision: false,
            supportsToolCalls: true,
            supportsImageGeneration: false,
            contextWindow: 32768,
            maxOutputTokens: 4096
        },
        {
            huggingfaceId: "astronomical-test/example-laguna",
            displayName: "Second",
            family: "laguna",
            approximateSize: "2.00 GB",
            approximateSizeBytes: 2_000_000_000,
            readyOnThisMac: false,
            destinationDirectory: null,
            downloadState: null,
            description: "A test model for exercising Library rendering.",
            quantizationLabel: "oQ6e (6-bit enhanced)",
            architectureSummary: "Test architecture",
            upstreamLicense: "MIT",
            requestableModelId: null,
            supportsReasoning: true,
            supportsVision: false,
            supportsToolCalls: true,
            supportsImageGeneration: false,
            contextWindow: 32768,
            maxOutputTokens: 4096
        }
    ]);
});

test("presents progress and only the controls valid for each durable download state", () => {
    const scriptContext = createLibraryContext();
    scriptContext.document = createLibraryDocument();
    scriptContext.catalogRow = {
        huggingfaceId: "astronomical-test/example-qwen",
        displayName: "Example model",
        family: "qwen3_5",
        approximateSize: "4.00 GB",
        readyOnThisMac: false,
        downloadState: "downloading"
    };
    vm.runInContext(
        `libraryCurrentDownload = {
            state: "downloading",
            huggingface_id: "astronomical-test/example-qwen",
            bytes_completed: 7_500_000_000,
            bytes_total: 30_000_000_000
        }`,
        scriptContext
    );

    assert.equal(
        vm.runInContext("libraryStateTitle(catalogRow)", scriptContext),
        "Downloading 25% · 7.50 GB of 30.00 GB"
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(vm.runInContext(
            "libraryActionButtons(catalogRow).map(button => button.textContent)",
            scriptContext
        ))),
        ["Pause", "Cancel"]
    );
    assert.equal(vm.runInContext("formatLibraryRate(27_500_000)", scriptContext), "27.5 MB/s");
    assert.equal(
        vm.runInContext("formatLibraryRemainingTime(16 * 60)", scriptContext),
        "About 16 minutes left"
    );

    vm.runInContext(
        `libraryCurrentDownload = {
            state: "publishing",
            huggingface_id: "astronomical-test/example-qwen",
            bytes_completed: 30_000_000_000,
            bytes_total: 30_000_000_000
        }`,
        scriptContext
    );
    assert.equal(
        vm.runInContext("libraryStateTitle(catalogRow)", scriptContext),
        "Download complete. Adding it to Library…"
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(vm.runInContext(
            "libraryActionButtons(catalogRow).map(button => button.textContent)",
            scriptContext
        ))),
        []
    );

    vm.runInContext(
        `libraryCurrentDownload = {
            state: "failed",
            huggingface_id: "astronomical-test/example-qwen",
            error_code: "checksum_mismatch"
        }`,
        scriptContext
    );
    assert.equal(
        vm.runInContext("libraryStateTitle(catalogRow)", scriptContext),
        "Downloaded files did not pass verification"
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(vm.runInContext(
            "libraryActionButtons(catalogRow).map(button => button.textContent)",
            scriptContext
        ))),
        ["Resume", "Cancel"]
    );
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

test("renders catalog entries with safe document operations and an explicit download action", () => {
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
    let renderedHeading = null;
    const pendingElements = [...libraryDocument.catalogContainer.children];
    while (pendingElements.length > 0) {
        const renderedElement = pendingElements.shift();
        renderedTags.push(renderedElement.tagName);
        if (renderedElement.tagName === "H3") renderedHeading = renderedElement;
        pendingElements.push(...renderedElement.children);
    }

    assert.equal(renderedState, "ready");
    assert.equal(libraryDocument.catalogStatus.textContent, "Model catalog loaded.");
    assert.equal(renderedTags.includes("BUTTON"), true);
    // The filter bar uses a search input; no raw text inputs or anchors appear.
    assert.equal(renderedTags.includes("A"), false);
    assert.equal(
        renderedHeading.textContent,
        "<img src=x onerror=alert(1)>"
    );
});

test("renders daemon-authored catalog families without a client allowlist", () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogDocument = {
        schema_version: 1,
        entries: [
            validCatalogEntry(),
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-laguna",
                family: "laguna"
            }),
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-flux",
                family: "flux2_klein"
            })
        ]
    };

    const renderedState = vm.runInContext(
        "renderLibraryCatalogDocument(catalogDocument)",
        scriptContext
    );
    const catalogRows = vm.runInContext(
        "libraryCatalogRowsFromDocument(catalogDocument)",
        scriptContext
    );

    assert.equal(renderedState, "ready");
    assert.deepEqual(
        JSON.parse(JSON.stringify(catalogRows.map((catalogRow) => catalogRow.family))),
        ["qwen3_5", "laguna", "flux2_klein"]
    );
    const filterBar = libraryDocument.catalogContainer.children[2];
    const familyFilter = filterBar.children[2];
    assert.deepEqual(
        familyFilter.children.map((option) => option.value),
        ["all", "qwen3_5", "laguna", "flux2_klein"]
    );
});

test("does not offer chat for a ready image-generation-only model", () => {
    const scriptContext = createLibraryContext();
    scriptContext.document = createLibraryDocument();
    scriptContext.catalogRow = {
        huggingfaceId: "astronomical-test/example-flux",
        readyOnThisMac: true,
        requestableModelId: "example-flux",
        supportsImageGeneration: true
    };

    const actionLabels = vm.runInContext(
        "createLibraryPrimaryActions(catalogRow, false).children.map(button => button.textContent)",
        scriptContext
    );

    assert.deepEqual(JSON.parse(JSON.stringify(actionLabels)), ["Details"]);
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

test("recovers the catalog automatically after a daemon interruption", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    let catalogAttemptCount = 0;
    scriptContext.document = libraryDocument;
    scriptContext.fetch = async (requestPath) => {
        if (requestPath === "/v1/library/catalog") {
            catalogAttemptCount += 1;
            if (catalogAttemptCount === 1) throw new Error("daemon restarting");
            return {
                ok: true,
                async json() {
                    return { schema_version: 1, entries: [validCatalogEntry()] };
                }
            };
        }
        return { ok: true, async json() { return { state: "downloading" }; } };
    };
    scriptContext.catalogContainer = libraryDocument.catalogContainer;

    await vm.runInContext("loadLibraryCatalog(catalogContainer, 10)", scriptContext);
    assert.equal(libraryDocument.catalogContainer.dataset.libraryState, "unavailable");

    await vm.runInContext("refreshLibraryDownloadState()", scriptContext);

    assert.equal(catalogAttemptCount, 2);
    assert.equal(libraryDocument.catalogContainer.dataset.libraryState, "ready");
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
