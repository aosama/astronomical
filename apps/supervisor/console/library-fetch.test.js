// Library Observatory catalog-fetch and wiring journey tests.

const {
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
} = require("./library-test-support.js");

test("fetches the immutable catalog once when Library wiring is repeated", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    let catalogFetchCount = 0;
    scriptContext.document = libraryDocument;
    scriptContext.fetch = async () => {
        catalogFetchCount += 1;
        return { ok: true, async json() { return { schema_version: 2, entries: [] }; } };
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
    malformedContext.catalogDocument = { schema_version: 2, entries: [{}] };
    assert.equal(
        vm.runInContext("renderLibraryCatalogDocument(catalogDocument)", malformedContext),
        "unavailable"
    );

    const nonSuccessContext = createLibraryContext();
    const nonSuccessDocument = createLibraryDocument();
    nonSuccessContext.document = nonSuccessDocument;
    nonSuccessContext.fetch = async () => ({
        ok: false,
        async json() { return { schema_version: 2, entries: [] }; }
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
                    return { schema_version: 2, entries: [validCatalogEntry()] };
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
                return { ok: true, async json() { return { schema_version: 2, entries: [] }; } };
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
