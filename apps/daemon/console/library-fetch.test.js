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
    scriptContext.fetch = async (requestPath) => {
        // Only the catalog request is counted; the memory status read is a separate, advisory request.
        if (requestPath === "/v1/library/catalog") catalogFetchCount += 1;
        return { ok: true, async json() { return { schema_version: 3, entries: [] }; } };
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
    malformedContext.catalogDocument = { schema_version: 3, entries: [{}] };
    assert.equal(
        vm.runInContext("renderLibraryCatalogDocument(catalogDocument)", malformedContext),
        "unavailable"
    );

    const nonSuccessContext = createLibraryContext();
    const nonSuccessDocument = createLibraryDocument();
    nonSuccessContext.document = nonSuccessDocument;
    nonSuccessContext.fetch = async () => ({
        ok: false,
        async json() { return { schema_version: 3, entries: [] }; }
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
                    return { schema_version: 3, entries: [validCatalogEntry()] };
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
                return { ok: true, async json() { return { schema_version: 3, entries: [] }; } };
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
    )
        // The chat render stack is the shared Thin Talk shell: heavy vendor
        // bundles (KaTeX, Mermaid) with browser dependencies the Library journey
        // never touches. It is verified by the Rust build (embedded assets) and
        // Thin Talk's own tests, so this focused Library test loads only the
        // console's own scripts.
        .filter((scriptSource) => !scriptSource.startsWith("/render/"));
    for (const scriptSource of scriptSources) {
        // Script sources are console-rooted URL paths; resolve them against the
        // console dir so the shipped file is found.
        const shippedScriptPath = path.join(__dirname, scriptSource.replace(/^\//, ""));
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

function catalogAndStatusFetch(statusHandler) {
    return async (requestPath) => {
        if (requestPath === "/v1/library/catalog") {
            return { ok: true, async json() { return { schema_version: 3, entries: [validCatalogEntry()] }; } };
        }
        return statusHandler(requestPath);
    };
}

test("should_read_the_serving_memory_ceiling_from_status_with_the_catalog", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogContainer = libraryDocument.catalogContainer;
    scriptContext.fetch = catalogAndStatusFetch(async () => ({
        ok: true,
        async json() {
            return {
                mlx_memory_ceiling_bytes: 34_359_738_368,
                machine_mlx_memory_ceiling_bytes: 42_949_672_960
            };
        }
    }));

    await vm.runInContext("loadLibraryCatalog(catalogContainer, 10_000)", scriptContext);

    assert.equal(vm.runInContext("libraryServingCeilingBytes", scriptContext), 34_359_738_368);
    assert.equal(libraryDocument.catalogContainer.dataset.libraryState, "ready");
});

test("should_use_the_effective_ceiling_and_fall_back_to_the_machine_ceiling_only_when_it_is_missing", () => {
    const scriptContext = createLibraryContext();
    scriptContext.statusDocument = {
        mlx_memory_ceiling_bytes: 34_359_738_368,
        machine_mlx_memory_ceiling_bytes: 42_949_672_960
    };
    assert.equal(
        vm.runInContext("readLibraryServingCeilingBytes(statusDocument)", scriptContext),
        34_359_738_368
    );
    for (const missingEffectiveCeiling of [undefined, 0, -1, "34359738368", Infinity]) {
        scriptContext.statusDocument = {
            mlx_memory_ceiling_bytes: missingEffectiveCeiling,
            machine_mlx_memory_ceiling_bytes: 42_949_672_960
        };
        assert.equal(
            vm.runInContext("readLibraryServingCeilingBytes(statusDocument)", scriptContext),
            42_949_672_960,
            `effective ceiling ${String(missingEffectiveCeiling)} must fall back to the machine ceiling`
        );
    }
});

test("should_return_null_serving_ceiling_when_both_status_ceilings_are_missing_or_zero", () => {
    const scriptContext = createLibraryContext();
    for (const statusDocument of [{}, { mlx_memory_ceiling_bytes: 0, machine_mlx_memory_ceiling_bytes: 0 }]) {
        scriptContext.statusDocument = statusDocument;
        assert.equal(
            vm.runInContext("readLibraryServingCeilingBytes(statusDocument)", scriptContext),
            null
        );
    }
});

test("should_treat_a_failed_or_zero_status_as_unknown_without_failing_the_catalog", ASYNC_TEST_OPTIONS, async () => {
    const zeroCeilingContext = createLibraryContext();
    const zeroCeilingDocument = createLibraryDocument();
    zeroCeilingContext.document = zeroCeilingDocument;
    zeroCeilingContext.catalogContainer = zeroCeilingDocument.catalogContainer;
    zeroCeilingContext.fetch = catalogAndStatusFetch(async () => ({
        ok: true,
        async json() { return { machine_mlx_memory_ceiling_bytes: 0 }; }
    }));
    await vm.runInContext("loadLibraryCatalog(catalogContainer, 10_000)", zeroCeilingContext);
    assert.equal(vm.runInContext("libraryServingCeilingBytes", zeroCeilingContext), null);

    const failedStatusContext = createLibraryContext();
    const failedStatusDocument = createLibraryDocument();
    failedStatusContext.document = failedStatusDocument;
    failedStatusContext.catalogContainer = failedStatusDocument.catalogContainer;
    failedStatusContext.fetch = catalogAndStatusFetch(async () => { throw new Error("status down"); });
    await vm.runInContext("loadLibraryCatalog(catalogContainer, 10_000)", failedStatusContext);
    assert.equal(vm.runInContext("libraryServingCeilingBytes", failedStatusContext), null);
    assert.equal(failedStatusDocument.catalogContainer.dataset.libraryState, "ready");
});

test("should_reread_the_serving_ceiling_only_once_it_is_stale_or_unknown", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    let statusReadCount = 0;
    let reportedCeilingBytes = 34_359_738_368;
    scriptContext.fetch = catalogAndStatusFetch(async () => {
        statusReadCount += 1;
        return { ok: true, async json() { return { mlx_memory_ceiling_bytes: reportedCeilingBytes }; } };
    });

    await vm.runInContext("refetchLibraryCatalogDocument(100_000)", scriptContext);
    assert.equal(statusReadCount, 1, "an unknown ceiling is read at once");

    reportedCeilingBytes = 17_179_869_184;
    await vm.runInContext("refetchLibraryCatalogDocument(129_999)", scriptContext);
    assert.equal(statusReadCount, 1, "one millisecond short of the refresh interval reuses the known ceiling");
    assert.equal(vm.runInContext("libraryServingCeilingBytes", scriptContext), 34_359_738_368);

    await vm.runInContext("refetchLibraryCatalogDocument(130_000)", scriptContext);
    assert.equal(statusReadCount, 2, "exactly the refresh interval re-reads the ceiling");
    assert.equal(vm.runInContext("libraryServingCeilingBytes", scriptContext), 17_179_869_184);
});

test("should_ask_the_daemon_to_reveal_a_model_by_identity_and_report_the_outcome", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    const revealRequests = [];
    let revealStatus = 204;
    scriptContext.fetch = async (requestPath, requestOptions) => {
        revealRequests.push({ requestPath, requestOptions });
        return { ok: revealStatus >= 200 && revealStatus < 300, status: revealStatus };
    };

    assert.equal(
        await vm.runInContext('revealLibraryModelInFinder("astronomical-test/example-qwen")', scriptContext),
        true
    );
    assert.equal(revealRequests[0].requestPath, "/v1/library/reveal");
    assert.equal(revealRequests[0].requestOptions.method, "POST");
    assert.deepEqual(
        JSON.parse(revealRequests[0].requestOptions.body),
        { huggingface_id: "astronomical-test/example-qwen" },
        "the page names the model and never sends a path"
    );

    revealStatus = 409;
    assert.equal(
        await vm.runInContext('revealLibraryModelInFinder("astronomical-test/example-qwen")', scriptContext),
        false,
        "a model that is not on this Mac is reported as not revealed"
    );
    scriptContext.fetch = async () => { throw new Error("daemon unreachable"); };
    assert.equal(
        await vm.runInContext('revealLibraryModelInFinder("astronomical-test/example-qwen")', scriptContext),
        false
    );
});

test("should_offer_show_in_finder_only_for_a_model_already_on_this_mac", () => {
    const scriptContext = createLibraryContext();
    scriptContext.document = createLibraryDocument();
    const buttonClassesFor = (readyOnThisMac) => {
        scriptContext.catalogRow = { readyOnThisMac, huggingfaceId: "astronomical-test/example-qwen" };
        const locationSection = vm.runInContext(
            'createLibraryLocationSection("/models/example-qwen", catalogRow)',
            scriptContext
        );
        const actionRow = locationSection.children[2];
        return actionRow.children.map((button) => button.className);
    };
    assert.deepEqual(buttonClassesFor(true), [
        "button button-secondary library-reveal-button",
        "button button-secondary library-copy-path-button"
    ]);
    assert.deepEqual(
        buttonClassesFor(false),
        ["button button-secondary library-copy-path-button"],
        "a model still to be downloaded has no folder for Finder to show"
    );
});

test("should_say_finder_could_not_open_when_the_daemon_refuses_the_reveal", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    scriptContext.document = createLibraryDocument();
    scriptContext.fetch = async () => ({ ok: false, status: 409 });
    const revealButton = vm.runInContext(
        'createLibraryRevealButton("astronomical-test/example-qwen")',
        scriptContext
    );
    assert.equal(revealButton.textContent, "Show in Finder");

    await revealButton.listeners.click[0]();

    assert.equal(revealButton.textContent, "Could not open Finder");
});

test("should_keep_the_show_in_finder_label_when_finder_opens", ASYNC_TEST_OPTIONS, async () => {
    const scriptContext = createLibraryContext();
    scriptContext.document = createLibraryDocument();
    scriptContext.fetch = async () => ({ ok: true, status: 204 });
    const revealButton = vm.runInContext(
        'createLibraryRevealButton("astronomical-test/example-qwen")',
        scriptContext
    );

    await revealButton.listeners.click[0]();

    assert.equal(revealButton.textContent, "Show in Finder");
});
