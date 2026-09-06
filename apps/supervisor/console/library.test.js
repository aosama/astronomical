// Focused Library rendering behavior tests keep the Observatory suite bounded.

const {
    ASYNC_TEST_OPTIONS,
    assert,
    createLibraryContext,
    createLibraryDocument,
    createLibraryElement,
    test,
    validCatalogEntry,
    vm
} = require("./library-test-support.js");

test("renders an empty catalog through the bounded empty state", () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogDocument = { schema_version: 2, entries: [] };

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
        schema_version: 2,
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
            supportsEmbeddings: false,
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
            supportsEmbeddings: false,
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
        vm.runInContext("smoothLibraryTransferRate(20_000_000, 4_000_000)", scriptContext),
        16_000_000
    );
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

    vm.runInContext(
        `libraryCurrentDownload = {
            state: "failed",
            huggingface_id: "astronomical-test/example-qwen",
            error_code: "model_already_present"
        }`,
        scriptContext
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(vm.runInContext(
            "libraryActionButtons(catalogRow).map(button => button.textContent)",
            scriptContext
        ))),
        ["Cancel"]
    );
    assert.match(
        vm.runInContext("libraryStateTitle(catalogRow)", scriptContext),
        /move it elsewhere/i
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
        schema_version: 2,
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
        schema_version: 2,
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

test("surfaces ModernBERT embedding entries with their family and capability badge", () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogDocument = {
        schema_version: 2,
        entries: [
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-embedder",
                family: "modernbert",
                capabilities: { supports_embeddings: true }
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
    assert.equal(catalogRows[0].supportsEmbeddings, true);
    const renderedTexts = [];
    const pendingElements = [...libraryDocument.catalogContainer.children];
    while (pendingElements.length > 0) {
        const renderedElement = pendingElements.shift();
        renderedTexts.push(renderedElement.textContent);
        pendingElements.push(...renderedElement.children);
    }
    assert.equal(
        renderedTexts.some((text) => text.includes("Embeddings")),
        true,
        "the embedding capability badge should be rendered"
    );
    assert.equal(
        renderedTexts.some((text) => text.includes("ModernBERT")),
        true,
        "the ModernBERT family label should be rendered"
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

