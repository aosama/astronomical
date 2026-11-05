const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const consoleScriptPath = path.join(__dirname, "console.js");
const consoleScript = fs.readFileSync(consoleScriptPath, "utf8");
const memoryControlScriptPath = path.join(__dirname, "memory-control.js");
const memoryControlScript = fs.readFileSync(memoryControlScriptPath, "utf8");
const overviewCompactScriptPath = path.join(__dirname, "overview-compact.js");
const overviewCompactScript = fs.readFileSync(overviewCompactScriptPath, "utf8");

function createConsoleContext() {
    const scriptContext = vm.createContext({
        console: { log() {} },
        document: { addEventListener() {} },
        setInterval() {},
        history: {
            pushState() {},
            replaceState() {},
            state: null
        },
        TextEncoder
    });
    vm.runInContext(memoryControlScript, scriptContext, { filename: memoryControlScriptPath });
    vm.runInContext(consoleScript, scriptContext, { filename: consoleScriptPath });
    vm.runInContext(overviewCompactScript, scriptContext, { filename: overviewCompactScriptPath });
    return scriptContext;
}

test("formats exact Stable and dirty Development build identities from status", () => {
    const scriptContext = createConsoleContext();

    assert.equal(
        scriptContext.applicationIdentityTitle({
            version: "0.2.0",
            channel_display_name: "Stable",
            commit: "abcdef1",
            is_dirty: false
        }),
        "0.2.0 · Stable · abcdef1"
    );
    assert.equal(
        scriptContext.applicationIdentityTitle({
            version: "0.2.0",
            channel_display_name: "Development",
            commit: "1234567",
            is_dirty: true
        }),
        "0.2.0 · Development · 1234567-dirty"
    );
});

function createNavigationButton(observatoryDestination) {
    return {
        dataset: { observatoryDestination },
        attributes: {},
        addEventListener() {},
        setAttribute(attributeName, attributeText) { this.attributes[attributeName] = attributeText; },
        removeAttribute(attributeName) { delete this.attributes[attributeName]; }
    };
}

function createObservatoryView(observatoryView) {
    return { dataset: { observatoryView }, hidden: false };
}

test("defaults Observatory navigation to Overview", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("chat")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("chat")];
    scriptContext.navigationButtons = navigationButtons;
    scriptContext.observatoryViews = observatoryViews;

    const activeViewIdentifier = vm.runInContext(
        "activateObservatoryView(null, navigationButtons, observatoryViews)",
        scriptContext
    );

    assert.equal(activeViewIdentifier, "overview");
    assert.equal(navigationButtons[0].attributes["aria-current"], "page");
    assert.equal(navigationButtons[1].attributes["aria-current"], undefined);
    assert.equal(observatoryViews[0].hidden, false);
    assert.equal(observatoryViews[1].hidden, true);
});

test("moves Observatory visibility and current state to a selected destination", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("model")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("model")];
    scriptContext.navigationButtons = navigationButtons;
    scriptContext.observatoryViews = observatoryViews;

    const activeViewIdentifier = vm.runInContext(
        'activateObservatoryView("model", navigationButtons, observatoryViews)',
        scriptContext
    );

    assert.equal(activeViewIdentifier, "model");
    assert.equal(navigationButtons[0].attributes["aria-current"], undefined);
    assert.equal(navigationButtons[1].attributes["aria-current"], "page");
    assert.equal(observatoryViews[0].hidden, true);
    assert.equal(observatoryViews[1].hidden, false);
});

test("activates a directly requested Observatory path without replacing it", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("model")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("model")];
    const replacedPaths = [];
    scriptContext.document = {
        querySelectorAll(selector) {
            return selector === "[data-observatory-destination]" ? navigationButtons : observatoryViews;
        }
    };
    scriptContext.window = {
        location: { pathname: "/model" },
        addEventListener() {}
    };
    scriptContext.history = {
        pushState() {},
        replaceState(unusedState, unusedTitle, observatoryPath) {
            replacedPaths.push(observatoryPath);
        }
    };

    vm.runInContext("wireObservatoryNavigation()", scriptContext);

    assert.equal(navigationButtons[1].attributes["aria-current"], "page");
    assert.equal(observatoryViews[1].hidden, false);
    assert.deepEqual(replacedPaths, []);
});

test("retains direct Library navigation and restores it from browser history", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("library")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("library")];
    let popstateHandler;
    scriptContext.document = {
        querySelectorAll(selector) {
            return selector === "[data-observatory-destination]" ? navigationButtons : observatoryViews;
        }
    };
    scriptContext.window = {
        location: { pathname: "/library" },
        addEventListener(eventName, eventHandler) {
            if (eventName === "popstate") { popstateHandler = eventHandler; }
        }
    };
    scriptContext.history = { pushState() {}, replaceState() {} };

    vm.runInContext("wireObservatoryNavigation()", scriptContext);
    assert.equal(navigationButtons[1].attributes["aria-current"], "page");
    assert.equal(observatoryViews[1].hidden, false);

    scriptContext.window.location.pathname = "/overview";
    popstateHandler();
    scriptContext.window.location.pathname = "/library";
    popstateHandler();

    assert.equal(navigationButtons[1].attributes["aria-current"], "page");
    assert.equal(observatoryViews[1].hidden, false);
});

test("selects the ready model metadata instead of the first advertised model", () => {
    const scriptContext = createConsoleContext();
    const selectedModel = vm.runInContext(
        'selectAdvertisedModel([{ id: "first" }, { id: "ready" }], "ready")',
        scriptContext
    );

    assert.equal(selectedModel.id, "ready");
});

test("detects configured and effective generation mismatch", () => {
    const scriptContext = createConsoleContext();

    assert.equal(
        scriptContext.configurationRequiresRestart({
            configured_generation: "configured",
            effective_generation: "effective",
            restart_required: false
        }),
        true
    );
    assert.equal(
        scriptContext.configurationRequiresRestart({
            configured_generation: "same",
            effective_generation: "same",
            restart_required: false
        }),
        false
    );
});

test("renders a path-safe duplicate-model warning with its reveal control", () => {
    const scriptContext = createConsoleContext();
    const warningElement = { hidden: true };
    const messageElement = { textContent: "" };
    scriptContext.document.getElementById = (elementId) => ({
        "model-discovery-warning": warningElement,
        "model-discovery-warning-message": messageElement
    })[elementId];

    scriptContext.renderModelDiscoveryDiagnostic({
        model_discovery_diagnostics: [{
            code: "ambiguous_model_identity",
            model_id: "shared-model",
            configured_root_numbers: [1, 2]
        }]
    });

    assert.equal(warningElement.hidden, false);
    assert.equal(
        messageElement.textContent,
        "Model shared-model appears in model_directories entries 1, 2. Remove one duplicate root."
    );
    assert.equal(messageElement.textContent.includes("private-root-marker"), false);
});

test("computes whole gigabyte bounds from exact status bytes", () => {
    const scriptContext = createConsoleContext();
    const bounds = vm.runInContext(
        "wholeGigabyteBounds({ minimum_mlx_memory_ceiling_bytes: 34359738369, machine_mlx_memory_ceiling_bytes: 44080218931 })",
        scriptContext
    );
    assert.deepEqual(JSON.parse(JSON.stringify(bounds)), { minimumGigabytes: 1, maximumGigabytes: 41 });
});

function renderMemoryControlFor(statusDocument, editedInputValue) {
    const scriptContext = createConsoleContext();
    const elementsById = {
        "maximum-mlx-memory-gb": { min: "", max: "", disabled: false, value: "" },
        "maximum-mlx-memory-apply": { hidden: false },
        "maximum-mlx-memory-reset": { hidden: false },
        "maximum-mlx-memory-status": { textContent: "" },
        "maximum-mlx-memory-notice": { textContent: "", hidden: false }
    };
    scriptContext.document.getElementById = (elementId) => elementsById[elementId];
    if (editedInputValue !== undefined) {
        vm.runInContext("maximumMlxMemoryEditIsActive = true", scriptContext);
        elementsById["maximum-mlx-memory-gb"].value = editedInputValue;
    }
    scriptContext.renderMemoryLimitControl(statusDocument);
    return elementsById;
}

// A 48 GB Mac whose 40,200,896,512-byte wired claim the engine plans 2.5%
// below: 39,195,874,112 bytes (36.50 GB).
const WIRED_MAC_STATUS = {
    machine_mlx_memory_ceiling_bytes: 42949672960,
    wired_residency_budget_bytes: 40200896512
};

test("shows only the reachable limit while an explicit request is served", () => {
    const elementsById = renderMemoryControlFor({
        ...WIRED_MAC_STATUS,
        mlx_memory_ceiling_bytes: 36507222016,
        configured_maximum_mlx_memory_gb: 34
    });

    assert.equal(elementsById["maximum-mlx-memory-gb"].max, "36");
    assert.equal(elementsById["maximum-mlx-memory-gb"].value, "34");
    assert.equal(elementsById["maximum-mlx-memory-status"].textContent, "Up to 36 GB on this Mac");
    assert.equal(elementsById["maximum-mlx-memory-apply"].hidden, true);
    assert.equal(elementsById["maximum-mlx-memory-reset"].hidden, false);
    assert.equal(elementsById["maximum-mlx-memory-notice"].hidden, true);
});

test("offers apply only once the input differs from the applied value", () => {
    const elementsById = renderMemoryControlFor({
        ...WIRED_MAC_STATUS,
        mlx_memory_ceiling_bytes: 36507222016,
        configured_maximum_mlx_memory_gb: 34
    }, "30");

    assert.equal(elementsById["maximum-mlx-memory-apply"].hidden, false);
});

test("treats the Mac maximum landing below the machine ceiling as honoured, not capped", () => {
    const elementsById = renderMemoryControlFor({
        ...WIRED_MAC_STATUS,
        mlx_memory_ceiling_bytes: 39195874112,
        configured_maximum_mlx_memory_gb: null
    });

    assert.equal(elementsById["maximum-mlx-memory-gb"].value, "36");
    assert.equal(elementsById["maximum-mlx-memory-reset"].hidden, true);
    assert.equal(elementsById["maximum-mlx-memory-notice"].hidden, true);
});

test("reports a cap when an explicit request exceeds the live ceiling", () => {
    const elementsById = renderMemoryControlFor({
        ...WIRED_MAC_STATUS,
        mlx_memory_ceiling_bytes: 39195874112,
        configured_maximum_mlx_memory_gb: 40
    });

    assert.equal(elementsById["maximum-mlx-memory-gb"].value, "36");
    assert.equal(elementsById["maximum-mlx-memory-notice"].hidden, false);
    assert.equal(elementsById["maximum-mlx-memory-notice"].textContent, "Capped at 36.50 GB on this Mac");
    assert.equal(elementsById["maximum-mlx-memory-apply"].hidden, false);
});

test("prefers a pending change and then an error over the cap notice", () => {
    const pendingElements = renderMemoryControlFor({
        ...WIRED_MAC_STATUS,
        mlx_memory_ceiling_bytes: 32212254720,
        configured_maximum_mlx_memory_gb: 34,
        pending_mlx_memory_ceiling_bytes: 36507222016
    });
    const failedElements = renderMemoryControlFor({
        ...WIRED_MAC_STATUS,
        mlx_memory_ceiling_bytes: 32212254720,
        configured_maximum_mlx_memory_gb: 34,
        pending_mlx_memory_ceiling_bytes: 36507222016,
        mlx_memory_limit_error: "The worker could not apply the requested MLX memory ceiling"
    });

    assert.equal(pendingElements["maximum-mlx-memory-notice"].textContent, "Pending 34.00 GB");
    assert.equal(
        failedElements["maximum-mlx-memory-notice"].textContent,
        "The worker could not apply the requested MLX memory ceiling"
    );
});

test("reaches the machine ceiling when serving is pageable and stays quiet before a ceiling exists", () => {
    const elementsById = renderMemoryControlFor({
        machine_mlx_memory_ceiling_bytes: 42949672960,
        mlx_memory_ceiling_bytes: 0,
        configured_maximum_mlx_memory_gb: 34
    });

    assert.equal(elementsById["maximum-mlx-memory-gb"].max, "40");
    assert.equal(elementsById["maximum-mlx-memory-notice"].hidden, true);
});

test("maps every observatory destination to a stable URL path", () => {
    const scriptContext = createConsoleContext();

    const pathMap = vm.runInContext("OBSERVATORY_PATH_MAP", scriptContext);

    assert.equal(pathMap.overview, "/overview");
    assert.equal(pathMap.chat, "/chat");
    assert.equal(pathMap.library, "/library");
    assert.equal(pathMap.connect, "/connect");
    assert.equal(pathMap.memory, undefined);
    assert.equal(pathMap.cache, undefined);
    assert.equal(pathMap.model, "/model");
    assert.equal(pathMap.settings, "/settings");
});

test("maps every observatory URL path back to its destination", () => {
    const scriptContext = createConsoleContext();

    const reverseMap = vm.runInContext("OBSERVATORY_PATH_TO_DESTINATION_MAP", scriptContext);

    assert.equal(reverseMap["/overview"], "overview");
    assert.equal(reverseMap["/chat"], "chat");
    assert.equal(reverseMap["/library"], "library");
    assert.equal(reverseMap["/connect"], "connect");
    assert.equal(reverseMap["/memory"], undefined);
    assert.equal(reverseMap["/cache"], undefined);
    assert.equal(reverseMap["/model"], "model");
    assert.equal(reverseMap["/settings"], "settings");
});

test("defaults observatory to the overview destination path", () => {
    const scriptContext = createConsoleContext();

    const defaultDestination = vm.runInContext("OBSERVATORY_DEFAULT_DESTINATION", scriptContext);
    const defaultPath = vm.runInContext("OBSERVATORY_DEFAULT_PATH", scriptContext);

    assert.equal(defaultDestination, "overview");
    assert.equal(defaultPath, "/overview");
});

test("clamps memory segments so they never exceed active memory", () => {
    const scriptContext = createConsoleContext();
    scriptContext.mlxMemorySnapshot = {
        active_memory_bytes: 100,
        expert_payload_bytes: 80,
        model_core_payload_bytes: 40,
        context_state_payload_bytes: 10
    };
    scriptContext.mlxMemoryCeilingBytes = 200;

    const segmentBytes = vm.runInContext(
        "reconciledMlxMemorySegmentBytes(mlxMemorySnapshot, mlxMemoryCeilingBytes)",
        scriptContext
    );

    assert.deepEqual(
        JSON.parse(JSON.stringify(segmentBytes)),
        {
            activeMemoryBytes: 100,
            expertBytes: 80,
            modelCoreBytes: 20,
            contextStateBytes: 0,
            runtimeWorkBytes: 0,
            availableBytes: 100
        }
    );
});

test("computes available bytes as ceiling minus active when nothing is clamped", () => {
    const scriptContext = createConsoleContext();
    scriptContext.mlxMemorySnapshot = {
        active_memory_bytes: 40,
        expert_payload_bytes: 30,
        model_core_payload_bytes: 5,
        context_state_payload_bytes: 2
    };
    scriptContext.mlxMemoryCeilingBytes = 200;

    const segmentBytes = vm.runInContext(
        "reconciledMlxMemorySegmentBytes(mlxMemorySnapshot, mlxMemoryCeilingBytes)",
        scriptContext
    );

    assert.deepEqual(
        JSON.parse(JSON.stringify(segmentBytes)),
        {
            activeMemoryBytes: 40,
            expertBytes: 30,
            modelCoreBytes: 5,
            contextStateBytes: 2,
            runtimeWorkBytes: 3,
            availableBytes: 160
        }
    );
});

test("reports a null memory snapshot as all zero segments with full available headroom", () => {
    const scriptContext = createConsoleContext();

    const segmentBytes = vm.runInContext(
        "reconciledMlxMemorySegmentBytes(null, 200)",
        scriptContext
    );

    assert.deepEqual(
        JSON.parse(JSON.stringify(segmentBytes)),
        {
            activeMemoryBytes: 0,
            expertBytes: 0,
            modelCoreBytes: 0,
            contextStateBytes: 0,
            runtimeWorkBytes: 0,
            availableBytes: 200
        }
    );
});

test("renders shared prompt-cache statistics", () => {
    const scriptContext = createConsoleContext();
    const cachePanelElements = new Map([
        ["compact-cache-hit-rate", { textContent: "" }],
        ["compact-cache-tokens-saved", { textContent: "" }],
        ["compact-cache-disk-fill", { style: { width: "" } }],
        ["compact-cache-disk-label", { textContent: "" }]
    ]);
    scriptContext.document = {
        getElementById(elementIdentifier) {
            return cachePanelElements.get(elementIdentifier);
        }
    };

    vm.runInContext(
        `renderCompactCachePanel({
            persistent_prompt_cache_hit_rate: 0.25,
            persistent_prompt_cache_tokens_saved: 12,
            persistent_prompt_cache_total_size_bytes: 500000000,
            persistent_prompt_cache_maximum_size_bytes: 1000000000
        })`,
        scriptContext
    );

    assert.equal(cachePanelElements.get("compact-cache-hit-rate").textContent, "25.0%");
    assert.equal(cachePanelElements.get("compact-cache-tokens-saved").textContent, "12 prompt tokens reused");
    assert.equal(cachePanelElements.get("compact-cache-disk-fill").style.width, "50.0%");
    assert.equal(
        cachePanelElements.get("compact-cache-disk-label").textContent,
        "0.47 GB / 0.93 GB"
    );
});

test("attributes unowned active memory to runtime work", () => {
    const scriptContext = createConsoleContext();
    scriptContext.mlxMemorySnapshot = {
        active_memory_bytes: 100,
        expert_payload_bytes: 20,
        model_core_payload_bytes: 30,
        context_state_payload_bytes: 10
    };
    scriptContext.mlxMemoryCeilingBytes = 200;

    const segmentBytes = vm.runInContext(
        "reconciledMlxMemorySegmentBytes(mlxMemorySnapshot, mlxMemoryCeilingBytes)",
        scriptContext
    );

    assert.deepEqual(
        JSON.parse(JSON.stringify(segmentBytes)),
        {
            activeMemoryBytes: 100,
            expertBytes: 20,
            modelCoreBytes: 30,
            contextStateBytes: 10,
            runtimeWorkBytes: 40,
            availableBytes: 100
        }
    );
});

test("maps every macOS memory pressure state and unknown input to a safe presentation", () => {
    const scriptContext = createConsoleContext();
    const memoryPressureStatePresentations = vm.runInContext(
        `["normal", "warning", "critical", null, "unexpected"].map(
            (memoryPressureState) => memoryPressurePresentationForState(memoryPressureState)
        )`,
        scriptContext
    );

    assert.deepEqual(JSON.parse(JSON.stringify(memoryPressureStatePresentations)), [
        { state: "normal", title: "Normal" },
        { state: "warning", title: "Warning" },
        { state: "critical", title: "Critical" },
        { state: "unavailable", title: "Unavailable" },
        { state: "unavailable", title: "Unavailable" }
    ]);
});

function nowStripContext() {
    const scriptContext = createConsoleContext();
    const elementsById = new Map();
    scriptContext.document.getElementById = function (elementId) {
        if (!elementsById.has(elementId)) {
            const classNames = new Set();
            elementsById.set(elementId, {
                textContent: "",
                style: {},
                classNames,
                classList: {
                    add(className) { classNames.add(className); },
                    remove(className) { classNames.delete(className); },
                },
            });
        }
        return elementsById.get(elementId);
    };
    return { scriptContext, elementsById };
}

function renderNowStripWith(scriptContext, statusPayload) {
    scriptContext.__statusPayload = statusPayload;
    vm.runInContext("renderNowStrip(__statusPayload)", scriptContext);
}

test("should_mark_the_phase_live_while_generating", () => {
    const { scriptContext, elementsById } = nowStripContext();

    renderNowStripWith(scriptContext, { progress: { phase: "generation", processed_tokens: 5, total_tokens: 10, elapsed_ms: 100 } });

    assert.equal(elementsById.get("phase-label").textContent, "Generating");
    assert.equal(elementsById.get("phase-label").classNames.has("phase-live"), true);
});

test("should_mark_an_unnamed_phase_live_because_work_is_still_running", () => {
    const { scriptContext, elementsById } = nowStripContext();

    renderNowStripWith(scriptContext, { progress: { processed_tokens: 0, total_tokens: 0, elapsed_ms: 0 } });

    assert.equal(elementsById.get("phase-label").textContent, "Active");
    assert.equal(elementsById.get("phase-label").classNames.has("phase-live"), true);
});

test("should_stop_marking_the_phase_live_when_the_request_ends", () => {
    const { scriptContext, elementsById } = nowStripContext();
    renderNowStripWith(scriptContext, { progress: { phase: "target", processed_tokens: 1, total_tokens: 2, elapsed_ms: 1 } });

    renderNowStripWith(scriptContext, {});

    assert.equal(elementsById.get("phase-label").textContent, "Idle");
    assert.equal(elementsById.get("phase-label").classNames.has("phase-live"), false);
});

test("should_show_a_readable_label_for_every_phase_the_server_reports", () => {
    const expectedLabels = [
        ["target", "Prefilling"],
        ["generation_preparation", "Preparing output"],
        ["generation", "Generating"],
        ["preparing", "Preparing"],
        ["encoding_prompt", "Encoding prompt"],
        ["denoising", "Denoising"],
        ["decoding", "Decoding"],
        ["encoding_image", "Encoding image"],
    ];
    for (const [phase, expectedLabel] of expectedLabels) {
        const { scriptContext, elementsById } = nowStripContext();

        renderNowStripWith(scriptContext, { progress: { phase, processed_tokens: 0, total_tokens: 1, elapsed_ms: 1 } });

        assert.equal(elementsById.get("phase-label").textContent, expectedLabel, phase);
    }
});

test("should_never_show_an_unknown_phase_name_on_the_page", () => {
    for (const unknownPhase of ["speculative_draft", "constructor"]) {
        const { scriptContext, elementsById } = nowStripContext();

        renderNowStripWith(scriptContext, { progress: { phase: unknownPhase, processed_tokens: 0, total_tokens: 1, elapsed_ms: 1 } });

        assert.equal(elementsById.get("phase-label").textContent, "Active", unknownPhase);
    }
});

test("should_drop_the_generating_style_when_the_next_prompt_starts", () => {
    const { scriptContext, elementsById } = nowStripContext();
    renderNowStripWith(scriptContext, { progress: { phase: "generation", processed_tokens: 1, total_tokens: 2, elapsed_ms: 1 } });
    assert.equal(elementsById.get("phase-fill").classNames.has("generating"), true);

    renderNowStripWith(scriptContext, { progress: { phase: "target", processed_tokens: 1, total_tokens: 2, elapsed_ms: 1 } });

    assert.equal(elementsById.get("phase-fill").classNames.has("generating"), false);
});

function thinTalkLinkContext() {
    const scriptContext = createConsoleContext();
    const linkAttributes = new Map();
    scriptContext.document.getElementById = function () {
        return {
            setAttribute(attributeName, attributeValue) { linkAttributes.set(attributeName, attributeValue); },
            removeAttribute(attributeName) { linkAttributes.delete(attributeName); },
        };
    };
    return { scriptContext, linkAttributes };
}

test("should_open_thin_talk_through_the_development_scheme_in_development", () => {
    const { scriptContext, linkAttributes } = thinTalkLinkContext();

    vm.runInContext('renderThinTalkLink("development")', scriptContext);

    assert.equal(linkAttributes.get("href"), "astronomical-dev://chat");
});

test("should_open_thin_talk_through_the_stable_scheme_in_stable", () => {
    const { scriptContext, linkAttributes } = thinTalkLinkContext();

    vm.runInContext('renderThinTalkLink("stable")', scriptContext);

    assert.equal(linkAttributes.get("href"), "astronomical://chat");
});

test("should_offer_no_thin_talk_link_for_an_unknown_channel", () => {
    for (const unknownChannel of ["unknown", "", "constructor"]) {
        const { scriptContext, linkAttributes } = thinTalkLinkContext();
        linkAttributes.set("href", "astronomical://chat");

        vm.runInContext(`renderThinTalkLink("${unknownChannel}")`, scriptContext);

        assert.equal(linkAttributes.has("href"), false, unknownChannel);
    }
});

test("should_keep_every_chat_link_out_of_the_library_and_the_shell", () => {
    const libraryScript = fs.readFileSync(path.join(__dirname, "library-render.js"), "utf8");
    const consoleShell = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
    for (const hardcodingSource of [libraryScript, consoleShell]) {
        assert.equal(hardcodingSource.includes("astronomical://"), false);
        assert.equal(hardcodingSource.includes("astronomical-dev://"), false);
    }
});
