const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const { createConsoleContext, attachRecordingElements } = require("./console-test-support.js");

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
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("download-models")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("download-models")];
    scriptContext.navigationButtons = navigationButtons;
    scriptContext.observatoryViews = observatoryViews;

    const activeViewIdentifier = vm.runInContext(
        'activateObservatoryView("download-models", navigationButtons, observatoryViews)',
        scriptContext
    );

    assert.equal(activeViewIdentifier, "download-models");
    assert.equal(navigationButtons[0].attributes["aria-current"], undefined);
    assert.equal(navigationButtons[1].attributes["aria-current"], "page");
    assert.equal(observatoryViews[0].hidden, true);
    assert.equal(observatoryViews[1].hidden, false);
});

test("activates a directly requested Observatory path without replacing it", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("download-models")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("download-models")];
    const replacedPaths = [];
    scriptContext.document = {
        querySelectorAll(selector) {
            return selector === "[data-observatory-destination]" ? navigationButtons : observatoryViews;
        }
    };
    scriptContext.window = {
        location: { pathname: "/download-models" },
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

test("should_open_overview_for_the_retired_model_path_and_show_its_address", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("download-models")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("download-models")];
    const replacedPaths = [];
    scriptContext.document = {
        querySelectorAll(selector) {
            return selector === "[data-observatory-destination]" ? navigationButtons : observatoryViews;
        }
    };
    scriptContext.window = { location: { pathname: "/model" }, addEventListener() {} };
    scriptContext.history = {
        pushState() {},
        replaceState(unusedState, unusedTitle, observatoryPath) {
            replacedPaths.push(observatoryPath);
        }
    };

    vm.runInContext("wireObservatoryNavigation()", scriptContext);

    assert.equal(observatoryViews[0].hidden, false, "the model facts now live on Overview");
    assert.equal(navigationButtons[0].attributes["aria-current"], "page");
    assert.deepEqual(replacedPaths, ["/overview"]);
});

test("should_open_download_models_for_the_retired_library_path_and_show_its_address", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("download-models")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("download-models")];
    const replacedPaths = [];
    scriptContext.document = {
        querySelectorAll(selector) {
            return selector === "[data-observatory-destination]" ? navigationButtons : observatoryViews;
        }
    };
    scriptContext.window = { location: { pathname: "/library" }, addEventListener() {} };
    scriptContext.history = {
        pushState() {},
        replaceState(unusedState, unusedTitle, observatoryPath) {
            replacedPaths.push(observatoryPath);
        }
    };

    vm.runInContext("wireObservatoryNavigation()", scriptContext);

    assert.equal(observatoryViews[1].hidden, false);
    assert.equal(navigationButtons[1].attributes["aria-current"], "page");
    assert.deepEqual(replacedPaths, ["/download-models"]);
});

test("should_open_configuration_for_the_retired_settings_path_and_show_its_address", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("configuration")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("configuration")];
    const replacedPaths = [];
    scriptContext.document = {
        querySelectorAll(selector) {
            return selector === "[data-observatory-destination]" ? navigationButtons : observatoryViews;
        }
    };
    scriptContext.window = { location: { pathname: "/settings" }, addEventListener() {} };
    scriptContext.history = {
        pushState() {},
        replaceState(unusedState, unusedTitle, observatoryPath) {
            replacedPaths.push(observatoryPath);
        }
    };

    vm.runInContext("wireObservatoryNavigation()", scriptContext);

    assert.equal(observatoryViews[1].hidden, false, "the memory limit now lives on Configuration");
    assert.equal(navigationButtons[1].attributes["aria-current"], "page");
    assert.deepEqual(replacedPaths, ["/configuration"]);
});

test("should_retain_direct_download_models_navigation_and_restore_it_from_browser_history", () => {
    const scriptContext = createConsoleContext();
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("download-models")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("download-models")];
    let popstateHandler;
    scriptContext.document = {
        querySelectorAll(selector) {
            return selector === "[data-observatory-destination]" ? navigationButtons : observatoryViews;
        }
    };
    scriptContext.window = {
        location: { pathname: "/download-models" },
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
    scriptContext.window.location.pathname = "/download-models";
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

test("should_show_a_reveal_failure_inside_the_discovery_warning_that_holds_the_button", async () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    scriptContext.fetch = async () => ({ ok: false });

    await vm.runInContext("revealConfig()", scriptContext);

    assert.equal(elementsById["model-discovery-reveal-feedback"].textContent, "Could not reveal the configuration file.");
});

test("should_clear_an_earlier_reveal_failure_when_the_reveal_succeeds", async () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    elementsById["model-discovery-reveal-feedback"] = { textContent: "Could not reveal the configuration file." };
    scriptContext.fetch = async () => ({ ok: true });

    await vm.runInContext("revealConfig()", scriptContext);

    assert.equal(elementsById["model-discovery-reveal-feedback"].textContent, "");
});

test("maps every observatory destination to a stable URL path", () => {
    const scriptContext = createConsoleContext();

    const pathMap = vm.runInContext("OBSERVATORY_PATH_MAP", scriptContext);

    assert.equal(pathMap.overview, "/overview");
    assert.equal(pathMap.chat, "/chat");
    assert.equal(pathMap["download-models"], "/download-models");
    assert.equal(pathMap.library, undefined, "the Library page was renamed Download Models");
    assert.equal(pathMap.connect, "/connect");
    assert.equal(pathMap.memory, undefined);
    assert.equal(pathMap.cache, undefined);
    assert.equal(pathMap.model, undefined, "the Model page was folded into Overview");
    assert.equal(pathMap.configuration, "/configuration");
    assert.equal(pathMap.settings, undefined, "the Settings page was folded into Configuration");
});

test("maps every observatory URL path back to its destination", () => {
    const scriptContext = createConsoleContext();

    const reverseMap = vm.runInContext("OBSERVATORY_PATH_TO_DESTINATION_MAP", scriptContext);

    assert.equal(reverseMap["/overview"], "overview");
    assert.equal(reverseMap["/chat"], "chat");
    assert.equal(reverseMap["/download-models"], "download-models");
    assert.equal(reverseMap["/library"], undefined, "the retired path is resolved separately");
    assert.equal(reverseMap["/connect"], "connect");
    assert.equal(reverseMap["/memory"], undefined);
    assert.equal(reverseMap["/cache"], undefined);
    assert.equal(reverseMap["/model"], undefined, "the retired path is resolved separately");
    assert.equal(reverseMap["/configuration"], "configuration");
    assert.equal(reverseMap["/settings"], undefined, "the retired path is resolved separately");
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

test("should_keep_every_chat_link_out_of_download_models_and_the_shell", () => {
    const downloadModelsRenderScript = fs.readFileSync(path.join(__dirname, "download-models-render.js"), "utf8");
    const downloadModelsCardFooterScript = fs.readFileSync(path.join(__dirname, "download-models-card-footer.js"), "utf8");
    const consoleShell = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
    for (const hardcodingSource of [downloadModelsRenderScript, downloadModelsCardFooterScript, consoleShell]) {
        assert.equal(hardcodingSource.includes("astronomical://"), false);
        assert.equal(hardcodingSource.includes("astronomical-dev://"), false);
    }
});

/** A document whose elements are created on first lookup and keep their text. */
