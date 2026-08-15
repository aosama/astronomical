const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const consoleScriptPath = path.join(__dirname, "console.js");
const consoleScript = fs.readFileSync(consoleScriptPath, "utf8");
const memoryControlScriptPath = path.join(__dirname, "memory-control.js");
const memoryControlScript = fs.readFileSync(memoryControlScriptPath, "utf8");
const playgroundScriptPath = path.join(__dirname, "playground.js");
const playgroundScript = fs.readFileSync(playgroundScriptPath, "utf8");
const optimizerScriptPath = path.join(__dirname, "optimizer.js");
const optimizerScript = fs.readFileSync(optimizerScriptPath, "utf8");
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
    vm.runInContext(optimizerScript, scriptContext, { filename: optimizerScriptPath });
    vm.runInContext(consoleScript, scriptContext, { filename: consoleScriptPath });
    vm.runInContext(overviewCompactScript, scriptContext, { filename: overviewCompactScriptPath });
    vm.runInContext(playgroundScript, scriptContext, { filename: playgroundScriptPath });
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
    const navigationButtons = [createNavigationButton("overview"), createNavigationButton("optimizer")];
    const observatoryViews = [createObservatoryView("overview"), createObservatoryView("optimizer")];
    const replacedPaths = [];
    scriptContext.document = {
        querySelectorAll(selector) {
            return selector === "[data-observatory-destination]" ? navigationButtons : observatoryViews;
        }
    };
    scriptContext.window = {
        location: { pathname: "/optimizer" },
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

test("scopes complete candidate measurement coverage to one context range", () => {
    const scriptContext = createConsoleContext();
    scriptContext.optimizerDocument = {
        mode: "adaptive",
        latest_chunk_outcome: {
            selection: {
                reason: "minimize_projected_remaining_prompt_latency",
                selected_candidate_chunk_size_tokens: 4096
            },
            processed_prompt_token_count: 4096,
            was_reduced_by_memory_capacity: false,
            all_candidates_have_measurements: true,
            measurement_context: {
                chunk_start_token_position: 49152,
                position_range_start_token_position: 32768,
                position_range_end_token_position_exclusive: 65536
            },
            candidate_measurement_summaries: [
                { candidate_chunk_size_tokens: 2048, measurement_count: 3 },
                { candidate_chunk_size_tokens: 4096, measurement_count: 2 }
            ]
        }
    };

    const assessment = vm.runInContext(
        "optimizerAssessment(optimizerDocument)",
        scriptContext
    );

    assert.equal(assessment.title, "Profile ready for tokens 32,768–65,535");
    assert.match(assessment.detail, /only applies to this token range/i);
    assert.equal(assessment.tone, "ready");
});

test("keeps candidate selection and memory reduction distinct", () => {
    const scriptContext = createConsoleContext();
    scriptContext.optimizerDocument = {
        mode: "adaptive",
        latest_chunk_outcome: {
            selection: {
                reason: "explore_unmeasured_candidate",
                selected_candidate_chunk_size_tokens: 8192
            },
            processed_prompt_token_count: 4096,
            was_reduced_by_memory_capacity: true,
            all_candidates_have_measurements: false
        }
    };

    const assessment = vm.runInContext(
        "optimizerAssessment(optimizerDocument)",
        scriptContext
    );

    assert.equal(assessment.title, "Memory capacity changed the latest decision");
    assert.match(assessment.detail, /did not fit.*smaller amount/i);
    assert.equal(assessment.tone, "learning");
});

test("does not label unavailable optimizer configuration as fixed", () => {
    const scriptContext = createConsoleContext();
    assert.equal(vm.runInContext("optimizerModeTitle({ mode: 'unavailable' })", scriptContext), "Unavailable");
});

test("maps every optimizer selection reason without a catch-all substitute", () => {
    const scriptContext = createConsoleContext();
    const mappedReasons = vm.runInContext(`[
        optimizerSelectionReasonTitle("explore_unmeasured_candidate"),
        optimizerSelectionReasonTitle("refresh_stale_candidate_measurement"),
        optimizerSelectionReasonTitle("minimize_projected_remaining_prompt_latency"),
        optimizerSelectionReasonTitle("remaining_tokens_below_smallest_candidate"),
        optimizerSelectionReasonTitle("smallest_candidate_containing_final_prompt_segment"),
        optimizerSelectionReasonTitle("future_reason")
    ]`, scriptContext);

    assert.deepEqual(JSON.parse(JSON.stringify(mappedReasons)), [
        "Explore unmeasured candidate",
        "Refresh stale candidate measurement",
        "Minimize projected remaining prompt latency",
        "Remaining tokens below smallest candidate",
        "Smallest candidate containing final prompt segment",
        "Unknown selection reason"
    ]);
});

test("maps optimizer measurement provenance to user-facing sources", () => {
    const scriptContext = createConsoleContext();
    const mappedSources = vm.runInContext(`[
        optimizerMeasurementSourceTitle("current_position_range"),
        optimizerMeasurementSourceTitle("other_position_ranges_with_same_execution_profile"),
        optimizerMeasurementSourceTitle("no_measurements_available")
    ]`, scriptContext);

    assert.deepEqual(JSON.parse(JSON.stringify(mappedSources)), [
        "Measured in this token range",
        "Evidence from a matching execution context",
        "No evidence for this context yet"
    ]);
});

test("selects the ready model metadata instead of the first advertised model", () => {
    const scriptContext = createConsoleContext();
    const selectedModel = vm.runInContext(
        'selectAdvertisedModel([{ id: "first" }, { id: "ready" }], "ready")',
        scriptContext
    );

    assert.equal(selectedModel.id, "ready");
});

test("computes whole decimal gigabyte bounds from exact status bytes", () => {
    const scriptContext = createConsoleContext();
    const bounds = vm.runInContext(
        "wholeDecimalGigabyteBounds({ minimum_mlx_memory_ceiling_bytes: 32000000001, machine_mlx_memory_ceiling_bytes: 40999999999 })",
        scriptContext
    );
    assert.deepEqual(JSON.parse(JSON.stringify(bounds)), { minimumGigabytes: 33, maximumGigabytes: 40 });
});

test("rejects a serialized chat request after base64 expansion exceeds the HTTP body limit", () => {
    const scriptContext = createConsoleContext();
    const oversizedSerializedRequest = "x".repeat(32 * 1024 * 1024 + 1);
    scriptContext.oversizedSerializedRequest = oversizedSerializedRequest;

    assert.equal(
        vm.runInContext(
            "chatRequestFitsHttpBodyLimit(oversizedSerializedRequest)",
            scriptContext
        ),
        false
    );
});

test("retains partial assistant output in conversation history after interruption", () => {
    const scriptContext = createConsoleContext();
    const assistantHistoryMessage = vm.runInContext(
        'assistantHistoryMessage({ assistantText: "partial", reasoningText: "working" })',
        scriptContext
    );

    assert.deepEqual(
        JSON.parse(JSON.stringify(assistantHistoryMessage)),
        {
            role: "assistant",
            content: "partial",
            reasoning_content: "working"
        }
    );
});

test("describes an attached image in the visible user transcript", () => {
    const scriptContext = createConsoleContext();
    const visibleMessage = vm.runInContext(
        'visibleUserMessageText({ content: [{ type: "text", text: "inspect" }, { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } }] })',
        scriptContext
    );

    assert.equal(visibleMessage, "inspect\n[Image attached]");
});

test("maps every observatory destination to a stable URL path", () => {
    const scriptContext = createConsoleContext();

    const pathMap = vm.runInContext("OBSERVATORY_PATH_MAP", scriptContext);

    assert.equal(pathMap.overview, "/overview");
    assert.equal(pathMap.chat, "/chat");
    assert.equal(pathMap.memory, undefined);
    assert.equal(pathMap.cache, undefined);
    assert.equal(pathMap.optimizer, "/optimizer");
    assert.equal(pathMap.model, "/model");
    assert.equal(pathMap.settings, "/settings");
});

test("maps every observatory URL path back to its destination", () => {
    const scriptContext = createConsoleContext();

    const reverseMap = vm.runInContext("OBSERVATORY_PATH_TO_DESTINATION_MAP", scriptContext);

    assert.equal(reverseMap["/overview"], "overview");
    assert.equal(reverseMap["/chat"], "chat");
    assert.equal(reverseMap["/memory"], undefined);
    assert.equal(reverseMap["/cache"], undefined);
    assert.equal(reverseMap["/optimizer"], "optimizer");
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

test("identifies the candidate with highest observed throughput", () => {
    const scriptContext = createConsoleContext();
    const candidateMeasurementSummaries = [
        {
            candidate_chunk_size_tokens: 1024,
            measurement_count: 5,
            average_processed_prompt_token_count: 800,
            average_forward_elapsed_millis: 400
        },
        {
            candidate_chunk_size_tokens: 2048,
            measurement_count: 3,
            average_processed_prompt_token_count: 1500,
            average_forward_elapsed_millis: 600
        },
        {
            candidate_chunk_size_tokens: 4096,
            measurement_count: 2,
            average_processed_prompt_token_count: 2000,
            average_forward_elapsed_millis: 500
        }
    ];
    scriptContext.candidateMeasurementSummaries = candidateMeasurementSummaries;

    const highestThroughputCandidate = vm.runInContext(
        "highestObservedThroughputCandidate(candidateMeasurementSummaries)",
        scriptContext
    );

    assert.equal(highestThroughputCandidate.candidate_chunk_size_tokens, 4096);
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
            drafterBytes: 0,
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
            drafterBytes: 0,
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
            drafterBytes: 0,
            runtimeWorkBytes: 0,
            availableBytes: 200
        }
    );
});

test("keeps drafter memory separate from model core and runtime work", () => {
    const scriptContext = createConsoleContext();
    scriptContext.mlxMemorySnapshot = {
        active_memory_bytes: 100,
        expert_payload_bytes: 20,
        model_core_payload_bytes: 30,
        context_state_payload_bytes: 10,
        speculative_prefill_draft_memory_bytes: 25
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
            drafterBytes: 25,
            runtimeWorkBytes: 15,
            availableBytes: 100
        }
    );
});

test("returns null when no candidates have measurements", () => {
    const scriptContext = createConsoleContext();
    const candidateMeasurementSummaries = [
        {
            candidate_chunk_size_tokens: 1024,
            measurement_count: 0,
            average_processed_prompt_token_count: 0,
            average_forward_elapsed_millis: 0
        }
    ];
    scriptContext.candidateMeasurementSummaries = candidateMeasurementSummaries;

    const highestThroughputCandidate = vm.runInContext(
        "highestObservedThroughputCandidate(candidateMeasurementSummaries)",
        scriptContext
    );

    assert.equal(highestThroughputCandidate, null);
});

test("skips candidates with zero elapsed time", () => {
    const scriptContext = createConsoleContext();
    const candidateMeasurementSummaries = [
        {
            candidate_chunk_size_tokens: 1024,
            measurement_count: 1,
            average_processed_prompt_token_count: 500,
            average_forward_elapsed_millis: 0
        },
        {
            candidate_chunk_size_tokens: 2048,
            measurement_count: 2,
            average_processed_prompt_token_count: 1000,
            average_forward_elapsed_millis: 500
        }
    ];
    scriptContext.candidateMeasurementSummaries = candidateMeasurementSummaries;

    const highestThroughputCandidate = vm.runInContext(
        "highestObservedThroughputCandidate(candidateMeasurementSummaries)",
        scriptContext
    );

    assert.equal(highestThroughputCandidate.candidate_chunk_size_tokens, 2048);
});

test("returns null for empty candidate measurements", () => {
    const scriptContext = createConsoleContext();
    const highestThroughputCandidate = vm.runInContext(
        "highestObservedThroughputCandidate([])",
        scriptContext
    );

    assert.equal(highestThroughputCandidate, null);
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

test("does not declare one highest-throughput candidate when measurements are tied", () => {
    const scriptContext = createConsoleContext();
    const candidateMeasurementSummaries = [
        {
            candidate_chunk_size_tokens: 1024,
            measurement_count: 2,
            average_processed_prompt_token_count: 1000,
            average_forward_elapsed_millis: 500
        },
        {
            candidate_chunk_size_tokens: 2048,
            measurement_count: 2,
            average_processed_prompt_token_count: 2000,
            average_forward_elapsed_millis: 1000
        }
    ];
    scriptContext.candidateMeasurementSummaries = candidateMeasurementSummaries;

    const highestThroughputCandidate = vm.runInContext(
        "highestObservedThroughputCandidate(candidateMeasurementSummaries)",
        scriptContext
    );

    assert.equal(highestThroughputCandidate, null);
});
