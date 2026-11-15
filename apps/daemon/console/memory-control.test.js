// Maximum model RAM control tests: bounds, the applied value, notices, and the
// feedback and config-file-written event after a change.

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");

const { createConsoleContext, attachRecordingElements } = require("./console-test-support.js");

function recordConfigFileWrittenEvents(scriptContext) {
    const dispatchedEventTypes = [];
    scriptContext.CustomEvent = class { constructor(eventType) { this.type = eventType; } };
    scriptContext.document.dispatchEvent = (dispatchedEvent) => { dispatchedEventTypes.push(dispatchedEvent.type); };
    return dispatchedEventTypes;
}

test("should_report_a_memory_change_in_its_own_panel_and_tell_the_editor_the_file_changed", async () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    const dispatchedEventTypes = recordConfigFileWrittenEvents(scriptContext);
    scriptContext.pollStatus = async () => {};
    scriptContext.fetch = async () => ({ ok: true, status: 200, json: async () => ({ message: "Memory limit set to 34 GB" }) });

    await vm.runInContext("updateMaximumMlxMemory(34)", scriptContext);

    assert.equal(elementsById["maximum-mlx-memory-feedback"].textContent, "Memory limit set to 34 GB");
    assert.equal(elementsById["maximum-mlx-memory-feedback"].className, "control-feedback feedback-success");
    assert.deepEqual(dispatchedEventTypes, ["observatory:config-file-written"]);
});

test("should_show_a_queued_memory_change_as_progress", async () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    recordConfigFileWrittenEvents(scriptContext);
    scriptContext.pollStatus = async () => {};
    scriptContext.fetch = async () => ({ ok: true, status: 202, json: async () => ({ message: "Applies after the current reply" }) });

    await vm.runInContext("updateMaximumMlxMemory(34)", scriptContext);

    assert.equal(elementsById["maximum-mlx-memory-feedback"].className, "control-feedback feedback-progress");
});

test("should_tell_the_editor_the_file_changed_even_when_a_memory_change_is_refused", async () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    const dispatchedEventTypes = recordConfigFileWrittenEvents(scriptContext);
    scriptContext.fetch = async () => ({ ok: false, status: 400, json: async () => ({ message: "Out of range" }) });

    await vm.runInContext("updateMaximumMlxMemory(999)", scriptContext);

    assert.equal(elementsById["maximum-mlx-memory-feedback"].className, "control-feedback feedback-error");
    assert.deepEqual(dispatchedEventTypes, ["observatory:config-file-written"], "a refused ceiling can still be saved as the persisted intent");
});

test("should_not_tell_the_editor_anything_when_the_memory_request_never_answered", async () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    const dispatchedEventTypes = recordConfigFileWrittenEvents(scriptContext);
    scriptContext.fetch = async () => { throw new Error("offline"); };

    await vm.runInContext("updateMaximumMlxMemory(34)", scriptContext);

    assert.equal(elementsById["maximum-mlx-memory-feedback"].textContent, "Memory limit update failed: offline");
    assert.deepEqual(dispatchedEventTypes, []);
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
