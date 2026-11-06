const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const panelScriptPath = path.join(__dirname, "neural-engine-panel.js");
const panelContext = vm.createContext({});
vm.runInContext(fs.readFileSync(panelScriptPath, "utf8"), panelContext, { filename: panelScriptPath });

function call(functionName, ...functionArguments) {
    panelContext.__arguments = functionArguments;
    const returned = vm.runInContext(`${functionName}(...__arguments)`, panelContext);
    return returned === undefined ? undefined : JSON.parse(JSON.stringify(returned));
}

const GIGABYTE = 1073741824;

function statusDocument(prefillFields, totals) {
    return {
        mlx_memory_ceiling_bytes: 32 * GIGABYTE,
        neural_engine_prefill: prefillFields,
        serving_session: { neural_engine: totals || {} },
    };
}

function activePrefill(measuredSpeedPercent) {
    return {
        state: "active",
        layer_program_count: 28,
        neural_engine_tokens_per_chunk: 960,
        chunk_token_count: 2048,
        measured_memory_bytes: 15.04 * GIGABYTE,
        measured_speed_percent: measuredSpeedPercent,
    };
}

test("should_hide_the_panel_from_someone_who_left_the_neural_engine_off", () => {
    assert.equal(call("neuralEnginePanelContent", statusDocument(undefined)).isVisible, false);
    assert.equal(call("neuralEnginePanelContent", statusDocument({ state: "off" })).isVisible, false);
    assert.equal(call("neuralEnginePanelContent", {}).isVisible, false);
});

test("should_show_speed_work_and_saved_time_when_measured", () => {
    const content = call("neuralEnginePanelContent", statusDocument(activePrefill(128), {
        prompt_token_count: 98304,
        neural_engine_token_count: 45120,
        estimated_saved_ms: 42000,
    }));

    assert.equal(content.tiles.speed.value, "1.28×");
    assert.equal(content.tiles.work.value, "46%");
    assert.match(content.tiles.work.detail, /45,120 of 98,304 prompt tokens/);
    assert.equal(content.tiles.saved.value, "about 42 s");
    assert.match(content.tiles.saved.detail, /estimated, warm prompts only/);
    assert.match(content.note, /about 16 GB.*up to 47%/);
});

test("should_not_guess_at_saved_time_while_measuring", () => {
    const content = call("neuralEnginePanelContent", statusDocument(activePrefill(undefined), {
        prompt_token_count: 20480,
        neural_engine_token_count: 4800,
    }));

    assert.equal(content.tiles.speed.value, "Measuring");
    assert.equal(content.tiles.saved.value, "—");
    assert.match(content.note, /take turns/);
});

test("should_keep_time_already_saved_while_a_new_model_is_measured", () => {
    const content = call("neuralEnginePanelContent", statusDocument(activePrefill(undefined), {
        prompt_token_count: 20480,
        neural_engine_token_count: 4800,
        estimated_saved_ms: 90000,
    }));

    assert.equal(content.tiles.saved.value, "about 1 min 30 s");
});

test("should_say_none_yet_when_nothing_has_been_saved", () => {
    const content = call("neuralEnginePanelContent", statusDocument(activePrefill(128), {
        prompt_token_count: 2048,
        neural_engine_token_count: 0,
        estimated_saved_ms: 0,
    }));

    assert.equal(content.tiles.work.value, "0%");
    assert.equal(content.tiles.saved.value, "None yet");
});

test("should_explain_an_empty_work_tile_when_no_long_prompt_has_run", () => {
    const content = call("neuralEnginePanelContent", statusDocument(activePrefill(128), {}));

    assert.equal(content.tiles.work.value, "—");
    assert.match(content.tiles.work.detail, /only long prompts/);
});

test("should_name_the_ceiling_when_memory_moves_a_prompt_to_the_gpu", () => {
    const document = statusDocument(activePrefill(128), { last_chunk_placement: "gpu_memory_tight" });
    document.progress = { phase: "target" };

    assert.match(call("neuralEnginePanelContent", document).note, /16 GB wouldn't fit under your 32 GB memory ceiling/);
});

test("should_not_describe_memory_when_no_request_is_running", () => {
    const content = call("neuralEnginePanelContent", statusDocument(activePrefill(128), {
        last_chunk_placement: "gpu_memory_tight",
    }));

    assert.match(content.note, /Using about 16 GB/);
});

test("should_count_programs_while_preparing_and_show_no_tiles", () => {
    const content = call("neuralEnginePanelContent", statusDocument({
        state: "preparing", completed_program_count: 17, total_program_count: 56,
    }));

    assert.equal(content.tiles, null);
    assert.match(content.note, /Preparing programs, 17 of 56.*already serving on the GPU/);
});

test("should_say_when_built_programs_wait_for_a_pause", () => {
    const content = call("neuralEnginePanelContent", statusDocument({
        state: "preparing", completed_program_count: 56, total_program_count: 56, is_waiting_to_measure: true,
    }));

    assert.match(content.note, /56 programs built.*next pause/);
});

test("should_give_the_reason_when_the_programs_are_not_in_use", () => {
    const content = call("neuralEnginePanelContent", statusDocument({
        state: "unavailable",
        reason: "the Neural Engine ran this model at 0.90 times the speed of the GPU alone",
    }));

    assert.equal(content.tiles, null);
    assert.match(content.note, /^Not in use: the Neural Engine ran this model at 0\.90 times/);
});

test("should_round_saved_time_for_reading", () => {
    assert.equal(call("formatEstimatedSavedTime", 1), "under 1 s");
    assert.equal(call("formatEstimatedSavedTime", 999), "under 1 s");
    assert.equal(call("formatEstimatedSavedTime", 1000), "about 1 s");
    assert.equal(call("formatEstimatedSavedTime", 59000), "about 59 s");
    assert.equal(call("formatEstimatedSavedTime", 60000), "about 1 min 0 s");
    assert.equal(call("formatEstimatedSavedTime", 3600000), "about 1 h 0 min");
    assert.equal(call("formatEstimatedSavedTime", 5460000), "about 1 h 31 min");
});

test("should_label_the_current_request_with_its_engines", () => {
    const measured = activePrefill(128);
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument(measured, { last_chunk_placement: "both_engines" })), " · both engines");
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument(measured, { last_chunk_placement: "gpu_waking_programs" })), " · GPU");
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument(measured, { last_chunk_placement: "gpu_memory_tight" })), " · GPU, memory is tight");
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument(measured, {})), "");
});

test("should_hold_one_label_while_the_engines_are_compared", () => {
    const measuring = activePrefill(undefined);
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument(measuring, { last_chunk_placement: "both_engines" })), " · comparing engines");
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument(measuring, { last_chunk_placement: "gpu_while_comparing" })), " · comparing engines");
});

test("should_add_no_label_when_the_neural_engine_is_off", () => {
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument(undefined)), "");
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument({ state: "off" })), "");
    assert.equal(call("neuralEnginePhaseSuffix", statusDocument({ state: "unavailable", reason: "x" })), " · GPU");
});

test("should_never_write_markup_from_status_values", () => {
    const panelScript = fs.readFileSync(panelScriptPath, "utf8");
    assert.equal(panelScript.includes("innerHTML"), false);
    assert.equal(panelScript.includes("insertAdjacentHTML"), false);
});
