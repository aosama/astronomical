// Loaded model panel and status header tests: what Overview shows for the
// loaded model, and that nothing describes a model once it is unloaded.

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");

const { createConsoleContext, attachRecordingElements } = require("./console-test-support.js");

test("should_stop_naming_an_unloaded_model_in_the_status_header", () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);

    vm.runInContext("renderStatusHeader({ status: 'ready', ready_model_id: 'loaded-model' })", scriptContext);
    assert.equal(elementsById["status-model-id"].textContent, "loaded-model");
    vm.runInContext("renderStatusHeader({ status: 'ready', ready_model_id: null })", scriptContext);

    assert.equal(elementsById["status-model-id"].textContent, "No model loaded");
});

test("should_clear_the_last_model_footprint_once_nothing_is_loaded", () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    scriptContext.expertMemoryModePhrase = () => "All in RAM";

    vm.runInContext("renderLoadedModelFootprint({ status: 'ready', ready_model_id: 'loaded-model', ready_model_size_bytes: 622854144, expert_memory_mode: 'resident' })", scriptContext);
    assert.notEqual(elementsById["loaded-model-disk-size"].textContent, "—");
    vm.runInContext("renderLoadedModelFootprint({ status: 'ready', ready_model_id: null, expert_memory_mode: 'resident' })", scriptContext);

    assert.equal(elementsById["loaded-model-disk-size"].textContent, "—");
    assert.equal(elementsById["loaded-model-residency"].textContent, "Not loaded", "a leftover mode does not describe a model that is gone");
});

test("should_show_no_expert_memory_for_a_leftover_mode_once_nothing_is_loaded", () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    scriptContext.expertMemoryModePhrase = () => "All in RAM";

    vm.runInContext("renderLoadedModelFromStatus({ status: 'ready', ready_model_id: null, expert_memory_mode: 'resident' })", scriptContext);

    assert.equal(elementsById["loaded-model-expert-memory"].textContent, "—");
});

test("should_describe_only_the_loaded_model_and_show_dashes_when_none_is_loaded", () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    // experts.js supplies this phrase in the page; the expert wording is not under test.
    scriptContext.expertMemoryModePhrase = () => "";
    const loadedModel = { id: "loaded-model", context_window: 262144, max_input_tokens: 262143, max_output_tokens: 65535, input_modalities: ["text", "image"] };

    vm.runInContext("renderLoadedModelFromStatus({ ready_model_id: 'loaded-model', status: 'ready' })", scriptContext);
    scriptContext.loadedModelForTest = loadedModel;
    vm.runInContext("renderLoadedModelLimits(loadedModelForTest)", scriptContext);
    assert.equal(elementsById["loaded-model-id"].textContent, "loaded-model");
    assert.equal(elementsById["loaded-model-context-window"].textContent, (262144).toLocaleString());
    assert.equal(elementsById["loaded-model-modalities"].textContent, "Text + Image");

    vm.runInContext("renderLoadedModelFromStatus({ ready_model_id: null, status: 'ready' })", scriptContext);
    vm.runInContext("renderLoadedModelLimits(null)", scriptContext);
    assert.equal(elementsById["loaded-model-id"].textContent, "No model loaded", "an unloaded model's name does not linger");
    for (const factId of ["loaded-model-context-window", "loaded-model-max-input", "loaded-model-max-output", "loaded-model-modalities"]) {
        assert.equal(elementsById[factId].textContent, "—", `${factId} shows no other model's limits`);
    }
});
