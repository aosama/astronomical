const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const modelScriptPath = path.join(__dirname, "configuration-model.js");
const modelContext = vm.createContext({});
vm.runInContext(fs.readFileSync(modelScriptPath, "utf8"), modelContext, { filename: modelScriptPath });

const bundledSchema = JSON.parse(fs.readFileSync(
    path.join(__dirname, "../../../site/schemas/config/v1/astronomical-config.schema.json"), "utf8"));

function call(functionName, ...functionArguments) {
    // Crossing the vm boundary and back through JSON keeps deep equality honest.
    modelContext.__arguments = functionArguments;
    const returned = vm.runInContext(`${functionName}(...__arguments)`, modelContext);
    return returned === undefined ? undefined : JSON.parse(JSON.stringify(returned));
}

const baseDocument = {
    $schema: "./astronomical-config.schema.json",
    schema_version: 1,
    runtime: { model_directories: [] },
};

function descriptorAt(settingPath) {
    let descriptors = call("configurationSettingDescriptors", bundledSchema, bundledSchema, []);
    let found;
    for (const pathKey of settingPath) {
        found = descriptors.find((descriptor) => descriptor.key === pathKey);
        descriptors = found.children;
    }
    return found;
}

test("should_offer_every_section_of_the_schema_except_the_format_constants", () => {
    const descriptors = call("configurationSettingDescriptors", bundledSchema, bundledSchema, []);
    assert.deepEqual(descriptors.map((descriptor) => descriptor.key),
        ["runtime", "prompt_cache", "chunking", "models", "diagnostics"]);
});

test("should_describe_every_leaf_setting_with_a_supported_control", () => {
    const supportedKinds = new Set(["boolean", "integer", "number", "string", "enum",
        "string_list", "nullable_integer", "group", "keyed_objects"]);
    function visit(descriptors) {
        for (const descriptor of descriptors) {
            assert.ok(supportedKinds.has(descriptor.kind), `${descriptor.key}: ${descriptor.kind}`);
            if (descriptor.children) { visit(descriptor.children); }
        }
    }
    visit(call("configurationSettingDescriptors", bundledSchema, bundledSchema, []));
    const modelEntry = call("configurationSettingDescriptors", bundledSchema,
        bundledSchema.properties.models.additionalProperties, ["models", "some-model"]);
    assert.deepEqual(modelEntry.map((descriptor) => descriptor.key),
        ["limits", "generation_defaults", "chunking"]);
    visit(modelEntry);
});

test("should_expose_the_neural_engine_opt_in_as_a_boolean_that_defaults_off", () => {
    const descriptor = descriptorAt(["chunking", "experimental_neural_engine_prefill_enabled"]);
    assert.equal(descriptor.kind, "boolean");
    assert.equal(descriptor.defaultValue, false);
    assert.equal(descriptor.applyMode, "model-reload");
    assert.equal(call("configurationDefaultDescription", descriptor), "Default: off");
});

test("should_expose_metal_residency_refresh_as_a_switch_that_restarts_the_worker", () => {
    const descriptor = descriptorAt(["runtime", "metal_residency_refresh_enabled"]);
    assert.equal(descriptor.kind, "boolean");
    assert.equal(descriptor.defaultValue, false);
    assert.equal(descriptor.isRequired, false);
    assert.equal(call("configurationDefaultDescription", descriptor), "Default: off");
    assert.equal(call("configurationApplyModeDescription", descriptor.applyMode),
        "Restarts the inference worker");
});

test("should_keep_model_directories_required_so_it_cannot_be_removed", () => {
    assert.equal(descriptorAt(["runtime", "model_directories"]).isRequired, true);
    assert.equal(descriptorAt(["runtime", "default_model"]).isRequired, false);
});

test("should_read_bounds_from_the_integer_choice_of_a_nullable_setting", () => {
    const descriptor = descriptorAt(["chunking", "speculative_decoding_block_length_tokens"]);
    assert.equal(descriptor.kind, "nullable_integer");
    assert.equal(descriptor.minimum, 1);
    assert.equal(descriptor.maximum, 64);
});

test("should_set_a_nested_value_without_changing_the_original_document", () => {
    const updated = call("withConfigurationValue", baseDocument, ["diagnostics", "log_level"], "info");
    assert.equal(updated.diagnostics.log_level, "info");
    assert.equal(baseDocument.diagnostics, undefined);
});

test("should_remove_a_value_and_prune_parents_it_left_empty", () => {
    const withValue = call("withConfigurationValue", baseDocument, ["diagnostics", "log_level"], "info");
    assert.deepEqual(call("withoutConfigurationValue", withValue, ["diagnostics", "log_level"]), baseDocument);
});

test("should_keep_a_parent_that_still_has_other_settings", () => {
    const document = { ...baseDocument, diagnostics: { log_level: "info", retained_log_files: 3 } };
    assert.deepEqual(call("withoutConfigurationValue", document, ["diagnostics", "log_level"]).diagnostics,
        { retained_log_files: 3 });
});

test("should_never_prune_the_required_runtime_object", () => {
    const document = { ...baseDocument, runtime: { default_model: "some-model" } };
    assert.deepEqual(call("withoutConfigurationValue", document, ["runtime", "default_model"]).runtime, {});
});

test("should_ignore_removing_a_value_that_is_not_set", () => {
    assert.deepEqual(call("withoutConfigurationValue", baseDocument, ["diagnostics", "log_level"]), baseDocument);
});

test("should_treat_member_order_as_no_edit_and_a_changed_value_as_one", () => {
    assert.equal(call("configurationDocumentsAreEqual", { a: 1, b: { c: 2, d: 3 } }, { b: { d: 3, c: 2 }, a: 1 }), true);
    assert.equal(call("configurationDocumentsAreEqual", { a: 1 }, { a: 2 }), false);
    assert.equal(call("configurationDocumentsAreEqual", { a: [1, 2] }, { a: [2, 1] }), false);
});

test("should_parse_a_whole_number_inside_its_bounds", () => {
    const descriptor = { kind: "integer", minimum: 1, maximum: 10 };
    assert.deepEqual(call("parseConfigurationInput", descriptor, " 7 "), { isValid: true, value: 7 });
    assert.deepEqual(call("parseConfigurationInput", descriptor, "1"), { isValid: true, value: 1 });
    assert.deepEqual(call("parseConfigurationInput", descriptor, "10"), { isValid: true, value: 10 });
});

test("should_refuse_numbers_outside_their_bounds_with_a_message", () => {
    const descriptor = { kind: "integer", minimum: 1, maximum: 10 };
    assert.equal(call("parseConfigurationInput", descriptor, "0").message, "Must be at least 1");
    assert.equal(call("parseConfigurationInput", descriptor, "11").message, "Must be at most 10");
});

test("should_refuse_a_fraction_for_a_whole_number_and_accept_it_for_a_decimal", () => {
    assert.equal(call("parseConfigurationInput", { kind: "integer" }, "1.5").isValid, false);
    assert.deepEqual(call("parseConfigurationInput", { kind: "number" }, "1.5"), { isValid: true, value: 1.5 });
});

test("should_refuse_text_where_a_number_belongs", () => {
    assert.equal(call("parseConfigurationInput", { kind: "integer" }, "many").isValid, false);
    assert.equal(call("parseConfigurationInput", { kind: "integer" }, "").isValid, false);
});

test("should_accept_null_only_for_a_nullable_setting", () => {
    assert.deepEqual(call("parseConfigurationInput", { kind: "nullable_integer" }, "null"), { isValid: true, value: null });
    assert.equal(call("parseConfigurationInput", { kind: "integer" }, "null").isValid, false);
});

test("should_refuse_a_blank_string", () => {
    assert.equal(call("parseConfigurationInput", { kind: "string" }, "   ").isValid, false);
    assert.deepEqual(call("parseConfigurationInput", { kind: "string" }, " /models "), { isValid: true, value: "/models" });
});

test("should_label_settings_with_their_acronyms_uppercased", () => {
    assert.equal(call("configurationSettingLabel", "maximum_mlx_memory_gb"), "Maximum MLX memory GB");
    assert.equal(call("configurationSettingLabel", "experimental_neural_engine_prefill_enabled"),
        "Experimental neural engine prefill enabled");
});

test("should_report_a_save_that_applied", () => {
    assert.deepEqual(call("configurationSaveOutcomeMessage", { reload_http_status: 200, reload: { status: "reloaded" } }),
        { tone: "success", text: "Saved and applied." });
});

test("should_report_a_save_waiting_on_an_active_request_as_saved", () => {
    const outcome = call("configurationSaveOutcomeMessage", { reload_http_status: 409, reload: { status: "busy" } });
    assert.equal(outcome.tone, "progress");
    assert.match(outcome.text, /^Saved\./);
});

test("should_name_the_fields_a_restart_must_apply", () => {
    const outcome = call("configurationSaveOutcomeMessage", { reload_http_status: 200,
        reload: { rest_api_restart_required: true, restart_required_fields: ["model_directories"] } });
    assert.match(outcome.text, /Restart the app to apply: model_directories/);
});

test("should_report_a_saved_file_that_could_not_be_applied_as_an_error", () => {
    const outcome = call("configurationSaveOutcomeMessage", { reload_http_status: 500, reload: { message: "worker failed" } });
    assert.equal(outcome.tone, "error");
    assert.match(outcome.text, /worker failed/);
});

test("should_describe_each_neural_engine_state_in_words", () => {
    assert.match(call("neuralEnginePrefillStatusMessage", null), /^Off/);
    assert.match(call("neuralEnginePrefillStatusMessage", { state: "active",
        neural_engine_tokens_per_chunk: 960, chunk_token_count: 2048, layer_program_count: 32,
        measured_memory_bytes: 14814576640 }), /960 of every 2048.*32 layers.*14 GB.*Comparing/);
    assert.match(call("neuralEnginePrefillStatusMessage", { state: "active",
        neural_engine_tokens_per_chunk: 960, chunk_token_count: 2048, layer_program_count: 32,
        measured_memory_bytes: 14814576640, measured_speed_percent: 118 }), /Measured 1\.18 times/);
    assert.match(call("neuralEnginePrefillStatusMessage", { state: "preparing",
        completed_program_count: 56, total_program_count: 56, elapsed_ms: 1000,
        estimated_remaining_ms: 33000, is_waiting_to_measure: true }), /56 programs built.*next pause/);
    assert.match(call("neuralEnginePrefillStatusMessage", { state: "preparing",
        completed_program_count: 12, total_program_count: 56, elapsed_ms: 1000,
        estimated_remaining_ms: null }), /Preparing: 12 of 56 programs ready.*GPU/);
    assert.match(call("neuralEnginePrefillStatusMessage", { state: "unavailable", reason: "only dense models are supported" }),
        /Enabled but not running: only dense models are supported/);
});

test("should_never_write_markup_from_configuration_values", () => {
    const pageScript = fs.readFileSync(path.join(__dirname, "configuration.js"), "utf8");
    assert.equal(pageScript.includes("innerHTML"), false);
    assert.equal(pageScript.includes("insertAdjacentHTML"), false);
});

test("should_load_the_page_scripts_before_the_console_script_that_wires_them", () => {
    const shell = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
    assert.ok(shell.indexOf("/configuration.js") < shell.indexOf("/console.js"));
    assert.ok(shell.indexOf("/configuration-model.js") < shell.indexOf("/configuration.js"));
});
