const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const layoutContext = vm.createContext({});
for (const scriptName of ["configuration-model.js", "configuration-layout.js"]) {
    const scriptPath = path.join(__dirname, scriptName);
    vm.runInContext(fs.readFileSync(scriptPath, "utf8"), layoutContext, { filename: scriptPath });
}

const bundledSchema = JSON.parse(fs.readFileSync(
    path.join(__dirname, "../../../site/schemas/config/v1/astronomical-config.schema.json"), "utf8"));

function call(functionName, ...functionArguments) {
    // Crossing the vm boundary and back through JSON keeps deep equality honest.
    layoutContext.__arguments = functionArguments;
    const returned = vm.runInContext(`${functionName}(...__arguments)`, layoutContext);
    return returned === undefined ? undefined : JSON.parse(JSON.stringify(returned));
}

const baseDocument = {
    $schema: "./astronomical-config.schema.json",
    schema_version: 1,
    runtime: { model_directories: ["/models"] },
};

function groupTitles(pageName, schema = bundledSchema) {
    return call("configurationPageGroups", schema, pageName).map((group) => group.title);
}

function listedPaths(pageName) {
    return bundledSchema["x-astronomical-layout"][pageName].flatMap((group) => group.settings);
}

test("should_list_every_schema_setting_on_a_page_so_no_trailing_other_group_appears", () => {
    assert.deepEqual(groupTitles("settings"), ["Memory", "Locations", "Prompt cache", "Diagnostics"]);
    assert.ok(!groupTitles("tuning").includes("Other"));
});

test("should_resolve_every_path_the_layout_names_to_a_real_setting", () => {
    for (const pageName of ["settings", "tuning"]) {
        const groups = call("configurationPageGroups", bundledSchema, pageName);
        const resolvedPaths = groups.flatMap((group) => group.descriptors.map((descriptor) => descriptor.path.join(".")));
        assert.deepEqual(resolvedPaths.sort(), listedPaths(pageName).sort(), pageName);
    }
});

test("should_list_each_setting_once_across_both_pages", () => {
    const allListedPaths = listedPaths("settings").concat(listedPaths("tuning"));
    assert.equal(new Set(allListedPaths).size, allListedPaths.length);
});

test("should_keep_the_top_level_chunking_defaults_off_the_settings_page", () => {
    const settingsPaths = call("configurationPageGroups", bundledSchema, "settings")
        .flatMap((group) => group.descriptors.map((descriptor) => descriptor.path[0]));
    assert.ok(!settingsPaths.includes("chunking"));
    assert.ok(!settingsPaths.includes("models"));
});

test("should_offer_all_models_only_the_settings_every_model_starts_from", () => {
    const tuningGroups = call("configurationPageGroups", bundledSchema, "tuning");
    const allModelsGroups = call("configurationGroupsForAllModels", bundledSchema, tuningGroups);
    const paths = allModelsGroups.flatMap((group) => group.descriptors.map((descriptor) => descriptor.path[0]));
    assert.ok(paths.length > 0);
    assert.ok(paths.every((rootKey) => rootKey === "chunking"));
    assert.ok(!allModelsGroups.map((group) => group.title).includes("Generation"));
});

test("should_mark_the_slow_diagnostic_modes_and_say_so_once_on_their_group", () => {
    const diagnostics = call("configurationPageGroups", bundledSchema, "tuning").find((group) => group.tier === "diagnostic");
    assert.equal(diagnostics.title, "Diagnostics");
    assert.match(diagnostics.note, /Leave off/);
});

test("should_append_an_unlisted_setting_to_a_trailing_other_group", () => {
    const schemaWithNewSetting = JSON.parse(JSON.stringify(bundledSchema));
    schemaWithNewSetting.$defs.diagnostics.properties.trace_sampling_rate = { type: "integer", title: "Trace sampling rate" };
    const groups = call("configurationPageGroups", schemaWithNewSetting, "settings");
    assert.equal(groups[groups.length - 1].title, "Other");
    assert.deepEqual(groups[groups.length - 1].descriptors.map((descriptor) => descriptor.path.join(".")),
        ["diagnostics.trace_sampling_rate"]);
});

test("should_label_a_setting_with_its_schema_title_and_carry_its_unit", () => {
    const descriptor = call("configurationPageGroups", bundledSchema, "settings")[0].descriptors
        .find((candidate) => candidate.key === "idle_model_unload_seconds");
    assert.equal(descriptor.label, "Unload a model after idle for");
    assert.equal(descriptor.unit, "seconds");
});

test("should_give_every_boolean_a_default_so_its_switch_has_a_resting_value", () => {
    const leaves = ["settings", "tuning"].flatMap((pageName) =>
        call("configurationPageGroups", bundledSchema, pageName).flatMap((group) => group.descriptors));
    for (const descriptor of leaves.filter((leaf) => leaf.kind === "boolean")) {
        assert.equal(typeof descriptor.defaultValue, "boolean", descriptor.path.join("."));
    }
});

test("should_build_a_path_below_the_model_entry_or_leave_it_alone_for_all_models", () => {
    assert.deepEqual(call("configurationScopedPath", null, ["chunking", "x"]), ["chunking", "x"]);
    assert.deepEqual(call("configurationScopedPath", "some-model", ["chunking", "x"]), ["models", "some-model", "chunking", "x"]);
});

test("should_inherit_the_all_models_value_when_the_file_sets_one", () => {
    const descriptor = { path: ["chunking", "prompt_cache_common_prefix_stride_blocks"], defaultValue: 4 };
    const document = { ...baseDocument, chunking: { prompt_cache_common_prefix_stride_blocks: 8 } };
    assert.deepEqual(call("configurationBaselineValue", bundledSchema, document, "some-model", descriptor),
        { value: 8, isInherited: true });
});

test("should_fall_back_to_the_schema_default_for_all_models_and_for_a_model_without_a_default_layer", () => {
    const descriptor = { path: ["chunking", "prompt_cache_common_prefix_stride_blocks"], defaultValue: 4 };
    const document = { ...baseDocument, chunking: { prompt_cache_common_prefix_stride_blocks: 8 } };
    assert.deepEqual(call("configurationBaselineValue", bundledSchema, document, null, descriptor), { value: 4, isInherited: false });
    const temperature = { path: ["generation_defaults", "temperature"], defaultValue: undefined };
    assert.equal(call("configurationBaselineValue", bundledSchema, document, "some-model", temperature).isInherited, false);
});

test("should_show_the_reset_arrow_only_for_a_value_that_differs_from_its_baseline", () => {
    assert.equal(call("configurationValueDiffersFromBaseline", 3, 4), true);
    assert.equal(call("configurationValueDiffersFromBaseline", 4, 4), false);
    assert.equal(call("configurationValueDiffersFromBaseline", undefined, 4), false);
    assert.equal(call("configurationValueDiffersFromBaseline", 0.5, undefined), true);
    assert.equal(call("configurationValueDiffersFromBaseline", null, null), false);
});

test("should_drop_a_value_set_back_to_its_baseline_unless_the_saved_file_pinned_it", () => {
    assert.equal(call("configurationShouldRemoveValue", false, false, false), true);
    assert.equal(call("configurationShouldRemoveValue", false, false, true), false);
    assert.equal(call("configurationShouldRemoveValue", true, false, false), false);
    assert.equal(call("configurationShouldRemoveValue", 1, undefined, false), false);
});

test("should_note_the_apply_mode_and_the_default_or_inherited_value", () => {
    const descriptor = { applyMode: "model-reload", defaultValue: 2048, kind: "integer" };
    assert.equal(call("configurationSettingNote", descriptor, { value: 2048, isInherited: false }),
        "Applies when the model next loads. Default: 2048.");
    assert.equal(call("configurationSettingNote", descriptor, { value: 4096, isInherited: true }),
        "Applies when the model next loads. Inherits 4096 from All models.");
});

test("should_find_no_changes_in_an_unedited_document", () => {
    const summary = call("configurationChangeSummary", bundledSchema, baseDocument, baseDocument);
    assert.equal(summary.changeCount, 0);
    assert.deepEqual(summary.editedPages, { settings: false, tuning: false });
});

test("should_count_each_changed_value_and_name_the_restarts_it_needs", () => {
    const working = { ...baseDocument,
        runtime: { model_directories: ["/models"], metal_residency_refresh_enabled: true, idle_model_unload_seconds: 120 },
        diagnostics: { performance_attribution_enabled: true } };
    const summary = call("configurationChangeSummary", bundledSchema, baseDocument, working);
    assert.equal(summary.changeCount, 3);
    assert.equal(summary.workerRestartCount, 1);
    assert.equal(summary.applicationRestartCount, 1);
    assert.deepEqual(summary.editedPages, { settings: true, tuning: false });
});

test("should_attribute_a_model_edit_and_an_all_models_edit_to_the_tuning_page", () => {
    const modelEdit = { ...baseDocument, models: { "some-model": { generation_defaults: { temperature: 0.7 } } } };
    assert.deepEqual(call("configurationChangeSummary", bundledSchema, baseDocument, modelEdit).editedPages,
        { settings: false, tuning: true });
    const defaultsEdit = { ...baseDocument, chunking: { fixed_prompt_processing_chunk_size_tokens: 1024 } };
    assert.deepEqual(call("configurationChangeSummary", bundledSchema, baseDocument, defaultsEdit).editedPages,
        { settings: false, tuning: true });
});

test("should_count_a_removed_value_as_a_change_and_an_array_edit_as_one", () => {
    const saved = { ...baseDocument, diagnostics: { log_level: "info" } };
    assert.equal(call("configurationChangeSummary", bundledSchema, saved, baseDocument).changeCount, 1);
    const moreDirectories = { ...baseDocument, runtime: { model_directories: ["/models", "/more"] } };
    assert.equal(call("configurationChangeSummary", bundledSchema, baseDocument, moreDirectories).changeCount, 1);
});

test("should_write_the_change_count_and_restarts_in_words", () => {
    assert.deepEqual(call("configurationChangeSummaryText", { changeCount: 1, workerRestartCount: 0, applicationRestartCount: 0 }),
        { countText: "1 change", detailText: "" });
    assert.deepEqual(call("configurationChangeSummaryText", { changeCount: 3, workerRestartCount: 1, applicationRestartCount: 2 }),
        { countText: "3 changes", detailText: "1 restarts the inference worker · 2 needs an app restart" });
});

test("should_list_discovered_models_then_file_only_keys_then_models_that_cannot_chat", () => {
    const discovered = [
        { id: "b-chat", servesChat: true }, { id: "a-picture", servesChat: false }, { id: "a-chat", servesChat: true }];
    const document = { ...baseDocument, models: { "hand/written-id": { generation_defaults: { temperature: 0.5 } } } };
    const entries = call("configurationTuningListEntries", discovered, document, ["added-id"]);
    assert.deepEqual(entries.map((entry) => entry.id), ["a-chat", "b-chat", "added-id", "hand/written-id", "a-picture"]);
    assert.equal(entries.find((entry) => entry.id === "hand/written-id").isDiscovered, false);
    assert.equal(entries.find((entry) => entry.id === "hand/written-id").settingCount, 1);
});

test("should_count_zero_settings_for_a_model_the_file_does_not_mention", () => {
    const entries = call("configurationTuningListEntries", [{ id: "a-chat", servesChat: true }], baseDocument, []);
    assert.equal(entries[0].settingCount, 0);
});

test("should_report_unsaved_edits_for_one_model_only", () => {
    const edited = { ...baseDocument, models: { "a-chat": { limits: { maximum_context_tokens: 4096 } } } };
    assert.equal(call("configurationModelHasUnsavedEdits", baseDocument, edited, "a-chat"), true);
    assert.equal(call("configurationModelHasUnsavedEdits", baseDocument, edited, "b-chat"), false);
});

test("should_treat_a_model_that_serves_chat_completions_as_tunable", () => {
    assert.deepEqual(call("configurationDiscoveredModelFromAdvertisement",
        { id: "chat-model", supported_endpoints: ["/v1/chat/completions", "/v1/responses"] }),
        { id: "chat-model", servesChat: true });
    assert.deepEqual(call("configurationDiscoveredModelFromAdvertisement",
        { id: "picture-model", supported_endpoints: ["/v1/images/generations"] }),
        { id: "picture-model", servesChat: false });
    assert.equal(call("configurationDiscoveredModelFromAdvertisement", { id: "bare" }).servesChat, false);
});

test("should_count_the_values_a_reset_would_remove_but_not_the_format_constants", () => {
    const document = { ...baseDocument, diagnostics: { log_level: "info" },
        models: { "a-chat": { limits: { maximum_context_tokens: 4096 } } } };
    assert.equal(call("configurationValueCount", document), 3);
    assert.equal(call("configurationValueCount", { $schema: "x", schema_version: 1 }), 0);
});

test("should_show_the_value_in_force_in_a_field_and_leave_it_empty_only_without_one", () => {
    assert.equal(call("configurationFieldText", undefined, 2048), "2048");
    assert.equal(call("configurationFieldText", 512, 2048), "512");
    assert.equal(call("configurationFieldText", undefined, null), "");
    assert.equal(call("configurationFieldText", undefined, undefined), "");
    assert.equal(call("configurationPlaceholderText", { unsetText: "Full context" }), "Full context");
    assert.equal(call("configurationPlaceholderText", {}), "Automatic");
});

test("should_fall_back_to_a_real_folder_inside_the_instance_folder", () => {
    const descriptor = { path: ["runtime", "model_download_directory"], instanceFolderDefault: "models" };
    assert.deepEqual(call("configurationBaselineValue", bundledSchema, baseDocument, null, descriptor, "~/.example-instance"),
        { value: "~/.example-instance/models", isInherited: false });
    assert.equal(call("configurationBaselineValue", bundledSchema, baseDocument, null, descriptor, null).value, undefined);
});

test("should_offer_installed_chat_models_after_the_built_in_choice", () => {
    const descriptor = { builtinDefault: "built-in-model" };
    const discovered = [{ id: "b-chat", servesChat: true }, { id: "picture", servesChat: false }, { id: "a-chat", servesChat: true }];
    assert.deepEqual(call("configurationModelChoices", descriptor, discovered, undefined), [
        { value: "", label: "Built in (built-in-model)" },
        { value: "a-chat", label: "a-chat" }, { value: "b-chat", label: "b-chat" }]);
});

test("should_keep_a_saved_model_that_is_no_longer_installed_as_a_choice", () => {
    const choices = call("configurationModelChoices", { builtinDefault: "built-in-model" }, [{ id: "a-chat", servesChat: true }], "gone-model");
    assert.deepEqual(choices[1], { value: "gone-model", label: "gone-model (not installed)" });
});

test("should_say_what_an_unset_setting_does_in_the_tooltip", () => {
    assert.equal(call("configurationFallbackDescription", { builtinDefault: "built-in-model" }, undefined), "Default: built-in-model, built in");
    assert.equal(call("configurationFallbackDescription", { unsetText: "Automatic" }, undefined), "Default: automatic");
    assert.equal(call("configurationFallbackDescription", {}, undefined), "No default");
    assert.equal(call("configurationFallbackDescription", {}, []), "Default: none");
    assert.equal(call("configurationFallbackDescription", {}, false), "Default: off");
});

test("should_use_the_plain_labels_and_fallbacks_the_schema_gives_runtime_settings", () => {
    const descriptors = call("configurationPageGroups", bundledSchema, "settings").flatMap((group) => group.descriptors);
    const defaultModel = descriptors.find((descriptor) => descriptor.key === "default_model");
    assert.equal(defaultModel.choices, "chat-models");
    assert.equal(typeof defaultModel.builtinDefault, "string");
    assert.equal(descriptors.find((descriptor) => descriptor.key === "model_download_directory").instanceFolderDefault, "models");
});

test("should_edit_the_file_key_a_discovered_model_already_has_instead_of_a_second_row", () => {
    const document = { ...baseDocument, models: { "mlx-community/some-model": { generation_defaults: { temperature: 0.7 } } } };
    assert.equal(call("configurationEntryKeyForModel", document, "some-model"), "mlx-community/some-model");
    const entries = call("configurationTuningListEntries", [{ id: "some-model", servesChat: true }], document, []);
    assert.deepEqual(entries.map((entry) => entry.id), ["some-model"]);
    assert.equal(entries[0].settingCount, 1);
});

test("should_prefer_the_exact_folder_name_key_and_default_to_it_when_none_exists", () => {
    const document = { ...baseDocument, models: { "some-model": {}, "org/some-model": {} } };
    assert.equal(call("configurationEntryKeyForModel", document, "some-model"), "some-model");
    assert.equal(call("configurationEntryKeyForModel", baseDocument, "some-model"), "some-model");
    assert.equal(call("configurationEntryKeyForModel", { ...baseDocument, models: { "org/other": {} } }, "some-model"), "some-model");
});

test("should_report_edits_under_the_aliased_key_for_the_discovered_model", () => {
    const saved = { ...baseDocument, models: { "org/some-model": { limits: { maximum_context_tokens: 4096 } } } };
    const working = { ...baseDocument, models: { "org/some-model": { limits: { maximum_context_tokens: 8192 } } } };
    assert.equal(call("configurationModelHasUnsavedEdits", saved, working, "some-model"), true);
});

test("should_say_inherited_values_in_words_rather_than_raw_json", () => {
    const nullable = { applyMode: "model-reload", kind: "nullable_integer" };
    assert.match(call("configurationSettingNote", nullable, { value: null, isInherited: true }), /Inherits automatic/);
    const switchSetting = { applyMode: "model-reload", kind: "boolean" };
    assert.match(call("configurationSettingNote", switchSetting, { value: false, isInherited: true }), /Inherits off/);
});

test("should_count_no_values_when_the_file_could_not_be_read", () => {
    assert.equal(call("configurationValueCount", null), 0);
});

test("should_explain_every_section_so_its_info_button_has_something_to_say", () => {
    for (const pageName of ["settings", "tuning"]) {
        for (const group of call("configurationPageGroups", bundledSchema, pageName)) {
            assert.ok(group.description.length > 20, `${pageName}: ${group.title}`);
        }
    }
});

test("should_explain_every_setting_so_its_info_button_has_something_to_say", () => {
    const leaves = ["settings", "tuning"].flatMap((pageName) =>
        call("configurationPageGroups", bundledSchema, pageName).flatMap((group) => group.descriptors));
    for (const descriptor of leaves) {
        assert.ok(descriptor.description.length > 20, descriptor.path.join("."));
        assert.ok(descriptor.explanation && descriptor.explanation.length > 60,
            `${descriptor.path.join(".")} needs a plain explanation for someone new to the app`);
    }
});

test("should_hide_show_or_hover_the_reset_arrow_by_how_the_value_relates_to_its_baseline", () => {
    assert.equal(call("configurationResetArrowState", undefined, 4), "is-hidden");
    assert.equal(call("configurationResetArrowState", 8, 4), "is-visible");
    assert.equal(call("configurationResetArrowState", 4, 4), "is-hover-only");
    assert.equal(call("configurationResetArrowState", "/models", undefined), "is-visible");
});

const FOLDER_SETTING_PATHS = [
    "runtime.model_directories",
    "runtime.model_download_directory",
    "runtime.neural_engine_program_directory",
];

test("should_leave_out_a_group_whose_settings_the_instance_cannot_honour", () => {
    const groups = call("configurationPageGroups", bundledSchema, "settings", FOLDER_SETTING_PATHS);

    assert.deepEqual(groups.map((group) => group.title), ["Memory", "Prompt cache", "Diagnostics"],
        "no Locations group, and the folder settings do not reappear under Other");
});

test("should_keep_the_rest_of_a_group_when_only_some_of_its_settings_are_unavailable", () => {
    const groups = call("configurationPageGroups", bundledSchema, "settings", ["runtime.model_directories"]);
    const locationsGroup = groups.find((group) => group.title === "Locations");

    assert.deepEqual(locationsGroup.descriptors.map((descriptor) => descriptor.path.join(".")),
        ["runtime.model_download_directory", "runtime.neural_engine_program_directory"]);
});

test("should_show_every_group_when_nothing_is_unavailable_or_the_list_names_no_real_setting", () => {
    for (const unavailableSettingPaths of [undefined, [], ["runtime.setting_that_does_not_exist"]]) {
        const groupTitlesShown = call("configurationPageGroups", bundledSchema, "settings", unavailableSettingPaths)
            .map((group) => group.title);

        assert.deepEqual(groupTitlesShown, ["Memory", "Locations", "Prompt cache", "Diagnostics"]);
    }
});
