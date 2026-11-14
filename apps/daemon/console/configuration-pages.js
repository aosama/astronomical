// Astronomical Observatory configuration pages: draws the Settings page (settings
// that apply to the whole instance) and the Tuning page (a model list beside the
// chosen model's settings) from the schema's layout. Both pages read and write the
// one working copy owned by configuration.js, so an edit on either is saved together.

const CONFIGURATION_ALL_MODELS_SCOPE = null;
const CONFIGURATION_ADVANCED_TIERS = ["advanced", "diagnostic"];

// The memory control is static markup that memory-control.js wired once, and the page
// redraws its rows. The control is lifted out on first draw and set back into each new
// row, so its listeners and element ids survive every redraw.
function captureConfigurationMemoryControl() {
    if (configurationMemoryControlElement) { return; }
    const holder = document.getElementById("configuration-memory-control-holder");
    if (!holder) { return; }
    configurationMemoryControlElement = holder.firstElementChild;
    configurationMemoryStatusElement = configurationMemoryControlElement.querySelector("#maximum-mlx-memory-status");
}

function clearConfigurationPages() {
    document.getElementById("settings-groups").replaceChildren();
    document.getElementById("tuning-list").replaceChildren();
    document.getElementById("tuning-detail").replaceChildren();
}

function renderSettingsPage() {
    const groupsContainer = document.getElementById("settings-groups");
    groupsContainer.replaceChildren();
    const groups = configurationPageGroups(configurationEditorState.schema, "settings",
        configurationEditorState.unavailableSettingPaths);
    for (const group of groups) {
        groupsContainer.append(renderConfigurationGroup(group,
            { modelIdentifier: CONFIGURATION_ALL_MODELS_SCOPE, isLoadedModel: false }));
    }
}

function configurationTuningEntries() {
    const state = configurationEditorState;
    return configurationTuningListEntries(state.discoveredModels, state.workingDocument,
        state.addedModelIdentifiers);
}

// Opening Tuning with a model loaded lands on that model, because it is the one a
// person is most likely to be tuning; otherwise on the first model that generates text.
function resolveSelectedConfigurationModel(entries) {
    const selected = configurationEditorState.selectedModelIdentifier;
    if (selected === CONFIGURATION_ALL_MODELS_SCOPE || entries.some((entry) => entry.id === selected)) {
        return selected;
    }
    const loadedEntry = typeof loadedModelId !== "undefined"
        ? entries.find((entry) => entry.id === loadedModelId && entry.servesChat) : undefined;
    const firstChatEntry = entries.find((entry) => entry.servesChat);
    return (loadedEntry || firstChatEntry || { id: CONFIGURATION_ALL_MODELS_SCOPE }).id;
}

function tuningListButton(label, isSelected, onSelect) {
    const button = configurationElement("button", "tuning-list-item");
    button.type = "button";
    button.setAttribute("aria-current", String(isSelected));
    button.append(configurationElement("span", "", label));
    button.addEventListener("click", onSelect);
    return button;
}

function selectConfigurationModel(modelIdentifier) {
    configurationEditorState.selectedModelIdentifier = modelIdentifier;
    renderTuningPage();
}

function tuningEntryMark(entry) {
    const state = configurationEditorState;
    if (configurationModelHasUnsavedEdits(state.savedDocument, state.workingDocument, entry.id)) {
        const editDot = configurationElement("i", "tuning-list-edit-dot");
        editDot.title = "Unsaved changes";
        return editDot;
    }
    return entry.settingCount > 0 ? configurationElement("span", "tuning-list-count", String(entry.settingCount)) : null;
}

function tuningAddModelControl() {
    const addButton = configurationElement("button", "tuning-list-add", "+ Add by model ID");
    addButton.type = "button";
    addButton.addEventListener("click", () => {
        const input = configurationElement("input", "configuration-input");
        input.placeholder = "Model ID, then Enter";
        input.setAttribute("aria-label", "Add settings for a model by its ID");
        input.addEventListener("keydown", (keyEvent) => {
            if (keyEvent.key === "Escape") { renderTuningPage(); }
            const modelIdentifier = input.value.trim();
            if (keyEvent.key !== "Enter" || modelIdentifier === "") { return; }
            configurationEditorState.addedModelIdentifiers.push(modelIdentifier);
            selectConfigurationModel(modelIdentifier);
        });
        addButton.replaceWith(input);
        input.focus();
    });
    return addButton;
}

function renderTuningList(entries, selectedIdentifier) {
    const listContainer = document.getElementById("tuning-list");
    listContainer.replaceChildren();
    const allModelsButton = tuningListButton("All models",
        selectedIdentifier === CONFIGURATION_ALL_MODELS_SCOPE,
        () => selectConfigurationModel(CONFIGURATION_ALL_MODELS_SCOPE));
    allModelsButton.classList.add("is-all-models");
    listContainer.append(allModelsButton);
    for (const entry of entries) {
        const button = tuningListButton(entry.id, entry.id === selectedIdentifier,
            () => selectConfigurationModel(entry.id));
        button.classList.toggle("is-muted", !entry.servesChat);
        const mark = tuningEntryMark(entry);
        if (mark) { button.append(mark); }
        listContainer.append(button);
    }
    listContainer.append(tuningAddModelControl());
}

function tuningAdvancedToggle() {
    const toggle = configurationElement("button", "tuning-advanced-toggle", "Advanced");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", String(configurationEditorState.isAdvancedOpen));
    toggle.addEventListener("click", () => {
        configurationEditorState.isAdvancedOpen = !configurationEditorState.isAdvancedOpen;
        renderTuningPage();
    });
    return toggle;
}

function renderAllModelsDetail(detailContainer, tuningGroups) {
    detailContainer.append(configurationElement("h2", "tuning-title is-prose", "All models"));
    detailContainer.append(configurationElement("p", "tuning-subtitle",
        "Every model starts from these. A model's own value wins."));
    const scope = { modelIdentifier: CONFIGURATION_ALL_MODELS_SCOPE, isLoadedModel: false };
    for (const group of configurationGroupsForAllModels(configurationEditorState.schema, tuningGroups)) {
        detailContainer.append(renderConfigurationGroup(group, scope));
    }
}

function renderModelDetail(detailContainer, entry, tuningGroups) {
    detailContainer.append(configurationElement("h2", "tuning-title", entry.id));
    if (!entry.servesChat && entry.settingCount === 0) {
        detailContainer.append(configurationElement("p", "tuning-empty",
            "Nothing to tune. This model does not generate text."));
        return;
    }
    const isLoadedModel = typeof loadedModelId !== "undefined" && entry.id === loadedModelId;
    const scope = { modelIdentifier: configurationEntryKeyForModel(configurationEditorState.workingDocument, entry.id),
        isLoadedModel };
    const isAdvancedGroup = (group) => CONFIGURATION_ADVANCED_TIERS.includes(group.tier);
    tuningGroups.filter((group) => !isAdvancedGroup(group))
        .forEach((group) => detailContainer.append(renderConfigurationGroup(group, scope)));
    detailContainer.append(tuningAdvancedToggle());
    if (!configurationEditorState.isAdvancedOpen) { return; }
    tuningGroups.filter(isAdvancedGroup)
        .forEach((group) => detailContainer.append(renderConfigurationGroup(group, scope)));
}

function renderTuningPage() {
    const entries = configurationTuningEntries();
    const selectedIdentifier = resolveSelectedConfigurationModel(entries);
    configurationEditorState.selectedModelIdentifier = selectedIdentifier;
    renderTuningList(entries, selectedIdentifier);
    const detailContainer = document.getElementById("tuning-detail");
    detailContainer.replaceChildren();
    const tuningGroups = configurationPageGroups(configurationEditorState.schema, "tuning");
    const isAllModels = selectedIdentifier === CONFIGURATION_ALL_MODELS_SCOPE;
    document.getElementById("tuning-reset-model").hidden = isAllModels
        || !entries.some((entry) => entry.id === selectedIdentifier && entry.settingCount > 0);
    if (isAllModels) {
        renderAllModelsDetail(detailContainer, tuningGroups);
        return;
    }
    const selectedEntry = entries.find((entry) => entry.id === selectedIdentifier);
    renderModelDetail(detailContainer, selectedEntry, tuningGroups);
}

function renderConfigurationPages() {
    if (configurationEditorState.schema === null || configurationEditorState.workingDocument === null) { return; }
    captureConfigurationMemoryControl();
    renderSettingsPage();
    renderTuningPage();
}
