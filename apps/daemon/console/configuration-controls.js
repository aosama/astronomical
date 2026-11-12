// Astronomical Observatory configuration controls: draws one setting as a row (a
// label and a control) and the group of rows under a heading. At rest a row shows
// nothing else; the plain explanation, the file key, the default and when a change
// takes effect open from the (i) button beside the name. Values enter the page through
// textContent only, never as markup.

const CONFIGURATION_NEURAL_ENGINE_PATH = "chunking.experimental_neural_engine_prefill_enabled";

// A held reference to the live Maximum model RAM control, because the page redraws
// its rows and the control owns listeners that memory-control.js wired once.
let configurationMemoryControlElement = null;
let configurationMemoryStatusElement = null;

function configurationFieldPath(descriptor, scope) {
    return configurationScopedPath(scope.modelIdentifier, descriptor.path);
}

function configurationRowContext(descriptor, scope) {
    const state = configurationEditorState;
    const fullPath = configurationFieldPath(descriptor, scope);
    const baseline = configurationBaselineValue(state.schema, state.workingDocument,
        scope.modelIdentifier, descriptor, state.stateDirectory);
    const currentValue = configurationValueAtPath(state.workingDocument, fullPath);
    const savedValue = configurationValueAtPath(state.savedDocument, fullPath);
    return { fullPath, baseline, currentValue,
        isEdited: canonicalConfigurationJson(currentValue) !== canonicalConfigurationJson(savedValue) };
}

function configurationSwitchControl(descriptor, context) {
    const isOn = context.currentValue !== undefined ? context.currentValue : context.baseline.value === true;
    const switchButton = configurationElement("button", "configuration-switch");
    switchButton.type = "button";
    switchButton.setAttribute("role", "switch");
    switchButton.setAttribute("aria-checked", String(isOn));
    switchButton.setAttribute("aria-label", descriptor.label);
    switchButton.addEventListener("click", () => {
        commitConfigurationValue(descriptor, context.fullPath, !isOn, context.baseline.value);
        renderConfigurationPages();
    });
    return switchButton;
}

function configurationSelectControl(descriptor, context) {
    const select = configurationElement("select", "configuration-input");
    select.setAttribute("aria-label", descriptor.label);
    const selectedValue = context.currentValue !== undefined ? context.currentValue : context.baseline.value;
    for (const enumValue of descriptor.enumValues) {
        const option = configurationElement("option", "", enumValue);
        option.value = enumValue;
        select.append(option);
    }
    select.value = selectedValue === undefined ? descriptor.enumValues[0] : selectedValue;
    select.addEventListener("change", () => {
        commitConfigurationValue(descriptor, context.fullPath, select.value, context.baseline.value);
        renderConfigurationPages();
    });
    return select;
}

function applyConfigurationTypedValue(descriptor, context, typedText, messageElement) {
    const invalidKey = context.fullPath.join(".");
    if (typedText.trim() === "") {
        configurationEditorState.invalidSettings.delete(invalidKey);
        messageElement.textContent = "";
        configurationEditorState.workingDocument = withoutConfigurationValue(
            configurationEditorState.workingDocument, context.fullPath);
        updateConfigurationDirtyState();
        return;
    }
    const parsedInput = parseConfigurationInput(descriptor, typedText);
    messageElement.textContent = parsedInput.isValid ? "" : parsedInput.message;
    if (!parsedInput.isValid) {
        configurationEditorState.invalidSettings.set(invalidKey,
            { label: descriptor.label, message: parsedInput.message, typedText });
        updateConfigurationDirtyState();
        return;
    }
    configurationEditorState.invalidSettings.delete(invalidKey);
    commitConfigurationValue(descriptor, context.fullPath, parsedInput.value, context.baseline.value);
}

function configurationInputControl(descriptor, context, messageElement, scope) {
    const field = configurationElement("span", "configuration-field");
    const input = configurationElement("input", "configuration-input");
    input.type = "text";
    input.setAttribute("inputmode", descriptor.kind === "string" ? "text" : "decimal");
    input.setAttribute("aria-label", descriptor.label);
    input.value = configurationFieldText(context.currentValue, context.baseline.value);
    input.placeholder = configurationPlaceholderText(descriptor);
    // A value the page refused stays on screen with its message, so a redraw triggered
    // elsewhere does not make the error disappear while Save is still blocked by it.
    const refusedEntry = configurationEditorState.invalidSettings.get(context.fullPath.join("."));
    if (refusedEntry) {
        input.value = refusedEntry.typedText;
        messageElement.textContent = refusedEntry.message;
    }
    input.addEventListener("input",
        () => applyConfigurationTypedValue(descriptor, context, input.value, messageElement));
    input.addEventListener("change", () => {
        if (!messageElement.textContent) {
            refreshConfigurationRowIndicators(input.closest(".configuration-row"), descriptor, scope);
        }
    });
    field.append(input);
    if (descriptor.unit) { field.append(configurationElement("span", "configuration-unit", descriptor.unit)); }
    field.classList.toggle("is-wide", descriptor.kind === "string");
    return field;
}

function configurationAddFolderControl(descriptor, context, entries, messageElement) {
    const addButton = configurationElement("button", "configuration-text-button", "+ Add folder");
    addButton.type = "button";
    addButton.addEventListener("click", () => {
        const input = configurationElement("input", "configuration-input");
        input.placeholder = "Absolute path, then Enter";
        input.setAttribute("aria-label", `Add to ${descriptor.label}`);
        input.addEventListener("input", () => { messageElement.textContent = ""; });
        input.addEventListener("keydown", (keyEvent) => {
            if (keyEvent.key === "Escape") { renderConfigurationPages(); }
            if (keyEvent.key !== "Enter") { return; }
            const newEntry = input.value.trim();
            if (!newEntry.startsWith("/")) { messageElement.textContent = "Enter an absolute path"; return; }
            commitConfigurationValue(descriptor, context.fullPath, entries.concat(newEntry), undefined);
            renderConfigurationPages();
        });
        addButton.replaceWith(input);
        input.focus();
    });
    return addButton;
}

function configurationStringListControl(descriptor, context, messageElement) {
    const listContainer = configurationElement("div", "configuration-paths");
    const entries = Array.isArray(context.currentValue) ? context.currentValue : [];
    entries.forEach((entry, entryIndex) => {
        const entryRow = configurationElement("div", "configuration-path");
        entryRow.append(configurationElement("code", "", entry));
        const removeButton = configurationElement("button", "configuration-path-remove", "✕");
        removeButton.type = "button";
        removeButton.setAttribute("aria-label", `Remove ${entry}`);
        removeButton.addEventListener("click", () => {
            commitConfigurationValue(descriptor, context.fullPath,
                entries.filter((unused, candidateIndex) => candidateIndex !== entryIndex), undefined);
            renderConfigurationPages();
        });
        entryRow.append(removeButton);
        listContainer.append(entryRow);
    });
    listContainer.append(configurationAddFolderControl(descriptor, context, entries, messageElement));
    return listContainer;
}

function configurationModelChoiceControl(descriptor, context) {
    const select = configurationElement("select", "configuration-input");
    select.setAttribute("aria-label", descriptor.label);
    const choices = configurationModelChoices(descriptor,
        configurationEditorState.discoveredModels, context.currentValue);
    for (const choice of choices) {
        const option = configurationElement("option", "", choice.label);
        option.value = choice.value;
        select.append(option);
    }
    select.value = context.currentValue === undefined ? "" : context.currentValue;
    select.addEventListener("change", () => {
        if (select.value === "") {
            removeConfigurationValues([context.fullPath]);
            return;
        }
        commitConfigurationValue(descriptor, context.fullPath, select.value, undefined);
        renderConfigurationPages();
    });
    return select;
}

function configurationControlFor(descriptor, context, messageElement, scope) {
    if (descriptor.choices === "chat-models") { return configurationModelChoiceControl(descriptor, context); }
    if (descriptor.kind === "boolean") { return configurationSwitchControl(descriptor, context); }
    if (descriptor.kind === "enum") { return configurationSelectControl(descriptor, context); }
    if (descriptor.kind === "string_list") { return configurationStringListControl(descriptor, context, messageElement); }
    return configurationInputControl(descriptor, context, messageElement, scope);
}

function configurationExplanation(descriptor, scope, context) {
    const explanation = configurationElement("div", "configuration-tip");
    explanation.append(configurationElement("p", "", descriptor.explanation || descriptor.description));
    explanation.append(configurationElement("code", "", descriptor.path.join(".")));
    explanation.append(configurationElement("p", "configuration-tip-note",
        configurationSettingNote(descriptor, context.baseline)));
    if (descriptor.path.join(".") === CONFIGURATION_NEURAL_ENGINE_PATH && scope.isLoadedModel === true) {
        const statusLine = configurationElement("p", "configuration-tip-note");
        statusLine.dataset.neuralEngineStatus = "";
        statusLine.textContent = neuralEnginePrefillStatusMessage(
            configurationEditorState.neuralEngineStatusDocument
                ? configurationEditorState.neuralEngineStatusDocument.neural_engine_prefill : null);
        explanation.append(statusLine);
    }
    return explanation;
}

// The (i) button beside a name opens its explanation and closes any other one, so a
// person never has to guess that a name can be hovered.
function configurationInfoButton(subjectLabel, labelCell) {
    const infoButton = configurationElement("button", "configuration-info", "i");
    infoButton.type = "button";
    infoButton.setAttribute("aria-label", `About ${subjectLabel}`);
    infoButton.setAttribute("aria-expanded", "false");
    infoButton.addEventListener("click", () => {
        const willOpen = !labelCell.classList.contains("is-open");
        closeConfigurationExplanations();
        labelCell.classList.toggle("is-open", willOpen);
        infoButton.setAttribute("aria-expanded", String(willOpen));
    });
    return infoButton;
}

function closeConfigurationExplanations() {
    document.querySelectorAll(".configuration-label-cell.is-open").forEach((openCell) => {
        openCell.classList.remove("is-open");
        openCell.querySelector(".configuration-info").setAttribute("aria-expanded", "false");
    });
}

// A typed value updates its own row in place. Redrawing the page on blur would replace
// the switch or list entry the pointer is already pressing, and that click would be lost.
function refreshConfigurationRowIndicators(rowElement, descriptor, scope) {
    const context = configurationRowContext(descriptor, scope);
    rowElement.classList.toggle("is-edited", context.isEdited);
    rowElement.querySelector(".configuration-reset").className =
        `configuration-reset ${configurationResetArrowState(context.currentValue, context.baseline.value)}`;
    // A field someone emptied goes back to showing the value now in force.
    const input = rowElement.querySelector("input.configuration-input");
    if (input && input.value.trim() === "") {
        input.value = configurationFieldText(context.currentValue, context.baseline.value);
    }
}

function configurationResetButton(descriptor, context) {
    const resetButton = configurationElement("button", `configuration-reset ${configurationResetArrowState(context.currentValue, context.baseline.value)}`, "↺");
    resetButton.type = "button";
    resetButton.title = "Back to default";
    resetButton.setAttribute("aria-label", `Reset ${descriptor.label}`);
    resetButton.addEventListener("click", () => removeConfigurationValues([context.fullPath]));
    return resetButton;
}

function renderConfigurationRow(descriptor, scope) {
    const context = configurationRowContext(descriptor, scope);
    const row = configurationElement("div", "configuration-row");
    row.classList.toggle("is-edited", context.isEdited);
    const labelCell = configurationElement("div", "configuration-label-cell");
    labelCell.append(configurationElement("span", "configuration-label", descriptor.label),
        configurationInfoButton(descriptor.label, labelCell),
        configurationExplanation(descriptor, scope, context));
    const messageElement = configurationElement("p", "configuration-row-message");
    messageElement.setAttribute("role", "status");
    const controlCell = configurationElement("div", "configuration-control-cell");
    const isMemoryRow = hasDedicatedConfigurationControl(descriptor) && configurationMemoryControlElement;
    controlCell.append(isMemoryRow
        ? configurationMemoryControlElement
        : configurationControlFor(descriptor, context, messageElement, scope));
    // "Up to N GB on this Mac" is for whoever asks, so it moves into the explanation.
    if (isMemoryRow && configurationMemoryStatusElement) {
        labelCell.querySelector(".configuration-tip").append(configurationMemoryStatusElement);
    }
    row.append(labelCell, controlCell);
    row.append(isMemoryRow ? configurationElement("span") : configurationResetButton(descriptor, context));
    row.append(messageElement);
    return row;
}

function configurationGroupHasValues(group, scope) {
    return group.descriptors.some((descriptor) => configurationValueAtPath(
        configurationEditorState.workingDocument, configurationFieldPath(descriptor, scope)) !== undefined);
}

function renderConfigurationGroup(group, scope) {
    const section = configurationElement("section", "configuration-section");
    const heading = configurationElement("div", "configuration-section-heading");
    const titleCell = configurationElement("div", "configuration-label-cell");
    titleCell.append(configurationElement("h3", "", group.title));
    if (group.description) {
        const explanation = configurationElement("div", "configuration-tip");
        explanation.append(configurationElement("p", "", group.description));
        titleCell.append(configurationInfoButton(group.title, titleCell), explanation);
    }
    heading.append(titleCell);
    if (group.note) { heading.append(configurationElement("span", "configuration-section-note", group.note)); }
    if (configurationGroupHasValues(group, scope)) {
        const resetPaths = group.descriptors.map((descriptor) => configurationFieldPath(descriptor, scope));
        const resetSectionButton = configurationElement("button", "configuration-section-reset", "Reset section");
        resetSectionButton.type = "button";
        resetSectionButton.addEventListener("click", () => removeConfigurationValues(resetPaths));
        heading.append(resetSectionButton);
    }
    section.append(heading);
    group.descriptors.forEach((descriptor) => section.append(renderConfigurationRow(descriptor, scope)));
    return section;
}
