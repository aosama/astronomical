// Astronomical Observatory Configuration page: loads config.json through the
// daemon, edits a working copy, and saves or resets it. All values enter the page
// through textContent and created elements, never as markup, because a setting
// could hold any text a hand edit put there.

const CONFIGURATION_URL = "/v1/config";
const CONFIGURATION_RESET_URL = "/v1/config/reset";
const CONFIGURATION_RELOAD_URL = "/v1/config/reload";
const CONFIGURATION_MODELS_URL = "/v1/models";

const configurationEditorState = {
    schema: null,
    savedDocument: null,
    workingDocument: null,
    documentDigest: null,
    discoveredModelIdentifiers: [],
    isBusy: false,
};

function configurationElement(elementTag, className, textContent) {
    const createdElement = document.createElement(elementTag);
    if (className) { createdElement.className = className; }
    if (textContent !== undefined) { createdElement.textContent = textContent; }
    return createdElement;
}

function configurationButton(label, buttonStyle, onClick) {
    const button = configurationElement("button", `button ${buttonStyle}`, label);
    button.type = "button";
    button.addEventListener("click", onClick);
    return button;
}

function showConfigurationFeedback(tone, text) {
    const feedback = document.getElementById("configuration-feedback");
    feedback.className = `control-feedback feedback-${tone}`;
    feedback.textContent = text;
    feedback.hidden = !text;
}

function updateConfigurationDirtyState() {
    const hasUnsavedEdits = configurationEditorState.workingDocument !== null
        && !configurationDocumentsAreEqual(
            configurationEditorState.savedDocument, configurationEditorState.workingDocument);
    document.getElementById("configuration-save").disabled =
        !hasUnsavedEdits || configurationEditorState.isBusy;
    document.getElementById("configuration-discard").disabled =
        !hasUnsavedEdits || configurationEditorState.isBusy;
    document.getElementById("configuration-unsaved-note").hidden = !hasUnsavedEdits;
    document.getElementById("configuration-preview").textContent =
        JSON.stringify(configurationEditorState.workingDocument, null, 2);
}

function setConfigurationWorkingDocument(updatedDocument) {
    configurationEditorState.workingDocument = updatedDocument;
    renderConfigurationSections();
    updateConfigurationDirtyState();
}

function applyConfigurationInput(descriptor, rawText, messageElement) {
    const parsedInput = parseConfigurationInput(descriptor, rawText);
    messageElement.textContent = parsedInput.isValid ? "" : parsedInput.message;
    if (!parsedInput.isValid) { return; }
    configurationEditorState.workingDocument = withConfigurationValue(
        configurationEditorState.workingDocument, descriptor.path, parsedInput.value);
    updateConfigurationDirtyState();
}

function configurationBooleanControl(descriptor, currentValue) {
    const select = configurationElement("select", "configuration-input");
    select.setAttribute("aria-label", descriptor.label);
    for (const [optionValue, optionLabel] of [["", "Default"], ["true", "On"], ["false", "Off"]]) {
        const option = configurationElement("option", "", optionLabel);
        option.value = optionValue;
        select.append(option);
    }
    select.value = currentValue === undefined ? "" : String(currentValue);
    select.addEventListener("change", () => {
        setConfigurationWorkingDocument(select.value === ""
            ? withoutConfigurationValue(configurationEditorState.workingDocument, descriptor.path)
            : withConfigurationValue(configurationEditorState.workingDocument, descriptor.path,
                select.value === "true"));
    });
    return select;
}

function configurationEnumControl(descriptor, currentValue) {
    const select = configurationElement("select", "configuration-input");
    select.setAttribute("aria-label", descriptor.label);
    const defaultOption = configurationElement("option", "", "Default");
    defaultOption.value = "";
    select.append(defaultOption);
    for (const enumValue of descriptor.enumValues) {
        const option = configurationElement("option", "", enumValue);
        option.value = enumValue;
        select.append(option);
    }
    select.value = currentValue === undefined ? "" : currentValue;
    select.addEventListener("change", () => {
        setConfigurationWorkingDocument(select.value === ""
            ? withoutConfigurationValue(configurationEditorState.workingDocument, descriptor.path)
            : withConfigurationValue(configurationEditorState.workingDocument, descriptor.path,
                select.value));
    });
    return select;
}

function configurationTextControl(descriptor, currentValue, messageElement) {
    const input = configurationElement("input", "configuration-input");
    const isNumeric = descriptor.kind === "integer" || descriptor.kind === "number";
    input.type = isNumeric ? "number" : "text";
    input.setAttribute("aria-label", descriptor.label);
    if (descriptor.minimum !== undefined) { input.min = String(descriptor.minimum); }
    if (descriptor.maximum !== undefined) { input.max = String(descriptor.maximum); }
    input.value = currentValue === undefined || currentValue === null ? "" : String(currentValue);
    input.placeholder = configurationDefaultDescription(descriptor);
    input.addEventListener("input", () => {
        if (input.value === "") {
            messageElement.textContent = "";
            configurationEditorState.workingDocument = withoutConfigurationValue(
                configurationEditorState.workingDocument, descriptor.path);
            updateConfigurationDirtyState();
            return;
        }
        applyConfigurationInput(descriptor, input.value, messageElement);
    });
    // Redrawn on commit, not per keystroke, which would take the caret away. This
    // is what shows a Remove button for a value that was just set.
    input.addEventListener("change", () => {
        if (!messageElement.textContent) { renderConfigurationSections(); }
    });
    return input;
}

function configurationStringListControl(descriptor, currentValue) {
    const listContainer = configurationElement("div", "configuration-list");
    const entries = Array.isArray(currentValue) ? currentValue : [];
    entries.forEach((entry, entryIndex) => {
        const entryRow = configurationElement("div", "configuration-list-row");
        entryRow.append(configurationElement("code", "configuration-list-value", entry));
        entryRow.append(configurationButton("Remove", "button-secondary", () => {
            const remainingEntries = entries.filter((unused, candidateIndex) => candidateIndex !== entryIndex);
            setConfigurationWorkingDocument(withConfigurationValue(
                configurationEditorState.workingDocument, descriptor.path, remainingEntries));
        }));
        listContainer.append(entryRow);
    });
    const addRow = configurationElement("div", "configuration-list-row");
    const addInput = configurationElement("input", "configuration-input");
    addInput.type = "text";
    addInput.placeholder = "Absolute path";
    addInput.setAttribute("aria-label", `Add to ${descriptor.label}`);
    addRow.append(addInput);
    addRow.append(configurationButton("Add", "button-secondary", () => {
        const newEntry = addInput.value.trim();
        if (newEntry.length === 0) { return; }
        setConfigurationWorkingDocument(withConfigurationValue(
            configurationEditorState.workingDocument, descriptor.path, entries.concat(newEntry)));
    }));
    listContainer.append(addRow);
    return listContainer;
}

function configurationControl(descriptor, currentValue, messageElement) {
    if (descriptor.kind === "boolean") { return configurationBooleanControl(descriptor, currentValue); }
    if (descriptor.kind === "enum") { return configurationEnumControl(descriptor, currentValue); }
    if (descriptor.kind === "string_list") { return configurationStringListControl(descriptor, currentValue); }
    return configurationTextControl(descriptor, currentValue, messageElement);
}

function renderConfigurationSetting(descriptor) {
    const row = configurationElement("div", "configuration-setting");
    const currentValue = configurationValueAtPath(
        configurationEditorState.workingDocument, descriptor.path);
    const heading = configurationElement("div", "configuration-setting-heading");
    heading.append(configurationElement("span", "configuration-setting-label", descriptor.label));
    const applyModeDescription = configurationApplyModeDescription(descriptor.applyMode);
    if (applyModeDescription) {
        heading.append(configurationElement("span", "configuration-apply-mode", applyModeDescription));
    }
    row.append(heading);
    if (descriptor.description) {
        row.append(configurationElement("p", "configuration-setting-description", descriptor.description));
    }
    const messageElement = configurationElement("p", "configuration-setting-message");
    messageElement.setAttribute("role", "status");
    const controlRow = configurationElement("div", "configuration-control-row");
    controlRow.append(configurationControl(descriptor, currentValue, messageElement));
    if (currentValue !== undefined && !descriptor.isRequired) {
        controlRow.append(configurationButton("Remove", "button-secondary", () => {
            setConfigurationWorkingDocument(withoutConfigurationValue(
                configurationEditorState.workingDocument, descriptor.path));
        }));
    } else if (descriptor.kind !== "string_list") {
        controlRow.append(configurationElement("span", "configuration-default-note",
            configurationDefaultDescription(descriptor)));
    }
    row.append(controlRow);
    row.append(messageElement);
    return row;
}

function renderConfigurationGroup(descriptor) {
    const group = configurationElement("div", "configuration-group");
    group.append(configurationElement("h4", "", descriptor.label));
    for (const childDescriptor of descriptor.children) {
        group.append(renderConfigurationDescriptor(childDescriptor));
    }
    return group;
}

function renderConfigurationDescriptor(descriptor) {
    if (descriptor.kind === "group") { return renderConfigurationGroup(descriptor); }
    if (descriptor.kind === "keyed_objects") { return renderConfigurationKeyedObjects(descriptor); }
    return renderConfigurationSetting(descriptor);
}

function renderConfigurationModelAdder(descriptor, configuredIdentifiers) {
    const addRow = configurationElement("div", "configuration-control-row");
    const select = configurationElement("select", "configuration-input");
    select.setAttribute("aria-label", "Model to override");
    const availableIdentifiers = configurationEditorState.discoveredModelIdentifiers
        .filter((modelIdentifier) => !configuredIdentifiers.includes(modelIdentifier));
    for (const modelIdentifier of availableIdentifiers) {
        const option = configurationElement("option", "", modelIdentifier);
        option.value = modelIdentifier;
        select.append(option);
    }
    addRow.append(select);
    const addButton = configurationButton("Add model settings", "button-secondary", () => {
        if (!select.value) { return; }
        setConfigurationWorkingDocument(withConfigurationValue(
            configurationEditorState.workingDocument, descriptor.path.concat(select.value), {}));
    });
    addButton.disabled = availableIdentifiers.length === 0;
    addRow.append(addButton);
    return addRow;
}

function renderConfigurationKeyedObjects(descriptor) {
    const section = configurationElement("div", "configuration-group");
    if (descriptor.description) {
        section.append(configurationElement("p", "configuration-setting-description", descriptor.description));
    }
    const configuredEntries = configurationValueAtPath(
        configurationEditorState.workingDocument, descriptor.path) || {};
    const configuredIdentifiers = Object.keys(configuredEntries).sort();
    section.append(renderConfigurationModelAdder(descriptor, configuredIdentifiers));
    for (const modelIdentifier of configuredIdentifiers) {
        const entry = configurationElement("details", "configuration-model-entry");
        entry.open = true;
        const summary = configurationElement("summary", "", modelIdentifier);
        entry.append(summary);
        const entryPath = descriptor.path.concat(modelIdentifier);
        entry.append(configurationButton("Remove all settings for this model", "button-secondary", () => {
            setConfigurationWorkingDocument(withoutConfigurationValue(
                configurationEditorState.workingDocument, entryPath));
        }));
        const entryDescriptors = configurationSettingDescriptors(
            configurationEditorState.schema, descriptor.entryNode, entryPath);
        for (const entryDescriptor of entryDescriptors) {
            entry.append(renderConfigurationDescriptor(entryDescriptor));
        }
        section.append(entry);
    }
    return section;
}

function renderConfigurationSection(descriptor) {
    const section = configurationElement("section", "panel configuration-section");
    const heading = configurationElement("div", "panel-heading-row");
    heading.append(configurationElement("h3", "", descriptor.label));
    const canResetSection = configurationHasValueAtPath(
        configurationEditorState.workingDocument, descriptor.path) && !descriptor.isRequired;
    if (canResetSection) {
        heading.append(configurationButton("Reset section", "button-secondary", () => {
            setConfigurationWorkingDocument(withoutConfigurationValue(
                configurationEditorState.workingDocument, descriptor.path));
        }));
    }
    section.append(heading);
    if (descriptor.description && descriptor.kind === "group") {
        section.append(configurationElement("p", "panel-note", descriptor.description));
    }
    if (descriptor.kind === "group") {
        for (const childDescriptor of descriptor.children) {
            section.append(renderConfigurationDescriptor(childDescriptor));
        }
    } else {
        section.append(renderConfigurationDescriptor(descriptor));
    }
    return section;
}

function renderConfigurationSections() {
    const sectionsContainer = document.getElementById("configuration-sections");
    sectionsContainer.replaceChildren();
    const topLevelDescriptors = configurationSettingDescriptors(
        configurationEditorState.schema, configurationEditorState.schema, []);
    // The daemon's JSON sorts members, so the schema's order does not survive; the
    // section every document must have comes first because it is the one people need.
    const orderedDescriptors = topLevelDescriptors
        .filter((descriptor) => descriptor.isRequired)
        .concat(topLevelDescriptors.filter((descriptor) => !descriptor.isRequired));
    for (const descriptor of orderedDescriptors) {
        sectionsContainer.append(renderConfigurationSection(descriptor));
    }
}

async function fetchConfigurationModelIdentifiers() {
    try {
        const response = await fetch(CONFIGURATION_MODELS_URL);
        if (!response.ok) { return []; }
        const modelList = await response.json();
        return (modelList.data || []).map((model) => model.id).sort();
    } catch (fetchError) {
        return [];
    }
}

async function loadConfigurationDocument() {
    try {
        const response = await fetch(CONFIGURATION_URL);
        if (!response.ok) {
            showConfigurationFeedback("error", "This instance cannot edit its configuration.");
            return;
        }
        const snapshot = await response.json();
        configurationEditorState.schema = snapshot.schema;
        configurationEditorState.documentDigest = snapshot.document_digest;
        configurationEditorState.discoveredModelIdentifiers = await fetchConfigurationModelIdentifiers();
        if (snapshot.document === null) {
            document.getElementById("configuration-sections").replaceChildren();
            showConfigurationFeedback("error",
                `config.json could not be read: ${snapshot.read_error || "it is missing"}. Fix it by hand or reset it.`);
            return;
        }
        configurationEditorState.savedDocument = snapshot.document;
        setConfigurationWorkingDocument(cloneConfigurationDocument(snapshot.document));
    } catch (fetchError) {
        showConfigurationFeedback("error", "The local server did not answer.");
    }
}

function setConfigurationBusy(isBusy) {
    configurationEditorState.isBusy = isBusy;
    document.getElementById("configuration-reset-all").disabled = isBusy;
    updateConfigurationDirtyState();
}

// Reloads never discard a document just written; only a refusal keeps the edits.
async function finishConfigurationWrite(response) {
    const responseDocument = await response.json();
    if (!response.ok) {
        showConfigurationFeedback("error", responseDocument.message || "The configuration was refused.");
        return;
    }
    const saveOutcomeMessage = configurationSaveOutcomeMessage(responseDocument);
    await loadConfigurationDocument();
    showConfigurationFeedback(saveOutcomeMessage.tone, saveOutcomeMessage.text);
}

async function saveConfigurationDocument() {
    setConfigurationBusy(true);
    showConfigurationFeedback("progress", "Saving. A model may need to reload, which can take a while.");
    try {
        await finishConfigurationWrite(await fetch(CONFIGURATION_URL, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                document: configurationEditorState.workingDocument,
                expected_document_digest: configurationEditorState.documentDigest,
            }),
        }));
    } catch (fetchError) {
        showConfigurationFeedback("error", "The local server did not answer; the file may not have changed.");
    } finally {
        setConfigurationBusy(false);
    }
}

async function resetConfigurationDocument() {
    const isConfirmed = window.confirm(
        "Reset every setting to its default, including your model directories? Models in those folders stop appearing until you add them back. The current file is kept beside it as config.previous.json.");
    if (!isConfirmed) { return; }
    setConfigurationBusy(true);
    showConfigurationFeedback("progress", "Resetting.");
    try {
        await finishConfigurationWrite(await fetch(CONFIGURATION_RESET_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                expected_document_digest: configurationEditorState.documentDigest,
            }),
        }));
    } catch (fetchError) {
        showConfigurationFeedback("error", "The local server did not answer; the file may not have changed.");
    } finally {
        setConfigurationBusy(false);
    }
}

async function applySavedConfiguration() {
    setConfigurationBusy(true);
    showConfigurationFeedback("progress", "Applying the saved configuration.");
    try {
        const response = await fetch(CONFIGURATION_RELOAD_URL, { method: "POST" });
        const reload = await response.json();
        const saveOutcomeMessage = configurationSaveOutcomeMessage({
            reload_http_status: response.status, reload });
        showConfigurationFeedback(saveOutcomeMessage.tone, saveOutcomeMessage.text);
    } catch (fetchError) {
        showConfigurationFeedback("error", "The local server did not answer.");
    } finally {
        setConfigurationBusy(false);
    }
}

function discardConfigurationEdits() {
    setConfigurationWorkingDocument(cloneConfigurationDocument(configurationEditorState.savedDocument));
    showConfigurationFeedback("success", "");
}

function renderConfigurationNeuralEngineStatus(statusDocument) {
    const statusElement = document.getElementById("configuration-neural-engine-status");
    if (statusElement) {
        statusElement.textContent = neuralEnginePrefillStatusMessage(statusDocument.neural_engine_prefill);
    }
}

function wireConfigurationEditor() {
    document.getElementById("configuration-save").addEventListener("click", saveConfigurationDocument);
    document.getElementById("configuration-discard").addEventListener("click", discardConfigurationEdits);
    document.getElementById("configuration-reset-all").addEventListener("click", resetConfigurationDocument);
    document.getElementById("configuration-apply").addEventListener("click", applySavedConfiguration);
    document.querySelectorAll('[data-observatory-destination="configuration"]')
        .forEach((navigationButton) => {
            navigationButton.addEventListener("click", loadConfigurationDocument);
        });
    if (window.location.pathname === "/configuration") { void loadConfigurationDocument(); }
}
