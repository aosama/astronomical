// Astronomical Observatory configuration editor: loads config.json through the
// daemon, keeps one working copy that the Settings and Tuning pages both edit, and
// saves or resets it. The pages themselves are drawn by configuration-pages.js from
// the rules in configuration-layout.js. All values enter the page through textContent
// and created elements, never as markup, because a setting could hold any text a
// hand edit put there.

const CONFIGURATION_URL = "/v1/config";
const CONFIGURATION_RESET_URL = "/v1/config/reset";
const CONFIGURATION_RELOAD_URL = "/v1/config/reload";
const CONFIGURATION_MODELS_URL = "/v1/models";
const CONFIGURATION_SUCCESS_NOTICE_MILLISECONDS = 6000;

// One working copy serves both pages, because the file is one document and one Save
// must cover an edit made on either. selectedModelIdentifier is undefined until the
// person (or the loaded model) picks one, null for the All models entry.
const configurationEditorState = {
    schema: null,
    savedDocument: null,
    workingDocument: null,
    documentDigest: null,
    discoveredModels: [],
    addedModelIdentifiers: [],
    selectedModelIdentifier: undefined,
    isAdvancedOpen: false,
    invalidSettings: new Map(),
    isBusy: false,
    neuralEngineStatusDocument: null,
    stateDirectory: null,
    unavailableSettingPaths: [],
};
let configurationFeedbackTimer = null;

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

function configurationHasUnsavedEdits() {
    return configurationEditorState.workingDocument !== null
        && !configurationDocumentsAreEqual(
            configurationEditorState.savedDocument, configurationEditorState.workingDocument);
}

function showConfigurationFeedback(tone, text) {
    const feedback = document.getElementById("configuration-feedback");
    feedback.className = `configuration-feedback feedback-${tone}`;
    feedback.textContent = text;
    feedback.hidden = !text;
    clearTimeout(configurationFeedbackTimer);
    if (text && tone === "success") {
        configurationFeedbackTimer = setTimeout(
            () => showConfigurationFeedback("success", ""), CONFIGURATION_SUCCESS_NOTICE_MILLISECONDS);
    }
    updateConfigurationDirtyState();
}

// The bar exists only while there is something to say: edits to save, a value to fix,
// or the outcome of the last save. At rest the page shows no bar at all.
function updateConfigurationSaveBar(summary) {
    const invalidSettings = [...configurationEditorState.invalidSettings.values()];
    const summaryText = configurationChangeSummaryText(summary);
    const hasEditsOrProblems = summary.changeCount > 0 || invalidSettings.length > 0;
    const feedbackIsShown = !document.getElementById("configuration-feedback").hidden;
    document.getElementById("configuration-savebar").hidden = !hasEditsOrProblems && !feedbackIsShown;
    document.getElementById("configuration-savebar-actions").hidden = !hasEditsOrProblems;
    document.getElementById("configuration-savebar-count").textContent =
        summary.changeCount > 0 ? summaryText.countText : "";
    document.getElementById("configuration-savebar-detail").textContent = summaryText.detailText;
    document.getElementById("configuration-savebar-problem").textContent =
        invalidSettings.length > 0 ? `Fix ${invalidSettings[0].label} to save` : "";
    document.getElementById("configuration-save").disabled =
        !hasEditsOrProblems || invalidSettings.length > 0 || configurationEditorState.isBusy;
    document.getElementById("configuration-discard").disabled = configurationEditorState.isBusy;
}

// A dot beside a page's name says that page holds edits the other page's Save bar
// will also save, since the two pages edit one file.
function updateConfigurationEditDots(summary) {
    document.querySelectorAll("[data-edit-dot]").forEach((dot) => {
        dot.hidden = !summary.editedPages[dot.dataset.editDot];
    });
}

function updateConfigurationDirtyState() {
    if (configurationEditorState.workingDocument === null) {
        // Nothing loaded: the bar only carries the reason (a file that could not be read).
        const feedbackIsShown = !document.getElementById("configuration-feedback").hidden;
        document.getElementById("configuration-savebar").hidden = !feedbackIsShown;
        document.getElementById("configuration-savebar-actions").hidden = true;
        return;
    }
    const summary = configurationChangeSummary(configurationEditorState.schema,
        configurationEditorState.savedDocument, configurationEditorState.workingDocument);
    updateConfigurationSaveBar(summary);
    updateConfigurationEditDots(summary);
    document.getElementById("configuration-preview").textContent =
        JSON.stringify(configurationEditorState.workingDocument, null, 2);
}

function setConfigurationWorkingDocument(updatedDocument) {
    configurationEditorState.workingDocument = updatedDocument;
    configurationEditorState.invalidSettings.clear();
    renderConfigurationPages();
    updateConfigurationDirtyState();
}

// Typing writes the working copy without a redraw, which would take the caret away;
// the page redraws when the entry is committed.
function commitConfigurationValue(descriptor, fullPath, newValue, baselineValue) {
    const savedFileHasValue = configurationValueAtPath(
        configurationEditorState.savedDocument, fullPath) !== undefined;
    configurationEditorState.workingDocument =
        configurationShouldRemoveValue(newValue, baselineValue, savedFileHasValue)
            ? withoutConfigurationValue(configurationEditorState.workingDocument, fullPath)
            : withConfigurationValue(configurationEditorState.workingDocument, fullPath, newValue);
    updateConfigurationDirtyState();
}

function removeConfigurationValues(fullPaths) {
    let updatedDocument = configurationEditorState.workingDocument;
    for (const fullPath of fullPaths) {
        updatedDocument = withoutConfigurationValue(updatedDocument, fullPath);
    }
    setConfigurationWorkingDocument(updatedDocument);
}

// Called when another control on the page changed config.json. Unsaved edits
// are left alone; the Save conflict check reports the newer file instead.
function refreshConfigurationDocumentIfUnedited() {
    if (!configurationHasUnsavedEdits()) {
        loadConfigurationDocument();
    }
}

async function fetchConfigurationDiscoveredModels() {
    try {
        const response = await fetch(CONFIGURATION_MODELS_URL);
        if (!response.ok) { return []; }
        const modelList = await response.json();
        return (modelList.data || []).map(configurationDiscoveredModelFromAdvertisement);
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
        configurationEditorState.unavailableSettingPaths = snapshot.unavailable_settings || [];
        configurationEditorState.documentDigest = snapshot.document_digest;
        configurationEditorState.discoveredModels = await fetchConfigurationDiscoveredModels();
        if (snapshot.document === null) {
            clearConfigurationPages();
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
    document.querySelectorAll("[data-configuration-action]").forEach((menuItem) => { menuItem.disabled = isBusy; });
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

async function sendConfigurationWrite(url, method, requestBody, progressText) {
    setConfigurationBusy(true);
    showConfigurationFeedback("progress", progressText);
    try {
        await finishConfigurationWrite(await fetch(url, {
            method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
        }));
    } catch (fetchError) {
        showConfigurationFeedback("error", "The local server did not answer; the file may not have changed.");
    } finally {
        setConfigurationBusy(false);
    }
}

function saveConfigurationDocument() {
    return sendConfigurationWrite(CONFIGURATION_URL, "PUT", {
        document: configurationEditorState.workingDocument,
        expected_document_digest: configurationEditorState.documentDigest,
    }, "Saving. A model may need to reload, which can take a while.");
}

function resetConfigurationDocument() {
    const valueCount = configurationValueCount(configurationEditorState.savedDocument);
    const subject = valueCount > 0 ? `all ${valueCount} ${valueCount === 1 ? "value" : "values"} in config.json` : "config.json";
    const isConfirmed = window.confirm(
        `Reset ${subject} to its defaults, including your model directories? Models in those folders stop appearing until you add them back. The current file is kept beside it as config.previous.json.`);
    if (!isConfirmed) { return Promise.resolve(); }
    return sendConfigurationWrite(CONFIGURATION_RESET_URL, "POST", {
        expected_document_digest: configurationEditorState.documentDigest,
    }, "Resetting.");
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

function resetSelectedConfigurationModel() {
    const modelIdentifier = configurationEditorState.selectedModelIdentifier;
    if (typeof modelIdentifier !== "string") { return; }
    removeConfigurationValues([["models",
        configurationEntryKeyForModel(configurationEditorState.workingDocument, modelIdentifier)]]);
}

// Folder settings show the real path they fall back to, which hangs off this instance's
// folder. The page is redrawn once when the folder is first learned, unless someone is
// typing in it.
function rememberConfigurationStateDirectory(statusDocument) {
    const stateDirectory = statusDocument.application ? statusDocument.application.state_directory : null;
    if (!stateDirectory || stateDirectory === configurationEditorState.stateDirectory) { return; }
    configurationEditorState.stateDirectory = stateDirectory;
    const someoneIsTyping = document.activeElement
        && document.activeElement.classList
        && document.activeElement.classList.contains("configuration-input");
    if (!someoneIsTyping) { renderConfigurationPages(); }
}

// Called on every status poll. The Neural Engine status line lives in that row's
// explanation for the loaded model only, so it costs the page nothing until someone
// asks what the toggle is doing.
function renderConfigurationStatus(statusDocument) {
    configurationEditorState.neuralEngineStatusDocument = statusDocument;
    rememberConfigurationStateDirectory(statusDocument);
    document.querySelectorAll("[data-neural-engine-status]").forEach((statusElement) => {
        statusElement.textContent = neuralEnginePrefillStatusMessage(statusDocument.neural_engine_prefill);
    });
}

const CONFIGURATION_MENU_ACTIONS = {
    preview: () => document.getElementById("configuration-preview-dialog").showModal(),
    apply: applySavedConfiguration,
    "reset-model": resetSelectedConfigurationModel,
    "reset-all": resetConfigurationDocument,
};

function closeConfigurationMenus() {
    document.querySelectorAll(".configuration-menu").forEach((menu) => { menu.hidden = true; });
}

function handleConfigurationMenuClick(clickEvent) {
    const menuButton = clickEvent.target.closest(".configuration-menu-button");
    const menuAction = clickEvent.target.closest("[data-configuration-action]");
    const wasOpen = menuButton && !menuButton.nextElementSibling.hidden;
    closeConfigurationMenus();
    if (menuButton && !wasOpen) { menuButton.nextElementSibling.hidden = false; }
    if (menuAction) { void CONFIGURATION_MENU_ACTIONS[menuAction.dataset.configurationAction](); }
}

function wireConfigurationEditor() {
    document.getElementById("configuration-save").addEventListener("click", saveConfigurationDocument);
    document.getElementById("configuration-discard").addEventListener("click", discardConfigurationEdits);
    document.getElementById("configuration-preview-close").addEventListener("click",
        () => document.getElementById("configuration-preview-dialog").close());
    document.addEventListener("click", handleConfigurationMenuClick);
    document.addEventListener("click", (clickEvent) => {
        if (!clickEvent.target.closest(".configuration-label-cell")) { closeConfigurationExplanations(); }
    });
    document.addEventListener("keydown", (keyEvent) => { if (keyEvent.key === "Escape") { closeConfigurationMenus(); closeConfigurationExplanations(); } });
    document.querySelectorAll('[data-observatory-destination="settings"], [data-observatory-destination="tuning"]')
        .forEach((navigationButton) => {
            // Moving between the two pages must keep unsaved edits, since they share one
            // working copy; only an unedited editor reloads from the file.
            navigationButton.addEventListener("click", refreshConfigurationDocumentIfUnedited);
        });
    document.addEventListener("observatory:config-file-written", refreshConfigurationDocumentIfUnedited);
    // Reads the views, not the address, so a retired address such as /configuration
    // that opens Settings loads the editor whatever ran first at startup.
    if (!document.getElementById("settings-view").hidden || !document.getElementById("tuning-view").hidden) {
        void loadConfigurationDocument();
    }
}
