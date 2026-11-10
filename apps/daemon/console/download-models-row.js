// Astronomical Observatory Models page: one row of the Add a model list. A row
// is one model; its builds are versions chosen with the picker, and the speed
// meter, the action and the details all follow the chosen version.

function createDownloadModelsModelRow(model) {
    const selectedVersion = downloadModelsSelectedVersion(
        model, downloadModelsSelectedVersionIdByModelKey, downloadModelsCurrentDownload.huggingface_id
    );
    const isExpanded = downloadModelsExpandedModelKey === model.modelKey;
    const row = createDownloadModelsElement(
        "article", isExpanded ? "download-models-row download-models-row-expanded" : "download-models-row"
    );
    const rowLine = createDownloadModelsElement("div", "download-models-row-line");
    const rowControls = createDownloadModelsElement("div", "download-models-row-controls");
    const fitSlot = createDownloadModelsElement("div", "download-models-row-fit");
    const fitMeter = createDownloadModelsFitMeter(selectedVersion);
    if (fitMeter) fitSlot.appendChild(fitMeter);
    const actionSlot = createDownloadModelsElement("div", "download-models-row-action");
    actionSlot.appendChild(createDownloadModelsRowAction(selectedVersion));
    rowControls.replaceChildren(createDownloadModelsVersionPicker(model, selectedVersion), fitSlot, actionSlot);
    rowLine.replaceChildren(createDownloadModelsRowIdentity(model, selectedVersion, isExpanded), rowControls);
    row.appendChild(rowLine);
    if (isExpanded) row.appendChild(createDownloadModelsRowDetails(selectedVersion));
    return row;
}

// The name is the control that opens the details, so the row needs no separate
// Details button competing with Get.
function createDownloadModelsRowIdentity(model, selectedVersion, isExpanded) {
    const identity = createDownloadModelsElement("div", "download-models-row-identity");
    const nameButton = createDownloadModelsElement("button", "download-models-row-name", model.name);
    nameButton.type = "button";
    nameButton.setAttribute("aria-expanded", String(isExpanded));
    nameButton.addEventListener("click", () => {
        downloadModelsExpandedModelKey = isExpanded ? null : model.modelKey;
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    });
    const identityChildren = [nameButton];
    const summaryText = selectedVersion.summary || selectedVersion.description;
    if (summaryText) {
        identityChildren.push(createDownloadModelsParagraph("download-models-row-summary", summaryText));
    }
    const abilitiesText = downloadModelsAbilitiesText(selectedVersion);
    if (abilitiesText) {
        identityChildren.push(createDownloadModelsParagraph("download-models-row-abilities", abilitiesText));
    }
    identity.replaceChildren(...identityChildren);
    return identity;
}

// A model with one build shows an empty slot of the same width, so the speed
// meters and buttons of every row line up in columns.
function createDownloadModelsVersionPicker(model, selectedVersion) {
    const pickerSlot = createDownloadModelsElement("div", "download-models-row-versions");
    if (model.versions.length < 2) return pickerSlot;
    const picker = createDownloadModelsElement("div", "download-models-version-picker");
    picker.setAttribute("role", "group");
    picker.setAttribute("aria-label", "Version of " + model.name);
    picker.replaceChildren(...model.versions.map(
        (version) => createDownloadModelsVersionButton(model, version, version === selectedVersion)
    ));
    pickerSlot.appendChild(picker);
    return pickerSlot;
}

function createDownloadModelsVersionButton(model, version, isSelected) {
    const versionButton = createDownloadModelsElement(
        "button",
        isSelected ? "download-models-version download-models-version-selected" : "download-models-version",
        version.versionLabel || formatDownloadModelsCardSize(version.approximateSizeBytes)
    );
    versionButton.type = "button";
    versionButton.setAttribute("aria-pressed", String(isSelected));
    versionButton.addEventListener("click", () => {
        downloadModelsSelectedVersionIdByModelKey = {
            ...downloadModelsSelectedVersionIdByModelKey,
            [model.modelKey]: version.huggingfaceId
        };
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    });
    return versionButton;
}

// Exactly one of: it is here, it is on its way, or Get. The controls of a
// download in flight live in its banner under On this Mac, not on the row.
function createDownloadModelsRowAction(selectedVersion) {
    if (selectedVersion.readyOnThisMac) {
        return createDownloadModelsElement("span", "download-models-row-status download-models-row-status-ready", "On this Mac");
    }
    const downloadState = downloadModelsRowDownloadState(selectedVersion);
    if (downloadState && downloadState !== "idle") {
        return createDownloadModelsElement(
            "span", "download-models-row-status", downloadModelsRowStatusText(selectedVersion, downloadState)
        );
    }
    return createDownloadModelsGetButton(selectedVersion);
}

function downloadModelsRowStatusText(selectedVersion, downloadState) {
    const isCurrentDownload = downloadModelsCurrentDownload.huggingface_id === selectedVersion.huggingfaceId;
    const percentText = isCurrentDownload && downloadModelsCurrentDownload.bytes_total > 0
        ? " · " + downloadModelsProgressPercent(downloadModelsCurrentDownload) + "%"
        : "";
    if (downloadState === "paused") return "Paused" + percentText;
    if (downloadState === "failed") return "Stopped";
    if (downloadState === "verifying" || downloadState === "publishing") return "Almost ready";
    if (downloadState === "downloading") return "Downloading" + percentText;
    return "Getting ready";
}

function createDownloadModelsRowDetails(selectedVersion) {
    const details = createDownloadModelsElement("div", "download-models-row-details");
    const prose = createDownloadModelsElement("div", "download-models-row-prose");
    const proseChildren = [];
    if (selectedVersion.description) {
        proseChildren.push(createDownloadModelsParagraph("download-models-row-description", selectedVersion.description));
    }
    const fitOutcome = downloadModelsFitOutcome(
        selectedVersion, downloadModelsServingCeilingBytes, downloadModelsCeilingIsUserLimit
    );
    if (fitOutcome !== null) {
        const fitCopy = DOWNLOAD_MODELS_FIT_COPY[fitOutcome];
        proseChildren.push(createDownloadModelsParagraph(
            "download-models-row-fit-explanation", fitCopy.label + ". " + fitCopy.explanation
        ));
    }
    prose.replaceChildren(...proseChildren);
    const facts = createDownloadModelsElement("div", "download-models-row-facts");
    const factChildren = [createDownloadModelsFactList(selectedVersion)];
    const locationActions = createDownloadModelsLocationActions(selectedVersion);
    if (locationActions) factChildren.push(locationActions);
    facts.replaceChildren(...factChildren);
    details.replaceChildren(prose, facts);
    return details;
}

// Everything technical lives here, one click below the plain row: the people
// who want the compression scheme or the Hugging Face name know to open it.
function downloadModelsFactEntries(selectedVersion) {
    const factEntries = [
        ["Download", formatDownloadModelsCardSize(selectedVersion.approximateSizeBytes)],
        ["Memory while running", formatDownloadModelsCardSize(selectedVersion.approximateMemoryBytes)]
    ];
    const rememberedWordsText = downloadModelsRememberedWordsText(selectedVersion.contextWindow);
    if (rememberedWordsText) factEntries.push(["Remembers", rememberedWordsText]);
    if (selectedVersion.quantizationLabel) factEntries.push(["Compression", selectedVersion.quantizationLabel]);
    if (selectedVersion.upstreamLicense) factEntries.push(["License", selectedVersion.upstreamLicense]);
    factEntries.push(["Full name", selectedVersion.displayName]);
    factEntries.push(["Published as", selectedVersion.huggingfaceId]);
    const destinationPath = downloadModelsDestinationPath(selectedVersion);
    if (destinationPath) {
        factEntries.push([selectedVersion.readyOnThisMac ? "Saved in" : "Will be saved in", destinationPath]);
    }
    return factEntries;
}

function createDownloadModelsFactList(selectedVersion) {
    const factList = createDownloadModelsElement("dl", "download-models-fact-list");
    for (const [factLabel, factText] of downloadModelsFactEntries(selectedVersion)) {
        factList.appendChild(createDownloadModelsElement("dt", "", factLabel));
        factList.appendChild(createDownloadModelsElement("dd", "", factText));
    }
    return factList;
}

// Finder can only show a folder that exists, so Show in Finder appears once the
// model is on this Mac; before that the path is where it will be saved.
function createDownloadModelsLocationActions(selectedVersion) {
    const destinationPath = downloadModelsDestinationPath(selectedVersion);
    if (!destinationPath) return null;
    const actionRow = createDownloadModelsElement("div", "download-models-location-actions");
    const copyButton = createDownloadModelsElement(
        "button", "button button-secondary download-models-copy-path-button", "Copy Path"
    );
    copyButton.type = "button";
    copyButton.addEventListener("click", () => copyDownloadModelsPathToClipboard(destinationPath));
    const actionButtons = selectedVersion.readyOnThisMac
        ? [createDownloadModelsRevealButton(selectedVersion.huggingfaceId), copyButton]
        : [copyButton];
    actionRow.replaceChildren(...actionButtons);
    return actionRow;
}

function createDownloadModelsRevealButton(huggingfaceId) {
    const revealButton = createDownloadModelsElement(
        "button", "button button-secondary download-models-reveal-button", "Show in Finder"
    );
    revealButton.type = "button";
    revealButton.addEventListener("click", async () => {
        const isRevealed = await revealDownloadModelsModelInFinder(huggingfaceId);
        if (!isRevealed) revealButton.textContent = "Could not open Finder";
    });
    return revealButton;
}

async function copyDownloadModelsPathToClipboard(destinationPath) {
    try {
        await navigator.clipboard.writeText(destinationPath);
    } catch {
        // Clipboard may be unavailable without a secure context or user gesture.
    }
}
