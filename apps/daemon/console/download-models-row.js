// Astronomical Observatory Models page: one row of the Add a model list. A row
// is one model, named by its model ID; the speed meter, the action and the
// details all describe that one download.

function createDownloadModelsModelRow(catalogRow) {
    const isExpanded = downloadModelsExpandedModelId === catalogRow.modelId;
    const row = createDownloadModelsElement(
        "article", isExpanded ? "download-models-row download-models-row-expanded" : "download-models-row"
    );
    const rowLine = createDownloadModelsElement("div", "download-models-row-line");
    const rowControls = createDownloadModelsElement("div", "download-models-row-controls");
    const fitSlot = createDownloadModelsElement("div", "download-models-row-fit");
    const fitMeter = createDownloadModelsFitMeter(catalogRow);
    if (fitMeter) fitSlot.appendChild(fitMeter);
    const actionSlot = createDownloadModelsElement("div", "download-models-row-action");
    actionSlot.appendChild(createDownloadModelsRowAction(catalogRow));
    rowControls.replaceChildren(fitSlot, actionSlot);
    rowLine.replaceChildren(createDownloadModelsRowIdentity(catalogRow, isExpanded), rowControls);
    row.appendChild(rowLine);
    if (isExpanded) row.appendChild(createDownloadModelsRowDetails(catalogRow));
    return row;
}

// The name is the control that opens the details, so the row needs no separate
// Details button competing with Get.
function createDownloadModelsRowIdentity(catalogRow, isExpanded) {
    const identity = createDownloadModelsElement("div", "download-models-row-identity");
    const nameButton = createDownloadModelsElement("button", "download-models-row-name", downloadModelsRowName(catalogRow));
    nameButton.type = "button";
    nameButton.setAttribute("aria-expanded", String(isExpanded));
    nameButton.addEventListener("click", () => {
        downloadModelsExpandedModelId = isExpanded ? null : catalogRow.modelId;
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    });
    const identityChildren = [nameButton];
    const summaryText = catalogRow.summary || catalogRow.description;
    if (summaryText) {
        identityChildren.push(createDownloadModelsParagraph("download-models-row-summary", summaryText));
    }
    const abilitiesText = downloadModelsAbilitiesText(catalogRow);
    if (abilitiesText) {
        identityChildren.push(createDownloadModelsParagraph("download-models-row-abilities", abilitiesText));
    }
    identity.replaceChildren(...identityChildren);
    return identity;
}

// Exactly one of: it is here, it is on its way, or Get. The controls of a
// download in flight live in its banner under On this Mac, not on the row.
function createDownloadModelsRowAction(catalogRow) {
    if (catalogRow.readyOnThisMac) {
        return createDownloadModelsElement("span", "download-models-row-status download-models-row-status-ready", "On this Mac");
    }
    const downloadState = downloadModelsRowDownloadState(catalogRow);
    if (downloadState && downloadState !== "idle") {
        return createDownloadModelsElement(
            "span", "download-models-row-status", downloadModelsRowStatusText(catalogRow, downloadState)
        );
    }
    return createDownloadModelsGetButton(catalogRow);
}

function downloadModelsRowStatusText(catalogRow, downloadState) {
    const isCurrentDownload = downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const percentText = isCurrentDownload && downloadModelsCurrentDownload.bytes_total > 0
        ? " · " + downloadModelsProgressPercent(downloadModelsCurrentDownload) + "%"
        : "";
    if (downloadState === "paused") return "Paused" + percentText;
    if (downloadState === "failed") return "Stopped";
    if (downloadState === "verifying" || downloadState === "publishing") return "Almost ready";
    if (downloadState === "downloading") return "Downloading" + percentText;
    return "Getting ready";
}

function createDownloadModelsRowDetails(catalogRow) {
    const details = createDownloadModelsElement("div", "download-models-row-details");
    const prose = createDownloadModelsElement("div", "download-models-row-prose");
    const proseChildren = [];
    if (catalogRow.description) {
        proseChildren.push(createDownloadModelsParagraph("download-models-row-description", catalogRow.description));
    }
    const fitOutcome = downloadModelsFitOutcome(
        catalogRow, downloadModelsServingCeilingBytes, downloadModelsCeilingIsUserLimit
    );
    if (fitOutcome !== null) {
        const fitCopy = DOWNLOAD_MODELS_FIT_COPY[fitOutcome];
        proseChildren.push(createDownloadModelsParagraph(
            "download-models-row-fit-explanation", fitCopy.label + ". " + fitCopy.explanation
        ));
    }
    prose.replaceChildren(...proseChildren);
    const facts = createDownloadModelsElement("div", "download-models-row-facts");
    const factChildren = [createDownloadModelsFactList(catalogRow)];
    const locationActions = createDownloadModelsLocationActions(catalogRow);
    if (locationActions) factChildren.push(locationActions);
    facts.replaceChildren(...factChildren);
    details.replaceChildren(prose, facts);
    return details;
}

// Everything technical lives here, one click below the plain row: the people
// who want the compression scheme or where the files come from know to open it.
// The Hugging Face identity is shown as a source, not as another name.
function downloadModelsFactEntries(catalogRow) {
    const factEntries = [
        ["Download", formatDownloadModelsCardSize(catalogRow.approximateSizeBytes)],
        ["Memory while running", formatDownloadModelsCardSize(catalogRow.approximateMemoryBytes)]
    ];
    const rememberedWordsText = downloadModelsRememberedWordsText(catalogRow.contextWindow);
    if (rememberedWordsText) factEntries.push(["Remembers", rememberedWordsText]);
    if (catalogRow.quantizationLabel) factEntries.push(["Compression", catalogRow.quantizationLabel]);
    if (catalogRow.upstreamLicense) factEntries.push(["License", catalogRow.upstreamLicense]);
    factEntries.push(["Downloaded from", catalogRow.huggingfaceId]);
    const destinationPath = downloadModelsDestinationPath(catalogRow);
    if (destinationPath) {
        factEntries.push([catalogRow.readyOnThisMac ? "Saved in" : "Will be saved in", destinationPath]);
    }
    return factEntries;
}

function createDownloadModelsFactList(catalogRow) {
    const factList = createDownloadModelsElement("dl", "download-models-fact-list");
    for (const [factLabel, factText] of downloadModelsFactEntries(catalogRow)) {
        factList.appendChild(createDownloadModelsElement("dt", "", factLabel));
        factList.appendChild(createDownloadModelsElement("dd", "", factText));
    }
    return factList;
}

// Finder can only show a folder that exists, so Show in Finder appears once the
// model is on this Mac; before that the path is where it will be saved.
function createDownloadModelsLocationActions(catalogRow) {
    const destinationPath = downloadModelsDestinationPath(catalogRow);
    if (!destinationPath) return null;
    const actionRow = createDownloadModelsElement("div", "download-models-location-actions");
    const copyButton = createDownloadModelsElement(
        "button", "button button-secondary download-models-copy-path-button", "Copy Path"
    );
    copyButton.type = "button";
    copyButton.addEventListener("click", () => copyDownloadModelsPathToClipboard(destinationPath));
    const actionButtons = catalogRow.readyOnThisMac
        ? [createDownloadModelsRevealButton(catalogRow.huggingfaceId), copyButton]
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
