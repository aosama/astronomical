// Astronomical Observatory Library rendering engine.
// Owns model card construction, capability badges, download hero, expandable
// details, hero summary, and filter bar. All catalog values enter the document
// only through textContent and created elements.

function libraryFamilyLabel(family) {
    if (family === "qwen3_5") return "Qwen 3.5";
    if (family === "laguna") return "Laguna";
    if (family === "flux2_klein") return "FLUX.2 Klein";
    return family;
}

function formatLibraryContextTokens(tokenCount) {
    if (!tokenCount || tokenCount <= 0) return null;
    if (tokenCount >= 1_000_000) {
        return (tokenCount / 1_000_000).toFixed(0) + "M context";
    }
    if (tokenCount >= 1_000) {
        return (tokenCount / 1_000).toFixed(0) + "K context";
    }
    return tokenCount + " context";
}

function createLibraryHeroSummary(catalogRows) {
    const summary = document.createElement("div");
    summary.className = "library-hero-summary";
    const readyCount = catalogRows.filter((row) => row.readyOnThisMac).length;
    const downloadableCount = catalogRows.length - readyCount;
    const totalSizeBytes = catalogRows.reduce(
        (total, row) => total + (row.approximateSizeBytes || 0), 0
    );
    const headline = document.createElement("p");
    headline.className = "library-hero-headline";
    const readySpan = document.createElement("span");
    readySpan.className = "library-hero-ready-count";
    readySpan.textContent = readyCount + " model" + (readyCount === 1 ? "" : "s") + " on this Mac";
    const downloadSpan = document.createElement("span");
    downloadSpan.className = "library-hero-downloadable-count";
    downloadSpan.textContent = " · " + downloadableCount + " ready to download";
    headline.replaceChildren(readySpan, downloadSpan);
    summary.appendChild(headline);
    if (totalSizeBytes > 0 && readyCount > 0) {
        const sizeNote = document.createElement("p");
        sizeNote.className = "library-hero-size";
        sizeNote.textContent = formatLibrarySizeGigabytes(totalSizeBytes) + " total across all catalog models";
        summary.appendChild(sizeNote);
    }
    return summary;
}

function createLibraryFilterBar(catalogRows) {
    const bar = document.createElement("div");
    bar.className = "library-filter-bar";
    const searchInput = document.createElement("input");
    searchInput.id = "library-search";
    searchInput.type = "search";
    searchInput.placeholder = "Search models";
    searchInput.value = librarySearchQuery;
    searchInput.setAttribute("aria-label", "Search models");
    const readinessFilter = document.createElement("select");
    readinessFilter.id = "library-readiness-filter";
    readinessFilter.setAttribute("aria-label", "Filter by readiness");
    const readinessAll = document.createElement("option");
    readinessAll.value = "all";
    readinessAll.textContent = "All";
    const readinessReady = document.createElement("option");
    readinessReady.value = "ready";
    readinessReady.textContent = "On this Mac";
    const readinessDownloadable = document.createElement("option");
    readinessDownloadable.value = "downloadable";
    readinessDownloadable.textContent = "Downloadable";
    readinessFilter.replaceChildren(readinessAll, readinessReady, readinessDownloadable);
    readinessFilter.value = libraryReadinessFilter;
    const familyFilter = document.createElement("select");
    familyFilter.id = "library-family-filter";
    familyFilter.setAttribute("aria-label", "Filter by family");
    const familyAll = document.createElement("option");
    familyAll.value = "all";
    familyAll.textContent = "All families";
    familyFilter.replaceChildren(familyAll);
    const catalogFamilies = new Set(catalogRows.map((catalogRow) => catalogRow.family));
    for (const catalogFamily of catalogFamilies) {
        const familyOption = document.createElement("option");
        familyOption.value = catalogFamily;
        familyOption.textContent = libraryFamilyLabel(catalogFamily);
        familyFilter.appendChild(familyOption);
    }
    familyFilter.value = libraryFamilyFilter;
    bar.replaceChildren(searchInput, readinessFilter, familyFilter);
    // Wire controls after insertion so the coordinator's listeners attach.
    setTimeout(wireLibraryControls, 0);
    return bar;
}

function createLibraryCatalogRow(catalogRow, allRows) {
    const isExpanded = libraryExpandedModelId === catalogRow.huggingfaceId;
    const isActiveDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const downloadState = isActiveDownload
        ? libraryCurrentDownload.state
        : catalogRow.downloadState;
    const isDownloading = downloadState === "downloading" || downloadState === "paused";
    const row = document.createElement("article");
    const rowClasses = ["library-model-card"];
    if (catalogRow.readyOnThisMac) rowClasses.push("library-model-card-ready");
    if (isDownloading) rowClasses.push("library-model-card-downloading");
    if (isExpanded) rowClasses.push("library-model-card-expanded");
    row.className = rowClasses.join(" ");

    row.appendChild(createLibraryCardHeader(catalogRow));
    if (catalogRow.description) {
        const description = document.createElement("p");
        description.className = "library-model-description";
        description.textContent = catalogRow.description;
        row.appendChild(description);
    }
    row.appendChild(createLibraryCapabilityBadges(catalogRow));
    row.appendChild(createLibraryModelFacts(catalogRow));

    const progress = createLibraryProgress(catalogRow);
    if (progress.firstChild) row.appendChild(progress);
    const detailText = libraryProgressDetail(catalogRow);
    if (detailText) {
        const detail = document.createElement("p");
        detail.className = "library-download-detail";
        detail.textContent = detailText;
        row.appendChild(detail);
    }

    row.appendChild(createLibraryPrimaryActions(catalogRow, isExpanded));
    if (isExpanded) {
        row.appendChild(createLibraryDetailsPanel(catalogRow, allRows));
    }
    return row;
}

function createLibraryCardHeader(catalogRow) {
    const header = document.createElement("header");
    header.className = "library-model-heading";
    const identity = document.createElement("div");
    identity.className = "library-model-identity";
    const displayName = document.createElement("h3");
    displayName.textContent = catalogRow.displayName;
    const providerIdentity = document.createElement("p");
    providerIdentity.className = "library-model-provider-id";
    providerIdentity.textContent = catalogRow.huggingfaceId;
    identity.replaceChildren(displayName, providerIdentity);
    const availability = document.createElement("span");
    availability.className = catalogRow.readyOnThisMac
        ? "library-availability library-availability-ready"
        : isLibraryRowDownloading(catalogRow)
            ? "library-availability library-availability-downloading"
            : "library-availability";
    availability.textContent = catalogRow.readyOnThisMac
        ? "On this Mac"
        : isLibraryRowDownloading(catalogRow)
            ? "Downloading"
            : "Available";
    header.replaceChildren(identity, availability);
    return header;
}

function isLibraryRowDownloading(catalogRow) {
    const isActiveDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const downloadState = isActiveDownload
        ? libraryCurrentDownload.state
        : catalogRow.downloadState;
    return downloadState === "downloading"
        || downloadState === "paused"
        || downloadState === "verifying"
        || downloadState === "publishing";
}

function createLibraryCapabilityBadges(catalogRow) {
    const badgeContainer = document.createElement("div");
    badgeContainer.className = "library-capability-badges";
    const badges = [];
    if (catalogRow.supportsReasoning) badges.push("Reasoning");
    if (catalogRow.supportsVision) badges.push("Vision");
    if (catalogRow.supportsToolCalls) badges.push("Tools");
    if (catalogRow.supportsImageGeneration) badges.push("Image generation");
    const contextLabel = formatLibraryContextTokens(catalogRow.contextWindow);
    if (contextLabel) badges.push(contextLabel);
    for (const badgeText of badges) {
        const badge = document.createElement("span");
        badge.className = "library-capability-badge";
        badge.textContent = badgeText;
        badgeContainer.appendChild(badge);
    }
    return badgeContainer;
}

function createLibraryModelFacts(catalogRow) {
    const facts = document.createElement("dl");
    facts.className = "library-model-facts";
    const factEntries = [];
    factEntries.push(["Family", libraryFamilyLabel(catalogRow.family)]);
    factEntries.push(["Download size", catalogRow.approximateSize]);
    if (catalogRow.quantizationLabel) {
        factEntries.push(["Quantization", catalogRow.quantizationLabel]);
    }
    if (catalogRow.architectureSummary) {
        factEntries.push(["Architecture", catalogRow.architectureSummary]);
    }
    for (const [label, value] of factEntries) {
        const term = document.createElement("dt");
        term.textContent = label;
        const definition = document.createElement("dd");
        definition.textContent = value;
        facts.appendChild(term);
        facts.appendChild(definition);
    }
    return facts;
}

function createLibraryProgress(catalogRow) {
    const isActiveDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const downloadState = isActiveDownload
        ? libraryCurrentDownload.state
        : catalogRow.downloadState;
    const track = document.createElement("div");
    track.className = "library-download-progress";
    if (!isActiveDownload || !(libraryCurrentDownload.bytes_total > 0)
        || !["downloading", "paused", "verifying", "publishing"].includes(downloadState)) {
        return track;
    }
    const percent = downloadState === "publishing" || downloadState === "verifying"
        ? 100
        : libraryProgressPercent(libraryCurrentDownload);
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(percent));
    const fill = document.createElement("div");
    fill.className = "library-download-progress-fill";
    fill.style.width = percent + "%";
    track.appendChild(fill);
    return track;
}

function createLibraryPrimaryActions(catalogRow, isExpanded) {
    const actions = document.createElement("div");
    actions.className = "library-primary-actions";
    if (catalogRow.readyOnThisMac && !catalogRow.supportsImageGeneration) {
        const openInChatButton = document.createElement("button");
        openInChatButton.className = "button button-primary library-open-in-chat";
        openInChatButton.type = "button";
        openInChatButton.textContent = "Open in Chat";
        const requestableModelId = catalogRow.requestableModelId
            || deriveRequestableModelId(catalogRow.huggingfaceId);
        openInChatButton.addEventListener("click", () =>
            openLibraryModelInChat(requestableModelId)
        );
        actions.appendChild(openInChatButton);
    }
    for (const button of libraryActionButtons(catalogRow)) {
        actions.appendChild(button);
    }
    const detailsButton = document.createElement("button");
    detailsButton.className = "button button-secondary library-details-toggle";
    detailsButton.type = "button";
    detailsButton.textContent = isExpanded ? "Hide details" : "Details";
    detailsButton.setAttribute("aria-expanded", String(isExpanded));
    detailsButton.addEventListener("click", () => {
        libraryExpandedModelId = isExpanded
            ? null
            : catalogRow.huggingfaceId;
        renderLibraryCatalogDocument(libraryCatalogDocument);
    });
    actions.appendChild(detailsButton);
    return actions;
}

function deriveRequestableModelId(huggingfaceId) {
    const leafSegment = huggingfaceId.split("/").pop();
    return leafSegment || huggingfaceId;
}

function openLibraryModelInChat(requestableModelId) {
    if (typeof selectedModelId !== "undefined") {
        selectedModelId = requestableModelId;
    }
    const chatNavigationButton = document.querySelector(
        '[data-observatory-destination="chat"]'
    );
    if (chatNavigationButton) {
        chatNavigationButton.click();
    }
}

function createLibraryDetailsPanel(catalogRow, allRows) {
    const panel = document.createElement("section");
    panel.className = "library-model-details";
    const heading = document.createElement("p");
    heading.className = "library-details-heading";
    heading.textContent = "Details";
    panel.appendChild(heading);
    const detailList = document.createElement("dl");
    detailList.className = "library-detail-list";
    const detailEntries = [];
    if (catalogRow.contextWindow) {
        detailEntries.push(["Context window", catalogRow.contextWindow.toLocaleString() + " tokens"]);
    }
    if (catalogRow.maxOutputTokens) {
        detailEntries.push(["Max output", catalogRow.maxOutputTokens.toLocaleString() + " tokens"]);
    }
    if (catalogRow.quantizationLabel) {
        detailEntries.push(["Quantization", catalogRow.quantizationLabel]);
    }
    if (catalogRow.architectureSummary) {
        detailEntries.push(["Architecture", catalogRow.architectureSummary]);
    }
    if (catalogRow.upstreamLicense) {
        detailEntries.push(["License", catalogRow.upstreamLicense]);
    }
    for (const [label, value] of detailEntries) {
        const term = document.createElement("dt");
        term.textContent = label;
        const definition = document.createElement("dd");
        definition.textContent = value;
        detailList.appendChild(term);
        detailList.appendChild(definition);
    }
    panel.appendChild(detailList);
    const destinationPath = libraryDestinationPath(catalogRow);
    if (destinationPath) {
        panel.appendChild(createLibraryLocationSection(destinationPath));
    }
    return panel;
}

function createLibraryLocationSection(destinationPath) {
    const location = document.createElement("section");
    location.className = "library-model-location";
    const locationLabel = document.createElement("p");
    locationLabel.className = "library-model-location-label";
    locationLabel.textContent = "Model location";
    const locationValue = document.createElement("code");
    locationValue.className = "library-download-path";
    locationValue.textContent = destinationPath;
    const actionRow = document.createElement("div");
    actionRow.className = "library-location-actions";
    const revealButton = document.createElement("button");
    revealButton.className = "button button-secondary library-reveal-button";
    revealButton.type = "button";
    revealButton.textContent = "Show in Finder";
    revealButton.addEventListener("click", () => revealLibraryPathInFinder(destinationPath));
    const copyButton = document.createElement("button");
    copyButton.className = "button button-secondary library-copy-path-button";
    copyButton.type = "button";
    copyButton.textContent = "Copy Path";
    copyButton.addEventListener("click", () => copyLibraryPathToClipboard(destinationPath));
    actionRow.replaceChildren(revealButton, copyButton);
    location.replaceChildren(locationLabel, locationValue, actionRow);
    return location;
}

async function revealLibraryPathInFinder(destinationPath) {
    // Observatory is a web UI; the daemon does not expose a reveal endpoint, so
    // we use the Web Share API as a progressive enhancement and fall back to
    // copying the path for the user.
    try {
        await navigator.clipboard.writeText(destinationPath);
    } catch {
        // Clipboard may be unavailable without a secure context or user gesture.
    }
}

async function copyLibraryPathToClipboard(destinationPath) {
    try {
        await navigator.clipboard.writeText(destinationPath);
    } catch {
        // Clipboard may be unavailable without a secure context or user gesture.
    }
}
