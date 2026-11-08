// Astronomical Observatory Library rendering engine.
// Owns model card construction, capability badges, download hero, expandable
// details, hero summary, and filter bar. All catalog values enter the document
// only through textContent and created elements.

function libraryFamilyLabel(family) {
    if (family === "qwen3_5") return "Qwen 3.5";
    if (family === "flux2_klein") return "FLUX.2 Klein";
    if (family === "modernbert") return "ModernBERT";
    return family;
}

// Plain-language copy for each role. The same wording labels the Get started
// cards and the More models groups, so a model reads the same on both pages.
const LIBRARY_ROLE_TITLES = {
    chat: "Chat",
    coding: "Code",
    image_generation: "Pictures",
    helper: "Helpers"
};
const LIBRARY_ROLE_DESCRIPTIONS = {
    chat: "Talk, write, and ask about photos.",
    coding: "Help writing, reading, and fixing code.",
    image_generation: "Turn a written description into a picture."
};
const LIBRARY_GET_STARTED_ROLES = ["chat", "coding", "image_generation"];
const LIBRARY_MORE_MODEL_GROUPS = [
    { key: "chat", title: "Chat", explanation: LIBRARY_ROLE_DESCRIPTIONS.chat },
    { key: "coding", title: "Code", explanation: LIBRARY_ROLE_DESCRIPTIONS.coding },
    { key: "image_generation", title: "Pictures", explanation: LIBRARY_ROLE_DESCRIPTIONS.image_generation },
    {
        key: "helper",
        title: "Helpers",
        explanation: "Background helpers other features use. You don't chat with these."
    },
    {
        key: "experimental",
        title: "Experimental",
        explanation: "Early builds that may be slower or less reliable."
    }
];
const LIBRARY_TABS = [
    { name: "get-started", label: "Get started" },
    { name: "more-models", label: "More models" }
];
const LIBRARY_FIT_LABELS = {
    fits: "Runs in memory on this Mac",
    streams: "Streams from your SSD on this Mac (slower)",
    slow: "May be slow on this Mac"
};

// Plain-language size for download decisions: one decimal in 2^30-byte gigabytes,
// the unit Apple prints on the box, so a teenager can compare two picks at a glance.
function formatLibraryDownloadSize(approximateSizeBytes) {
    return (approximateSizeBytes / 1073741824).toFixed(1) + " GB download";
}

// Fit compares the model's fully loaded memory with the serving ceiling, the
// limit the engine actually loads within. Equal counts as fitting because the
// ceiling is the runtime's own resident limit. Below that, a model with a
// streaming floor measured on a real engine streams experts or image components
// from the SSD; any other model still runs, with macOS paging the rest, so it is
// labelled as possibly slow. Nothing here ever says a model cannot run (#1902).
function libraryFitOutcome(catalogRow, servingCeilingBytes) {
    if (!(servingCeilingBytes > 0)) return null;
    if (catalogRow.approximateMemoryBytes <= servingCeilingBytes) return "fits";
    const streamingFloorBytes = catalogRow.minimumMemoryBytes;
    return streamingFloorBytes !== null && streamingFloorBytes <= servingCeilingBytes
        ? "streams"
        : "slow";
}

function libraryFitLabel(catalogRow, servingCeilingBytes) {
    const fitOutcome = libraryFitOutcome(catalogRow, servingCeilingBytes);
    return fitOutcome === null ? null : LIBRARY_FIT_LABELS[fitOutcome];
}

function libraryStarterRowForRole(catalogRows, role) {
    return catalogRows.find(
        (catalogRow) => catalogRow.role === role && catalogRow.getStarted
    ) || null;
}

// Experimental wins over role so an early build never appears in a stable group.
function libraryMoreModelsGroupKey(catalogRow) {
    return catalogRow.experimental ? "experimental" : catalogRow.role;
}

function createLibraryTabPanel(tabName, pageChildren) {
    const panel = document.createElement("section");
    panel.id = "library-panel-" + tabName;
    panel.className = "library-tab-panel";
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", "library-tab-" + tabName);
    panel.replaceChildren(...pageChildren);
    return panel;
}

function createLibraryTabBar() {
    const tabBar = document.createElement("div");
    tabBar.className = "library-tab-bar";
    tabBar.setAttribute("role", "tablist");
    tabBar.setAttribute("aria-label", "Library pages");
    tabBar.replaceChildren(...LIBRARY_TABS.map(createLibraryTabButton));
    return tabBar;
}

function createLibraryTabButton(tabDefinition) {
    const isSelected = libraryActiveTab === tabDefinition.name;
    const tabButton = document.createElement("button");
    tabButton.className = isSelected ? "library-tab library-tab-selected" : "library-tab";
    tabButton.type = "button";
    tabButton.id = "library-tab-" + tabDefinition.name;
    tabButton.setAttribute("role", "tab");
    tabButton.setAttribute("aria-selected", String(isSelected));
    tabButton.setAttribute("aria-controls", "library-panel-" + tabDefinition.name);
    // Roving tabindex: only the selected tab sits in the Tab order, as the ARIA tabs pattern expects.
    tabButton.tabIndex = isSelected ? 0 : -1;
    tabButton.textContent = tabDefinition.label;
    tabButton.addEventListener("click", () => selectLibraryTab(tabDefinition.name));
    tabButton.addEventListener("keydown", (keyboardEvent) => handleLibraryTabKeydown(keyboardEvent, tabDefinition.name));
    return tabButton;
}

// WAI-ARIA tabs pattern: arrows step through the tabs with wrap-around, and
// Home and End jump to the ends. Returns the target tab name, or null for keys
// the tablist does not handle, so those keep their default behavior.
function libraryTabNameForKey(currentTabName, pressedKey) {
    const tabCount = LIBRARY_TABS.length;
    const currentIndex = LIBRARY_TABS.findIndex((tab) => tab.name === currentTabName);
    const targetIndexByKey = {
        ArrowRight: (currentIndex + 1) % tabCount,
        ArrowLeft: (currentIndex - 1 + tabCount) % tabCount,
        Home: 0,
        End: tabCount - 1
    };
    if (!Object.hasOwn(targetIndexByKey, pressedKey)) return null;
    return LIBRARY_TABS[targetIndexByKey[pressedKey]].name;
}

// Keyboard selection uses the same path as a click, so focus follows the newly selected tab.
function handleLibraryTabKeydown(keyboardEvent, tabName) {
    const targetTabName = libraryTabNameForKey(tabName, keyboardEvent.key);
    if (targetTabName === null) return;
    keyboardEvent.preventDefault();
    selectLibraryTab(targetTabName);
}

function createLibraryActivePage(catalogRows, catalogDocument) {
    if (libraryActiveTab === "more-models") {
        return createLibraryMoreModelsPage(catalogRows, catalogDocument);
    }
    return createLibraryGetStartedPage(catalogRows);
}

function createLibraryGetStartedPage(catalogRows) {
    const intro = document.createElement("p");
    intro.className = "library-intro";
    intro.textContent = "Pick what you want to do. Each is one download.";
    const roleCards = LIBRARY_GET_STARTED_ROLES
        .map((role) => libraryStarterRowForRole(catalogRows, role))
        .filter((starterRow) => starterRow !== null)
        .map(createLibraryRoleCard);
    const cardGrid = document.createElement("div");
    cardGrid.className = "library-role-grid";
    cardGrid.replaceChildren(...roleCards);
    return createLibraryTabPanel("get-started", [intro, cardGrid]);
}

function createLibraryRoleCard(catalogRow) {
    const card = document.createElement("article");
    card.className = catalogRow.readyOnThisMac
        ? "library-role-card library-role-card-ready"
        : "library-role-card";
    const cardChildren = [
        createLibraryRoleHeading(catalogRow),
        createLibraryRoleFacts(catalogRow)
    ];
    const progress = createLibraryProgress(catalogRow);
    if (progress.firstChild) cardChildren.push(progress);
    const detailText = libraryProgressDetail(catalogRow);
    if (detailText) {
        cardChildren.push(createLibraryParagraph("library-download-detail", detailText));
    }
    cardChildren.push(createLibraryStartActions(catalogRow));
    card.replaceChildren(...cardChildren);
    return card;
}

function createLibraryRoleHeading(catalogRow) {
    const heading = document.createElement("div");
    heading.className = "library-role-heading";
    const roleTitle = document.createElement("h3");
    roleTitle.className = "library-role-title";
    roleTitle.textContent = LIBRARY_ROLE_TITLES[catalogRow.role];
    heading.appendChild(roleTitle);
    if (catalogRow.readyOnThisMac) {
        const availability = document.createElement("span");
        availability.className = "library-availability library-availability-ready";
        availability.textContent = "On this Mac";
        heading.appendChild(availability);
    }
    return heading;
}

// Get started shows only plain facts: no Hugging Face id, quantization, context
// size, or capability badges. Those stay on More models for people who look for them.
function createLibraryRoleFacts(catalogRow) {
    const facts = document.createElement("div");
    facts.className = "library-role-facts";
    const factElements = [
        createLibraryParagraph("library-role-description", LIBRARY_ROLE_DESCRIPTIONS[catalogRow.role]),
        createLibraryParagraph("library-role-model", catalogRow.displayName),
        createLibraryParagraph("library-role-size", formatLibraryDownloadSize(catalogRow.approximateSizeBytes))
    ];
    const fitLabel = createLibraryFitLabel(catalogRow);
    if (fitLabel) factElements.push(fitLabel);
    facts.replaceChildren(...factElements);
    return facts;
}

function createLibraryParagraph(className, text) {
    const paragraph = document.createElement("p");
    paragraph.className = className;
    paragraph.textContent = text;
    return paragraph;
}

// The modifier class carries the outcome so CSS can color fits, streams and slow apart.
function createLibraryFitLabel(catalogRow) {
    const fitOutcome = libraryFitOutcome(catalogRow, libraryServingCeilingBytes);
    if (fitOutcome === null) return null;
    return createLibraryParagraph(
        "library-fit-label library-fit-" + fitOutcome,
        LIBRARY_FIT_LABELS[fitOutcome]
    );
}

function createLibraryStartActions(catalogRow) {
    const actions = document.createElement("div");
    actions.className = "library-primary-actions";
    if (catalogRow.readyOnThisMac && catalogRow.role !== "image_generation") {
        actions.appendChild(createLibraryOpenInChatButton());
    }
    for (const button of libraryActionButtons(catalogRow)) {
        actions.appendChild(button);
    }
    return actions;
}

function createLibraryMoreModelsPage(catalogRows, catalogDocument) {
    const filteredRows = filterLibraryRows(catalogRows);
    const catalogList = document.createElement("div");
    catalogList.className = "library-catalog-list";
    if (filteredRows.length === 0) {
        const noMatches = document.createElement("p");
        noMatches.className = "library-no-matches";
        noMatches.textContent = "No models match the current filter.";
        catalogList.replaceChildren(noMatches);
    } else {
        catalogList.replaceChildren(...createLibraryGroupSections(filteredRows));
    }
    return createLibraryTabPanel("more-models", [
        createLibraryHeroSummary(catalogRows, catalogDocument),
        createLibraryFilterBar(catalogRows),
        catalogList
    ]);
}

function createLibraryGroupSections(filteredRows) {
    return LIBRARY_MORE_MODEL_GROUPS
        .map((group) => {
            const groupRows = filteredRows.filter(
                (catalogRow) => libraryMoreModelsGroupKey(catalogRow) === group.key
            );
            return groupRows.length === 0 ? null : createLibraryGroupSection(group, groupRows);
        })
        .filter((groupSection) => groupSection !== null);
}

function createLibraryGroupSection(group, groupRows) {
    const section = document.createElement("section");
    section.className = "library-group";
    const heading = document.createElement("h3");
    heading.className = "library-group-heading";
    heading.textContent = group.title;
    section.replaceChildren(
        heading,
        createLibraryParagraph("library-group-explanation", group.explanation),
        ...groupRows.map((catalogRow) => createLibraryCatalogRow(catalogRow, groupRows))
    );
    return section;
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

function createLibraryHeroSummary(catalogRows, catalogDocument) {
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
    const downloadLocation = catalogDocument?.models_download_directory;
    if (typeof downloadLocation === "string" && downloadLocation.trim().length > 0) {
        const locationNote = document.createElement("p");
        locationNote.className = "library-hero-location";
        locationNote.textContent = "New downloads are saved to " + downloadLocation;
        summary.appendChild(locationNote);
    }
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
    // The filter state stores only the normalized lowercase query used for
    // filtering; the input keeps the exact text the user is typing.
    const previousSearchInput = document.getElementById("library-search");
    searchInput.value = previousSearchInput
        ? previousSearchInput.value
        : librarySearchQuery;
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
    const displayName = document.createElement("h4");
    displayName.textContent = catalogRow.displayName;
    const providerIdentity = document.createElement("p");
    providerIdentity.className = "library-model-provider-id";
    providerIdentity.textContent = catalogRow.huggingfaceId;
    identity.replaceChildren(
        displayName,
        providerIdentity,
        createLibraryCardFacts(catalogRow)
    );
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

// The size and fit line sit under the title so the decision inputs are visible
// before the reader reaches the actions.
function createLibraryCardFacts(catalogRow) {
    const facts = document.createElement("div");
    facts.className = "library-model-facts";
    const sizeLine = createLibraryParagraph(
        "library-model-size",
        formatLibraryDownloadSize(catalogRow.approximateSizeBytes)
    );
    const fitLabel = createLibraryFitLabel(catalogRow);
    facts.replaceChildren(sizeLine, ...(fitLabel ? [fitLabel] : []));
    return facts;
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
    if (catalogRow.supportsEmbeddings) badges.push("Embeddings");
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
        actions.appendChild(createLibraryOpenInChatButton());
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

function createLibraryOpenInChatButton() {
    const openInChatButton = document.createElement("button");
    openInChatButton.className = "button button-primary library-open-in-chat";
    openInChatButton.type = "button";
    openInChatButton.textContent = "Open in Thin Talk";
    openInChatButton.addEventListener("click", openThinTalkFromLibrary);
    return openInChatButton;
}

// Thin Talk picks the model inside its own model chooser, so opening the
// conversation window is all the handoff needs; there is no in-Observatory
// chat to preselect a model for anymore.
function openThinTalkFromLibrary() {
    const chatLink = thinTalkChatLink(document.body.dataset.applicationChannel);
    if (chatLink) {
        window.location.assign(chatLink);
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
