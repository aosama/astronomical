// Astronomical Observatory Download Models rendering engine.
// Owns model card construction, capability badges, download hero, expandable
// details, hero summary, and filter bar. All catalog values enter the document
// only through textContent and created elements.

// Every family in registry/download_catalog.json needs a name here: the family
// filter otherwise shows the catalog's internal key. download-models.test.js
// reads the shipped catalog and fails when a family has no name.
const DOWNLOAD_MODELS_FAMILY_LABELS = {
    qwen3_5: "Qwen 3.5",
    flux2_klein: "FLUX.2 Klein",
    modernbert: "ModernBERT",
    qwen_image_21: "Qwen-Image 2.1",
    prism_hadamard_qwen35: "Ternary Bonsai"
};

function downloadModelsFamilyLabel(family) {
    return Object.hasOwn(DOWNLOAD_MODELS_FAMILY_LABELS, family)
        ? DOWNLOAD_MODELS_FAMILY_LABELS[family]
        : family;
}

// Plain-language copy for each role. The same wording labels the Get started
// cards and the More models groups, so a model reads the same on both pages.
const DOWNLOAD_MODELS_ROLE_TITLES = {
    chat: "Chat",
    coding: "Code",
    image_generation: "Image Generation",
    helper: "Helpers"
};
const DOWNLOAD_MODELS_ROLE_DESCRIPTIONS = {
    chat: "Talk, write, and ask about photos.",
    coding: "Help writing, reading, and fixing code.",
    image_generation: "Turn a written description into a picture."
};
const DOWNLOAD_MODELS_GET_STARTED_ROLES = ["chat", "coding", "image_generation"];
const DOWNLOAD_MODELS_MORE_MODEL_GROUPS = [
    { key: "chat", title: DOWNLOAD_MODELS_ROLE_TITLES.chat, explanation: DOWNLOAD_MODELS_ROLE_DESCRIPTIONS.chat },
    { key: "coding", title: DOWNLOAD_MODELS_ROLE_TITLES.coding, explanation: DOWNLOAD_MODELS_ROLE_DESCRIPTIONS.coding },
    { key: "image_generation", title: DOWNLOAD_MODELS_ROLE_TITLES.image_generation, explanation: DOWNLOAD_MODELS_ROLE_DESCRIPTIONS.image_generation },
    {
        key: "helper",
        title: DOWNLOAD_MODELS_ROLE_TITLES.helper,
        explanation: "Background helpers other features use. You don't chat with these."
    },
    {
        key: "experimental",
        title: "Experimental",
        explanation: "Early builds that may be slower or less reliable."
    },
    {
        key: "retiring",
        title: "Retiring soon",
        explanation: "These models will leave the catalog in a future release. A copy already on this Mac keeps working."
    }
];
const DOWNLOAD_MODELS_TABS = [
    { name: "get-started", label: "Get started" },
    { name: "more-models", label: "More models" }
];
const DOWNLOAD_MODELS_FIT_LABELS = {
    fits: "Runs in memory on this Mac",
    streams: "Streams from your SSD on this Mac (slower)",
    slow: "May be slow on this Mac",
    "over-your-limit": "Above the memory limit you set"
};

// Plain-language size for download decisions: one decimal in 2^30-byte gigabytes,
// the unit Apple prints on the box, so a teenager can compare two picks at a glance.
// Just the number and unit: beside a model name "GB" already reads as its size,
// and the word "download" added nothing. The tooltip says what the size is.
function formatDownloadModelsCardSize(approximateSizeBytes) {
    return (approximateSizeBytes / 1073741824).toFixed(1) + " GB";
}

// Fit compares the model's fully loaded memory with the serving ceiling, the
// limit the engine actually loads within. Equal counts as fitting because the
// ceiling is the runtime's own resident limit. Below that, a model with a
// streaming floor measured on a real engine streams experts or image components
// from the SSD; any other model still runs, with macOS paging the rest, so it is
// labelled as possibly slow: the Mac's size never stops a model (#1902). Only
// a maximum the user set themself can hold a model back, and the label names it.
function downloadModelsFitOutcome(catalogRow, servingCeilingBytes, ceilingIsUserLimit = false) {
    if (!(servingCeilingBytes > 0)) return null;
    if (catalogRow.approximateMemoryBytes <= servingCeilingBytes) return "fits";
    const streamingFloorBytes = catalogRow.minimumMemoryBytes;
    if (streamingFloorBytes !== null && streamingFloorBytes <= servingCeilingBytes) {
        return "streams";
    }
    return ceilingIsUserLimit ? "over-your-limit" : "slow";
}

function downloadModelsFitLabel(catalogRow, servingCeilingBytes, ceilingIsUserLimit = false) {
    const fitOutcome = downloadModelsFitOutcome(catalogRow, servingCeilingBytes, ceilingIsUserLimit);
    return fitOutcome === null ? null : DOWNLOAD_MODELS_FIT_LABELS[fitOutcome];
}

function downloadModelsStarterRowForRole(catalogRows, role) {
    return catalogRows.find(
        (catalogRow) => catalogRow.role === role && catalogRow.getStarted
    ) || null;
}

// Retiring wins over everything so a model on its way out is never offered
// beside the ones replacing it; experimental wins over role so an early build
// never appears in a stable group.
function downloadModelsMoreModelsGroupKey(catalogRow) {
    if (catalogRow.retiring) return "retiring";
    return catalogRow.experimental ? "experimental" : catalogRow.role;
}

function createDownloadModelsTabPanel(tabName, pageChildren) {
    const panel = document.createElement("section");
    panel.id = "download-models-panel-" + tabName;
    panel.className = "download-models-tab-panel";
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", "download-models-tab-" + tabName);
    panel.replaceChildren(...pageChildren);
    return panel;
}

function createDownloadModelsTabBar() {
    const tabBar = document.createElement("div");
    tabBar.className = "download-models-tab-bar";
    tabBar.setAttribute("role", "tablist");
    tabBar.setAttribute("aria-label", "Download Models pages");
    tabBar.replaceChildren(...DOWNLOAD_MODELS_TABS.map(createDownloadModelsTabButton));
    return tabBar;
}

function createDownloadModelsTabButton(tabDefinition) {
    const isSelected = downloadModelsActiveTab === tabDefinition.name;
    const tabButton = document.createElement("button");
    tabButton.className = isSelected ? "download-models-tab download-models-tab-selected" : "download-models-tab";
    tabButton.type = "button";
    tabButton.id = "download-models-tab-" + tabDefinition.name;
    tabButton.setAttribute("role", "tab");
    tabButton.setAttribute("aria-selected", String(isSelected));
    tabButton.setAttribute("aria-controls", "download-models-panel-" + tabDefinition.name);
    // Roving tabindex: only the selected tab sits in the Tab order, as the ARIA tabs pattern expects.
    tabButton.tabIndex = isSelected ? 0 : -1;
    tabButton.textContent = tabDefinition.label;
    tabButton.addEventListener("click", () => selectDownloadModelsTab(tabDefinition.name));
    tabButton.addEventListener("keydown", (keyboardEvent) => handleDownloadModelsTabKeydown(keyboardEvent, tabDefinition.name));
    return tabButton;
}

// WAI-ARIA tabs pattern: arrows step through the tabs with wrap-around, and
// Home and End jump to the ends. Returns the target tab name, or null for keys
// the tablist does not handle, so those keep their default behavior.
function downloadModelsTabNameForKey(currentTabName, pressedKey) {
    const tabCount = DOWNLOAD_MODELS_TABS.length;
    const currentIndex = DOWNLOAD_MODELS_TABS.findIndex((tab) => tab.name === currentTabName);
    const targetIndexByKey = {
        ArrowRight: (currentIndex + 1) % tabCount,
        ArrowLeft: (currentIndex - 1 + tabCount) % tabCount,
        Home: 0,
        End: tabCount - 1
    };
    if (!Object.hasOwn(targetIndexByKey, pressedKey)) return null;
    return DOWNLOAD_MODELS_TABS[targetIndexByKey[pressedKey]].name;
}

// Keyboard selection uses the same path as a click, so focus follows the newly selected tab.
function handleDownloadModelsTabKeydown(keyboardEvent, tabName) {
    const targetTabName = downloadModelsTabNameForKey(tabName, keyboardEvent.key);
    if (targetTabName === null) return;
    keyboardEvent.preventDefault();
    selectDownloadModelsTab(targetTabName);
}

function createDownloadModelsActivePage(catalogRows, catalogDocument) {
    if (downloadModelsActiveTab === "more-models") {
        return createDownloadModelsMoreModelsPage(catalogRows, catalogDocument);
    }
    return createDownloadModelsGetStartedPage(catalogRows);
}

function createDownloadModelsGetStartedPage(catalogRows) {
    const intro = document.createElement("p");
    intro.className = "download-models-intro";
    intro.textContent = "Pick what you want to do. Each is one download.";
    const roleCards = DOWNLOAD_MODELS_GET_STARTED_ROLES
        .map((role) => downloadModelsStarterRowForRole(catalogRows, role))
        .filter((starterRow) => starterRow !== null)
        .map(createDownloadModelsRoleCard);
    const cardGrid = document.createElement("div");
    cardGrid.className = "download-models-role-grid";
    cardGrid.replaceChildren(...roleCards);
    return createDownloadModelsTabPanel("get-started", [intro, cardGrid]);
}

function createDownloadModelsRoleCard(catalogRow) {
    const card = document.createElement("article");
    card.className = catalogRow.readyOnThisMac
        ? "download-models-role-card download-models-role-card-ready"
        : "download-models-role-card";
    const cardChildren = [
        createDownloadModelsRoleHeading(catalogRow),
        createDownloadModelsRoleFacts(catalogRow)
    ];
    const progress = createDownloadModelsProgress(catalogRow);
    if (progress.firstChild) cardChildren.push(progress);
    const detailText = downloadModelsProgressDetail(catalogRow);
    if (detailText) {
        cardChildren.push(createDownloadModelsParagraph("download-models-download-detail", detailText));
    }
    cardChildren.push(createDownloadModelsStartActions(catalogRow));
    card.replaceChildren(...cardChildren);
    return card;
}

function createDownloadModelsRoleHeading(catalogRow) {
    const heading = document.createElement("div");
    heading.className = "download-models-role-heading";
    const roleTitle = document.createElement("h3");
    roleTitle.className = "download-models-role-title";
    roleTitle.textContent = DOWNLOAD_MODELS_ROLE_TITLES[catalogRow.role];
    heading.appendChild(roleTitle);
    if (catalogRow.readyOnThisMac) {
        const availability = document.createElement("span");
        availability.className = "download-models-availability download-models-availability-ready";
        availability.textContent = "On this Mac";
        heading.appendChild(availability);
    }
    return heading;
}

// Get started shows only plain facts: no Hugging Face id, quantization, context
// size, or capability badges. Those stay on More models for people who look for them.
function createDownloadModelsRoleFacts(catalogRow) {
    const facts = document.createElement("div");
    facts.className = "download-models-role-facts";
    const factElements = [
        createDownloadModelsParagraph("download-models-role-description", DOWNLOAD_MODELS_ROLE_DESCRIPTIONS[catalogRow.role]),
        createDownloadModelsParagraph("download-models-role-model", catalogRow.displayName),
        createDownloadModelsParagraph("download-models-role-size", formatDownloadModelsCardSize(catalogRow.approximateSizeBytes))
    ];
    const fitLabel = createDownloadModelsFitLabel(catalogRow);
    if (fitLabel) factElements.push(fitLabel);
    facts.replaceChildren(...factElements);
    return facts;
}

function createDownloadModelsParagraph(className, text) {
    const paragraph = document.createElement("p");
    paragraph.className = className;
    paragraph.textContent = text;
    return paragraph;
}

// The modifier class carries the outcome so CSS can color fits, streams and slow apart.
function createDownloadModelsFitLabel(catalogRow) {
    const fitOutcome = downloadModelsFitOutcome(
        catalogRow,
        downloadModelsServingCeilingBytes,
        downloadModelsCeilingIsUserLimit
    );
    if (fitOutcome === null) return null;
    return createDownloadModelsParagraph(
        "download-models-fit-label download-models-fit-" + fitOutcome,
        DOWNLOAD_MODELS_FIT_LABELS[fitOutcome]
    );
}

function createDownloadModelsStartActions(catalogRow) {
    const actions = document.createElement("div");
    actions.className = "download-models-primary-actions";
    if (canChatWithCatalogModel(catalogRow)) {
        actions.appendChild(createDownloadModelsOpenInChatButton());
    }
    for (const button of downloadModelsActionButtons(catalogRow)) {
        actions.appendChild(button);
    }
    return actions;
}

function createDownloadModelsMoreModelsPage(catalogRows, catalogDocument) {
    const filteredRows = filterDownloadModelsRows(catalogRows);
    const catalogList = document.createElement("div");
    catalogList.className = "download-models-catalog-list";
    if (filteredRows.length === 0) {
        const noMatches = document.createElement("p");
        noMatches.className = "download-models-no-matches";
        noMatches.textContent = "No models match the current filter.";
        catalogList.replaceChildren(noMatches);
    } else {
        catalogList.replaceChildren(...createDownloadModelsGroupSections(filteredRows));
    }
    return createDownloadModelsTabPanel("more-models", [
        createDownloadModelsHeroSummary(catalogRows, catalogDocument),
        createDownloadModelsFilterBar(catalogRows),
        catalogList
    ]);
}

function createDownloadModelsGroupSections(filteredRows) {
    return DOWNLOAD_MODELS_MORE_MODEL_GROUPS
        .map((group) => {
            const groupRows = filteredRows.filter(
                (catalogRow) => downloadModelsMoreModelsGroupKey(catalogRow) === group.key
            );
            return groupRows.length === 0 ? null : createDownloadModelsGroupSection(group, groupRows);
        })
        .filter((groupSection) => groupSection !== null);
}

function createDownloadModelsGroupSection(group, groupRows) {
    const section = document.createElement("section");
    section.className = "download-models-group";
    const heading = document.createElement("h3");
    heading.className = "download-models-group-heading";
    heading.textContent = group.title;
    section.replaceChildren(
        heading,
        createDownloadModelsParagraph("download-models-group-explanation", group.explanation),
        ...groupRows.map((catalogRow) => createDownloadModelsCatalogRow(catalogRow, groupRows))
    );
    return section;
}

function formatDownloadModelsContextTokens(tokenCount) {
    if (!tokenCount || tokenCount <= 0) return null;
    if (tokenCount >= 1_000_000) {
        return (tokenCount / 1_000_000).toFixed(0) + "M context";
    }
    if (tokenCount >= 1_000) {
        return (tokenCount / 1_000).toFixed(0) + "K context";
    }
    return tokenCount + " context";
}

// Transfer rate and time left as the download card prints them.
function formatDownloadModelsRate(bytesPerSecond) {
    if (bytesPerSecond >= 1073741824) {
        return (bytesPerSecond / 1073741824).toFixed(1) + " GB/s";
    }
    if (bytesPerSecond >= 1_000_000) {
        return (bytesPerSecond / 1_000_000).toFixed(1) + " MB/s";
    }
    return Math.max(1, Math.round(bytesPerSecond / 1_000)) + " KB/s";
}

function formatDownloadModelsRemainingTime(remainingSeconds) {
    if (!(remainingSeconds > 0) || !Number.isFinite(remainingSeconds)) {
        return "Calculating time remaining";
    }
    if (remainingSeconds < 60) return "Less than a minute left";
    const remainingMinutes = Math.round(remainingSeconds / 60);
    if (remainingMinutes < 60) {
        return remainingMinutes === 1
            ? "About 1 minute left"
            : "About " + remainingMinutes + " minutes left";
    }
    const remainingHours = Math.round(remainingMinutes / 60);
    return remainingHours === 1
        ? "About 1 hour left"
        : "About " + remainingHours + " hours left";
}

function createDownloadModelsHeroSummary(catalogRows, catalogDocument) {
    const summary = document.createElement("div");
    summary.className = "download-models-hero-summary";
    const readyCount = catalogRows.filter((row) => row.readyOnThisMac).length;
    const downloadableCount = catalogRows.length - readyCount;
    const totalSizeBytes = catalogRows.reduce(
        (total, row) => total + (row.approximateSizeBytes || 0), 0
    );
    const headline = document.createElement("p");
    headline.className = "download-models-hero-headline";
    const readySpan = document.createElement("span");
    readySpan.className = "download-models-hero-ready-count";
    readySpan.textContent = readyCount + " model" + (readyCount === 1 ? "" : "s") + " on this Mac";
    const downloadSpan = document.createElement("span");
    downloadSpan.className = "download-models-hero-downloadable-count";
    downloadSpan.textContent = " · " + downloadableCount + " ready to download";
    headline.replaceChildren(readySpan, downloadSpan);
    summary.appendChild(headline);
    const downloadLocation = catalogDocument?.models_download_directory;
    if (typeof downloadLocation === "string" && downloadLocation.trim().length > 0) {
        const locationNote = document.createElement("p");
        locationNote.className = "download-models-hero-location";
        locationNote.textContent = "New downloads are saved to " + downloadLocation;
        summary.appendChild(locationNote);
    }
    if (totalSizeBytes > 0 && readyCount > 0) {
        const sizeNote = document.createElement("p");
        sizeNote.className = "download-models-hero-size";
        sizeNote.textContent = formatDownloadModelsSizeGigabytes(totalSizeBytes) + " total across all catalog models";
        summary.appendChild(sizeNote);
    }
    return summary;
}

function createDownloadModelsFilterBar(catalogRows) {
    const bar = document.createElement("div");
    bar.className = "download-models-filter-bar";
    const searchInput = document.createElement("input");
    searchInput.id = "download-models-search";
    searchInput.type = "search";
    searchInput.placeholder = "Search models";
    // The filter state stores only the normalized lowercase query used for
    // filtering; the input keeps the exact text the user is typing.
    const previousSearchInput = document.getElementById("download-models-search");
    searchInput.value = previousSearchInput
        ? previousSearchInput.value
        : downloadModelsSearchQuery;
    searchInput.setAttribute("aria-label", "Search models");
    const readinessFilter = document.createElement("select");
    readinessFilter.id = "download-models-readiness-filter";
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
    readinessFilter.value = downloadModelsReadinessFilter;
    const familyFilter = document.createElement("select");
    familyFilter.id = "download-models-family-filter";
    familyFilter.setAttribute("aria-label", "Filter by family");
    const familyAll = document.createElement("option");
    familyAll.value = "all";
    familyAll.textContent = "All families";
    familyFilter.replaceChildren(familyAll);
    const catalogFamilies = new Set(catalogRows.map((catalogRow) => catalogRow.family));
    for (const catalogFamily of catalogFamilies) {
        const familyOption = document.createElement("option");
        familyOption.value = catalogFamily;
        familyOption.textContent = downloadModelsFamilyLabel(catalogFamily);
        familyFilter.appendChild(familyOption);
    }
    familyFilter.value = downloadModelsFamilyFilter;
    bar.replaceChildren(searchInput, readinessFilter, familyFilter);
    // Wire controls after insertion so the coordinator's listeners attach.
    setTimeout(wireDownloadModelsControls, 0);
    return bar;
}

function createDownloadModelsCatalogRow(catalogRow, allRows) {
    const isExpanded = downloadModelsExpandedModelId === catalogRow.huggingfaceId;
    const isActiveDownload = downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const downloadState = isActiveDownload
        ? downloadModelsCurrentDownload.state
        : catalogRow.downloadState;
    const isDownloading = downloadState === "downloading" || downloadState === "paused";
    const row = document.createElement("article");
    const rowClasses = ["download-models-model-card"];
    if (catalogRow.readyOnThisMac) rowClasses.push("download-models-model-card-ready");
    if (isDownloading) rowClasses.push("download-models-model-card-downloading");
    if (isExpanded) rowClasses.push("download-models-model-card-expanded");
    row.className = rowClasses.join(" ");

    row.appendChild(createDownloadModelsCardHeader(catalogRow));
    if (catalogRow.description) {
        const description = document.createElement("p");
        description.className = "download-models-model-description";
        description.textContent = catalogRow.description;
        row.appendChild(description);
    }
    const progress = createDownloadModelsProgress(catalogRow);
    if (progress.firstChild) row.appendChild(progress);
    // A download's progress outranks an older removal notice on the same card.
    const progressDetail = downloadModelsProgressDetail(catalogRow);
    const detailText = progressDetail || downloadModelsRemovalNoticeFor(catalogRow);
    if (detailText) {
        const detail = document.createElement("p");
        detail.className = "download-models-download-detail";
        detail.textContent = detailText;
        // A removal outcome is announced; progress, which changes every few
        // seconds, is not.
        if (!progressDetail) detail.setAttribute("role", "status");
        row.appendChild(detail);
    }

    row.appendChild(createDownloadModelsCardFooter(catalogRow, isExpanded));
    if (isExpanded) {
        row.appendChild(createDownloadModelsDetailsPanel(catalogRow, allRows));
    }
    return row;
}

function createDownloadModelsCardHeader(catalogRow) {
    const header = document.createElement("header");
    header.className = "download-models-model-heading";
    const identity = document.createElement("div");
    identity.className = "download-models-model-identity";
    const providerIdentity = document.createElement("p");
    providerIdentity.className = "download-models-model-provider-id";
    providerIdentity.textContent = catalogRow.huggingfaceId;
    const fitLabel = createDownloadModelsFitLabel(catalogRow);
    identity.replaceChildren(
        createDownloadModelsCardTitleLine(catalogRow),
        providerIdentity,
        ...(fitLabel ? [fitLabel] : [])
    );
    const availability = createDownloadModelsAvailabilityBadge(catalogRow);
    header.replaceChildren(identity, ...(availability ? [availability] : []));
    return header;
}

// A badge only where it tells something: the model is on this Mac, or is on
// its way. A model that is neither already shows a Download button, so an
// "Available" badge beside it repeated the button and was dropped.
function createDownloadModelsAvailabilityBadge(catalogRow) {
    const isDownloading = isDownloadModelsRowDownloading(catalogRow);
    if (!catalogRow.readyOnThisMac && !isDownloading) {
        return null;
    }
    const availability = document.createElement("span");
    availability.className = catalogRow.readyOnThisMac
        ? "download-models-availability download-models-availability-ready"
        : "download-models-availability download-models-availability-downloading";
    availability.textContent = catalogRow.readyOnThisMac ? "On this Mac" : "Downloading";
    return availability;
}

// The download size is the first thing someone weighs when picking between two
// versions of a model, so it sits on the title line in large type, right beside
// the name, rather than in small print underneath.
function createDownloadModelsCardTitleLine(catalogRow) {
    const titleLine = document.createElement("div");
    titleLine.className = "download-models-model-title-line";
    const displayName = document.createElement("h4");
    displayName.textContent = catalogRow.displayName;
    const downloadSize = document.createElement("span");
    downloadSize.className = "download-models-model-size";
    downloadSize.textContent = formatDownloadModelsCardSize(catalogRow.approximateSizeBytes);
    downloadSize.title = "Download size";
    titleLine.replaceChildren(displayName, downloadSize);
    return titleLine;
}

function isDownloadModelsRowDownloading(catalogRow) {
    const isActiveDownload = downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const downloadState = isActiveDownload
        ? downloadModelsCurrentDownload.state
        : catalogRow.downloadState;
    return downloadState === "downloading"
        || downloadState === "paused"
        || downloadState === "verifying"
        || downloadState === "publishing";
}

function createDownloadModelsProgress(catalogRow) {
    const isActiveDownload = downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const downloadState = isActiveDownload
        ? downloadModelsCurrentDownload.state
        : catalogRow.downloadState;
    const track = document.createElement("div");
    track.className = "download-models-download-progress";
    if (!isActiveDownload || !(downloadModelsCurrentDownload.bytes_total > 0)
        || !["downloading", "paused", "verifying", "publishing"].includes(downloadState)) {
        return track;
    }
    const percent = downloadState === "publishing" || downloadState === "verifying"
        ? 100
        : downloadModelsProgressPercent(downloadModelsCurrentDownload);
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(percent));
    const fill = document.createElement("div");
    fill.className = "download-models-download-progress-fill";
    fill.style.width = percent + "%";
    track.appendChild(fill);
    return track;
}

function createDownloadModelsDetailsPanel(catalogRow, allRows) {
    const panel = document.createElement("section");
    panel.className = "download-models-model-details";
    const heading = document.createElement("p");
    heading.className = "download-models-details-heading";
    heading.textContent = "Details";
    panel.appendChild(heading);
    const detailList = document.createElement("dl");
    detailList.className = "download-models-detail-list";
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
    const destinationPath = downloadModelsDestinationPath(catalogRow);
    if (destinationPath) {
        panel.appendChild(createDownloadModelsLocationSection(destinationPath, catalogRow));
    }
    return panel;
}

function createDownloadModelsLocationSection(destinationPath, catalogRow) {
    const location = document.createElement("section");
    location.className = "download-models-model-location";
    const locationLabel = document.createElement("p");
    locationLabel.className = "download-models-model-location-label";
    locationLabel.textContent = "Model location";
    const locationValue = document.createElement("code");
    locationValue.className = "download-models-download-path";
    locationValue.textContent = destinationPath;
    const actionRow = document.createElement("div");
    actionRow.className = "download-models-location-actions";
    const copyButton = document.createElement("button");
    copyButton.className = "button button-secondary download-models-copy-path-button";
    copyButton.type = "button";
    copyButton.textContent = "Copy Path";
    copyButton.addEventListener("click", () => copyDownloadModelsPathToClipboard(destinationPath));
    // Finder can only show a folder that exists, so the button appears once the
    // model is on this Mac; before that the path is where it will be saved.
    const actionButtons = catalogRow.readyOnThisMac
        ? [createDownloadModelsRevealButton(catalogRow.huggingfaceId), copyButton]
        : [copyButton];
    actionRow.replaceChildren(...actionButtons);
    location.replaceChildren(locationLabel, locationValue, actionRow);
    return location;
}

function createDownloadModelsRevealButton(huggingfaceId) {
    const revealButton = document.createElement("button");
    revealButton.className = "button button-secondary download-models-reveal-button";
    revealButton.type = "button";
    revealButton.textContent = "Show in Finder";
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
