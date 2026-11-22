// Astronomical Observatory Models page assembly: what is on this Mac (or the
// first-visit choice when nothing is), then the Add a model list with its
// filter chips and search. All catalog values enter the document only through
// textContent and created elements.

const DOWNLOAD_MODELS_STARTER_CARDS = [
    { role: "chat", title: "Chat", promise: "Talk, write, and ask about photos." },
    { role: "image_generation", title: "Make pictures", promise: "Turn a written description into a picture." },
    { role: "coding", title: "Write code", promise: "Help writing, reading, and fixing code." }
];

function createDownloadModelsElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}

function createDownloadModelsParagraph(className, text) {
    return createDownloadModelsElement("p", className, text);
}

function createDownloadModelsPageSections(catalogRows) {
    const installedRows = catalogRows.filter((catalogRow) => catalogRow.readyOnThisMac);
    const activeDownloadRow = downloadModelsActiveDownloadRow(catalogRows);
    const hasModelsOrDownload = installedRows.length > 0 || activeDownloadRow !== null
        || hasDownloadModelsQueueEntries();
    const pageSections = [
        hasModelsOrDownload
            ? createDownloadModelsInstalledSection(installedRows, activeDownloadRow, catalogRows)
            : createDownloadModelsFirstVisitSection(catalogRows)
    ];
    // A first visit already asks what to do first; suggestions start once a
    // model is here, when the question becomes what would be better.
    const suggestionsSection = hasModelsOrDownload ? createDownloadModelsSuggestionsSection(catalogRows) : null;
    if (suggestionsSection) pageSections.push(suggestionsSection);
    const linkedSection = createDownloadModelsLinkedSection(catalogRows);
    if (linkedSection) pageSections.unshift(linkedSection);
    const removalNotice = createDownloadModelsRemovalNotice(catalogRows);
    if (removalNotice) pageSections.push(removalNotice);
    const queueNotice = createDownloadModelsQueueNotice();
    if (queueNotice) pageSections.push(queueNotice);
    pageSections.push(createDownloadModelsAddSection(downloadModelsRowsInPageOrder(catalogRows)));
    return pageSections;
}

// After a move to the Trash the model has no line under On this Mac left to
// carry the outcome, so it is announced at page level. A refusal stays on the
// line of the model it is about (see createDownloadModelsInstalledLine).
function createDownloadModelsRemovalNotice(catalogRows) {
    for (const catalogRow of catalogRows) {
        if (catalogRow.readyOnThisMac) continue;
        const noticeText = downloadModelsRemovalNoticeFor(catalogRow);
        if (noticeText) return createDownloadModelsNotice(noticeText, "download-models-notice download-models-notice-page");
    }
    return null;
}

// A removal outcome is announced to assistive technology; it changes once, not
// every few seconds like download progress.
function createDownloadModelsNotice(noticeText, className = "download-models-notice") {
    const notice = createDownloadModelsParagraph(className, noticeText);
    notice.setAttribute("role", "status");
    return notice;
}

// First visit: one question and three answers, each a single download, so a
// newcomer never has to read the catalog to get going.
function createDownloadModelsFirstVisitSection(catalogRows) {
    const section = createDownloadModelsElement("section", "download-models-first-visit");
    const cardGrid = createDownloadModelsElement("div", "download-models-starter-grid");
    const starterCards = [];
    for (const starterCard of DOWNLOAD_MODELS_STARTER_CARDS) {
        const starterRow = catalogRows.find(
            (catalogRow) => catalogRow.role === starterCard.role && catalogRow.getStarted
        );
        if (starterRow) {
            starterCards.push(createDownloadModelsStarterCard(starterCard, starterRow, starterCards.length === 0));
        }
    }
    cardGrid.replaceChildren(...starterCards);
    section.replaceChildren(
        createDownloadModelsElement("h3", "download-models-first-visit-title", "What would you like to do first?"),
        createDownloadModelsParagraph(
            "download-models-first-visit-lede",
            "Pick one. It downloads once, then runs entirely on this Mac. You can add the others any time."
        ),
        cardGrid
    );
    return section;
}

function createDownloadModelsStarterCard(starterCard, starterRow, isRecommended) {
    const card = createDownloadModelsElement(
        "article",
        isRecommended ? "download-models-starter-card download-models-starter-card-recommended" : "download-models-starter-card"
    );
    const cardChildren = [];
    if (isRecommended) {
        cardChildren.push(createDownloadModelsElement("span", "download-models-starter-tag", "Start here"));
    }
    cardChildren.push(
        createDownloadModelsElement("h4", "download-models-starter-title", starterCard.title),
        createDownloadModelsParagraph("download-models-starter-promise", starterCard.promise)
    );
    const pick = createDownloadModelsElement("div", "download-models-starter-pick");
    const pickChildren = [
        createDownloadModelsParagraph("download-models-starter-model", downloadModelsRowName(starterRow))
    ];
    const fitMeter = createDownloadModelsFitMeter(starterRow);
    if (fitMeter) pickChildren.push(fitMeter);
    pick.replaceChildren(...pickChildren);
    cardChildren.push(pick, createDownloadModelsGetButton(starterRow, isRecommended));
    card.replaceChildren(...cardChildren);
    return card;
}

// Suggested for this Mac: a few cards, each a model and why it would be better
// here. Only the first card's Get is the strong button, so the page asks for
// one decision at a time.
function createDownloadModelsSuggestionsSection(catalogRows) {
    const suggestions = downloadModelsSuggestionCards(downloadModelsCatalogDocument, catalogRows);
    if (suggestions.length === 0) return null;
    const section = createDownloadModelsElement("section", "download-models-suggestions");
    const cardGrid = createDownloadModelsElement("div", "download-models-suggestion-grid");
    cardGrid.replaceChildren(...suggestions.map(
        (suggestion, suggestionIndex) => createDownloadModelsSuggestionCard(suggestion, suggestionIndex === 0)
    ));
    section.replaceChildren(
        createDownloadModelsElement("h3", "download-models-suggestions-title", "Suggested for this Mac"),
        cardGrid
    );
    return section;
}

function createDownloadModelsSuggestionCard(suggestion, isFirst) {
    const card = createDownloadModelsElement(
        "article", "download-models-suggestion-card download-models-suggestion-" + suggestion.kind.replace(/_/g, "-")
    );
    const cardChildren = [
        createDownloadModelsElement("span", "download-models-suggestion-tag", suggestion.tag),
        createDownloadModelsParagraph("download-models-suggestion-model", downloadModelsRowName(suggestion.catalogRow)),
        createDownloadModelsParagraph("download-models-suggestion-reason", suggestion.reason)
    ];
    const fitMeter = createDownloadModelsFitMeter(suggestion.catalogRow);
    if (fitMeter) cardChildren.push(fitMeter);
    cardChildren.push(createDownloadModelsRowAction(suggestion.catalogRow, isFirst));
    card.replaceChildren(...cardChildren);
    return card;
}

function createDownloadModelsAddSection(catalogRows) {
    const section = createDownloadModelsElement("section", "download-models-add");
    const filterChips = downloadModelsFilterChips(catalogRows);
    // A chip whose section emptied (its last model removed) falls back to All.
    const activeFilterKey = filterChips.some((filterChip) => filterChip.key === downloadModelsActiveFilter)
        ? downloadModelsActiveFilter
        : "all";
    const visibleSectionKeys = downloadModelsVisibleSectionKeys(activeFilterKey, downloadModelsSearchQuery);
    const sectionElements = DOWNLOAD_MODELS_SECTIONS
        .filter((catalogSection) => visibleSectionKeys.includes(catalogSection.key))
        .map((catalogSection) => createDownloadModelsCatalogSection(catalogSection, catalogRows))
        .filter((sectionElement) => sectionElement !== null);
    const sectionChildren = [createDownloadModelsAddHeading(), createDownloadModelsFilterBar(filterChips, activeFilterKey)];
    sectionChildren.push(...sectionElements);
    if (sectionElements.length === 0) {
        sectionChildren.push(createDownloadModelsParagraph(
            "download-models-no-matches",
            "No model matches that search. Try a shorter word, or clear it to see everything."
        ));
    }
    section.replaceChildren(...sectionChildren);
    return section;
}

// The memory figure sits beside the list because it is what every memory bar
// in the list is measured against; the free disk figure beside it is the other
// limit a download meets.
function createDownloadModelsAddHeading() {
    const heading = createDownloadModelsElement("div", "download-models-section-heading");
    const headingChildren = [createDownloadModelsElement("h3", "", "Add a model")];
    const metaText = downloadModelsAddHeadingMetaText(downloadModelsServingCeilingBytes, downloadModelsFreeDiskBytes());
    if (metaText) {
        headingChildren.push(createDownloadModelsElement("span", "download-models-section-meta", metaText));
    }
    heading.replaceChildren(...headingChildren);
    return heading;
}

// "This Mac gives models 36.5 GB of memory · 362 GB free for downloads".
// Free space is the figure downloads are checked against, which leaves out
// space macOS could free on request, so it can read lower than Finder.
function downloadModelsAddHeadingMetaText(servingCeilingBytes, freeDiskBytes) {
    const metaParts = [];
    if (servingCeilingBytes > 0) {
        metaParts.push("This Mac gives models " + formatDownloadModelsCardSize(servingCeilingBytes) + " of memory");
    }
    if (freeDiskBytes !== null) metaParts.push(formatDownloadModelsFreeDiskText(freeDiskBytes) + " free for downloads");
    return metaParts.join(" · ");
}

function createDownloadModelsFilterBar(filterChips, activeFilterKey) {
    const filterBar = createDownloadModelsElement("div", "download-models-filter-bar");
    const chipGroup = createDownloadModelsElement("div", "download-models-chips");
    chipGroup.setAttribute("role", "group");
    chipGroup.setAttribute("aria-label", "What do you want to do?");
    chipGroup.replaceChildren(...filterChips.map(
        (filterChip) => createDownloadModelsFilterChip(filterChip, activeFilterKey)
    ));
    const searchInput = createDownloadModelsElement("input", "download-models-search");
    searchInput.id = "download-models-search";
    searchInput.type = "search";
    searchInput.placeholder = "Search";
    searchInput.setAttribute("aria-label", "Search models");
    // The state stores only the lowercase query used for matching; the input
    // keeps the exact text the person is typing.
    const previousSearchInput = document.getElementById("download-models-search");
    searchInput.value = previousSearchInput ? previousSearchInput.value : downloadModelsSearchQuery;
    filterBar.replaceChildren(chipGroup, searchInput);
    // Wire after insertion so the coordinator's listener attaches to the input
    // that is in the page.
    setTimeout(wireDownloadModelsControls, 0);
    return filterBar;
}

// While a search is active no chip is pressed: the search spans every section.
function createDownloadModelsFilterChip(filterDefinition, activeFilterKey) {
    const isPressed = activeFilterKey === filterDefinition.key && !downloadModelsSearchQuery;
    const chip = createDownloadModelsElement(
        "button",
        isPressed ? "download-models-chip download-models-chip-pressed" : "download-models-chip",
        filterDefinition.label
    );
    chip.type = "button";
    chip.setAttribute("aria-pressed", String(isPressed));
    chip.addEventListener("click", () => selectDownloadModelsFilter(filterDefinition.key));
    return chip;
}

function selectDownloadModelsFilter(filterKey) {
    downloadModelsActiveFilter = filterKey;
    renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
}

function createDownloadModelsCatalogSection(catalogSection, catalogRows) {
    const sectionRows = catalogRows.filter(
        (catalogRow) => downloadModelsSectionKey(catalogRow) === catalogSection.key
            && downloadModelsRowMatchesQuery(catalogRow, downloadModelsSearchQuery)
    );
    if (sectionRows.length === 0) return null;
    const section = createDownloadModelsElement("section", "download-models-group");
    const heading = createDownloadModelsElement("div", "download-models-group-heading");
    heading.replaceChildren(
        createDownloadModelsElement("h4", "", catalogSection.title),
        createDownloadModelsElement("span", "download-models-group-note", catalogSection.note)
    );
    const rowList = createDownloadModelsElement("div", "download-models-list");
    rowList.replaceChildren(...downloadModelsFoldedRows(sectionRows).flatMap(createDownloadModelsFoldedRowElements));
    section.replaceChildren(heading, rowList);
    return section;
}


// A bar for the share of this Mac's model memory the model uses, plus the
// words: "23.0 of 36.5 GB memory" when it fits, what happens when it does not.
// A model that does not fit draws a full, hollow bar, so the verdict never
// rests on colour.
function createDownloadModelsFitMeter(catalogRow) {
    const fitOutcome = downloadModelsFitOutcome(
        catalogRow, downloadModelsServingCeilingBytes, downloadModelsCeilingIsUserLimit
    );
    if (fitOutcome === null) return null;
    const meter = createDownloadModelsElement("span", "download-models-fit download-models-fit-" + fitOutcome);
    const bar = createDownloadModelsElement("span", "download-models-fit-bar");
    bar.setAttribute("aria-hidden", "true");
    const barFill = createDownloadModelsElement("span", "download-models-fit-bar-fill");
    barFill.style.width = downloadModelsMemoryBarPercent(catalogRow, downloadModelsServingCeilingBytes) + "%";
    bar.replaceChildren(barFill);
    meter.replaceChildren(
        createDownloadModelsElement(
            "span",
            "download-models-fit-label",
            downloadModelsFitLabel(catalogRow, downloadModelsServingCeilingBytes, downloadModelsCeilingIsUserLimit)
        ),
        bar
    );
    return meter;
}

// A lead row, then either a control that shows its other builds or the builds
// themselves under it. Most people want one build of a model; the others are a
// click away and each keeps its own name.
function createDownloadModelsFoldedRowElements(foldedRow) {
    const rowElements = [createDownloadModelsModelRow(foldedRow.catalogRow)];
    if (foldedRow.buildRows.length === 0) return rowElements;
    const leadModelId = foldedRow.catalogRow.modelId;
    const isUnfolded = downloadModelsUnfoldedLeadIds.includes(leadModelId) || Boolean(downloadModelsSearchQuery);
    if (!downloadModelsSearchQuery) {
        rowElements.push(createDownloadModelsFoldToggle(leadModelId, foldedRow.buildRows.length, isUnfolded));
    }
    if (isUnfolded) {
        const buildRows = [...foldedRow.buildRows].sort(
            (smallerRow, largerRow) => smallerRow.approximateMemoryBytes - largerRow.approximateMemoryBytes
        );
        for (const buildRow of buildRows) {
            rowElements.push(createDownloadModelsModelRow(buildRow, foldedRow.catalogRow));
        }
    }
    return rowElements;
}

function createDownloadModelsFoldToggle(leadModelId, buildCount, isUnfolded) {
    const buildsText = buildCount === 1 ? "1 other build" : buildCount + " other builds";
    const toggle = createDownloadModelsElement(
        "button",
        "download-models-fold-toggle",
        (isUnfolded ? "Hide " : "Show ") + buildsText + " of this model"
    );
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", String(isUnfolded));
    toggle.addEventListener("click", () => {
        downloadModelsUnfoldedLeadIds = isUnfolded
            ? downloadModelsUnfoldedLeadIds.filter((unfoldedLeadId) => unfoldedLeadId !== leadModelId)
            : [...downloadModelsUnfoldedLeadIds, leadModelId];
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    });
    return toggle;
}

// The button carries the size, the one number people weigh before a download.
function createDownloadModelsGetButton(catalogRow, isPrimary = true) {
    const getButton = downloadModelsActionButton(
        "Get · " + formatDownloadModelsCardSize(catalogRow.approximateSizeBytes),
        "start",
        catalogRow.huggingfaceId,
        "download-models-download-button"
    );
    getButton.className = (isPrimary ? "button button-primary" : "button button-secondary")
        + " download-models-download-button";
    focusDownloadModelsRemovalControlIfRequested(catalogRow, "download", getButton);
    return getButton;
}
