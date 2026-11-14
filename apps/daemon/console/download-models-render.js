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
    const hasModelsOrDownload = installedRows.length > 0 || activeDownloadRow !== null;
    const pageSections = [
        hasModelsOrDownload
            ? createDownloadModelsInstalledSection(installedRows, activeDownloadRow)
            : createDownloadModelsFirstVisitSection(catalogRows)
    ];
    const linkedSection = createDownloadModelsLinkedSection(catalogRows);
    if (linkedSection) pageSections.unshift(linkedSection);
    const removalNotice = createDownloadModelsRemovalNotice(catalogRows);
    if (removalNotice) pageSections.push(removalNotice);
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
    const fitMeter = createDownloadModelsFitMeter(starterRow, " on this Mac");
    if (fitMeter) pickChildren.push(fitMeter);
    pick.replaceChildren(...pickChildren);
    cardChildren.push(pick, createDownloadModelsGetButton(starterRow, isRecommended));
    card.replaceChildren(...cardChildren);
    return card;
}

function createDownloadModelsAddSection(catalogRows) {
    const section = createDownloadModelsElement("section", "download-models-add");
    const visibleSectionKeys = downloadModelsVisibleSectionKeys(downloadModelsActiveFilter, downloadModelsSearchQuery);
    const sectionElements = DOWNLOAD_MODELS_SECTIONS
        .filter((catalogSection) => visibleSectionKeys.includes(catalogSection.key))
        .map((catalogSection) => createDownloadModelsCatalogSection(catalogSection, catalogRows))
        .filter((sectionElement) => sectionElement !== null);
    const sectionChildren = [createDownloadModelsAddHeading(), createDownloadModelsFilterBar()];
    if (isDownloadModelsDownloadSlotBusy()) {
        sectionChildren.push(createDownloadModelsParagraph(
            "download-models-busy-hint",
            "Downloads run one at a time. You can get another model when this one finishes."
        ));
    }
    sectionChildren.push(...sectionElements);
    if (sectionElements.length === 0) {
        sectionChildren.push(createDownloadModelsParagraph(
            "download-models-no-matches",
            "No model matches that search. Try a shorter word, or clear it to see everything."
        ));
    }
    const showMoreButton = createDownloadModelsShowMoreButton(catalogRows);
    if (showMoreButton) sectionChildren.push(showMoreButton);
    section.replaceChildren(...sectionChildren);
    return section;
}

// The memory figure sits beside the list because it is what every speed meter
// in the list is measured against.
function createDownloadModelsAddHeading() {
    const heading = createDownloadModelsElement("div", "download-models-section-heading");
    const headingChildren = [createDownloadModelsElement("h3", "", "Add a model")];
    if (downloadModelsServingCeilingBytes > 0) {
        headingChildren.push(createDownloadModelsElement(
            "span",
            "download-models-section-meta",
            "This Mac gives models " + formatDownloadModelsCardSize(downloadModelsServingCeilingBytes) + " of memory"
        ));
    }
    heading.replaceChildren(...headingChildren);
    return heading;
}

function createDownloadModelsFilterBar() {
    const filterBar = createDownloadModelsElement("div", "download-models-filter-bar");
    const chipGroup = createDownloadModelsElement("div", "download-models-chips");
    chipGroup.setAttribute("role", "group");
    chipGroup.setAttribute("aria-label", "What do you want to do?");
    chipGroup.replaceChildren(...DOWNLOAD_MODELS_FILTERS.map(createDownloadModelsFilterChip));
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
function createDownloadModelsFilterChip(filterDefinition) {
    const isPressed = downloadModelsActiveFilter === filterDefinition.key && !downloadModelsSearchQuery;
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
    rowList.replaceChildren(...sectionRows.map(createDownloadModelsModelRow));
    section.replaceChildren(heading, rowList);
    return section;
}

function createDownloadModelsShowMoreButton(catalogRows) {
    if (downloadModelsActiveFilter !== "all" || downloadModelsSearchQuery) return null;
    const hiddenModelCount = downloadModelsHiddenModelCount(catalogRows);
    if (hiddenModelCount === 0) return null;
    const showMoreButton = createDownloadModelsElement(
        "button",
        "button button-secondary download-models-show-more",
        "Show " + hiddenModelCount + " more: helpers, experimental builds and models leaving soon"
    );
    showMoreButton.type = "button";
    showMoreButton.addEventListener("click", () => selectDownloadModelsFilter("more"));
    return showMoreButton;
}

// Three bars, like a signal meter, plus the words: the count differs as well
// as the wording, so the verdict never rests on colour.
function createDownloadModelsFitMeter(catalogRow, labelSuffix = "") {
    const fitOutcome = downloadModelsFitOutcome(
        catalogRow, downloadModelsServingCeilingBytes, downloadModelsCeilingIsUserLimit
    );
    if (fitOutcome === null) return null;
    const fitCopy = DOWNLOAD_MODELS_FIT_COPY[fitOutcome];
    const meter = createDownloadModelsElement("span", "download-models-fit download-models-fit-" + fitOutcome);
    const bars = createDownloadModelsElement("span", "download-models-fit-bars");
    bars.setAttribute("aria-hidden", "true");
    const barElements = [];
    for (let barNumber = 1; barNumber <= 3; barNumber += 1) {
        barElements.push(createDownloadModelsElement(
            "span",
            barNumber <= fitCopy.filledBarCount ? "download-models-fit-bar download-models-fit-bar-filled" : "download-models-fit-bar"
        ));
    }
    bars.replaceChildren(...barElements);
    meter.replaceChildren(bars, createDownloadModelsElement("span", "download-models-fit-label", fitCopy.label + labelSuffix));
    return meter;
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
    getButton.disabled = isDownloadModelsDownloadSlotBusy();
    focusDownloadModelsRemovalControlIfRequested(catalogRow, "download", getButton);
    return getButton;
}
