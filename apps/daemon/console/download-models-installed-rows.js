// Astronomical Observatory Models page: the rows of On this Mac. Each row reads
// left to right as a quiet table: what kind of model it is (an icon and a
// word), its name, its size, whether it is in memory now, one Chat button for
// a chat model, and a ⋯ menu that holds everything rarer (Show in Finder, Copy
// model ID, Details, Open on Hugging Face, Move to Trash). One visible action
// per row keeps the column calm; the menu keeps the rest one click away.

// How the rows are ordered: by kind first, so chat models sit together, unless
// the person clicks another column.
let downloadModelsInstalledSortKey = "kind";
let downloadModelsInstalledSortAscending = true;
// The model whose ⋯ menu is open, by Hugging Face identity, or null.
let downloadModelsOpenMenuHuggingfaceId = null;

const DOWNLOAD_MODELS_INSTALLED_SORT_COLUMNS = [
    { key: "name", label: "Model" },
    { key: "kind", label: "Kind" },
    { key: "size", label: "Size" }
];
const DOWNLOAD_MODELS_HUGGING_FACE_MODEL_URL = "https://huggingface.co/";
// A copy in the shared Hugging Face folder belongs to every app that reads it
// (oMLX, LM Studio, Stable), so the page names where it lives and never moves
// it to the Trash.
const DOWNLOAD_MODELS_HUGGING_FACE_FOLDER_MARKER = "/.cache/huggingface/";

// The model the engine has loaded now, from the status poll in console.js; the
// page shows no status while it is unknown.
function downloadModelsLoadedModelId() {
    return typeof loadedModelId === "string" ? loadedModelId : null;
}

function isDownloadModelsRowInMemory(installedRow) {
    return downloadModelsLoadedModelId() === downloadModelsRowName(installedRow);
}

function isDownloadModelsRowInHuggingFaceFolder(installedRow) {
    return (installedRow.destinationDirectory || "").includes(DOWNLOAD_MODELS_HUGGING_FACE_FOLDER_MARKER);
}

function downloadModelsInstalledKindTitle(installedRow) {
    const sectionKey = downloadModelsSectionKey(installedRow);
    return DOWNLOAD_MODELS_SECTIONS.find((section) => section.key === sectionKey)?.title || "";
}

function downloadModelsInstalledRowsInOrder(installedRows) {
    const sortValue = {
        name: (installedRow) => downloadModelsRowName(installedRow).toLowerCase(),
        kind: (installedRow) => DOWNLOAD_MODELS_SECTIONS.findIndex(
            (section) => section.key === downloadModelsSectionKey(installedRow)
        ),
        size: (installedRow) => installedRow.approximateSizeBytes
    }[downloadModelsInstalledSortKey];
    const direction = downloadModelsInstalledSortAscending ? 1 : -1;
    return [...installedRows].sort((leftRow, rightRow) => {
        const leftValue = sortValue(leftRow);
        const rightValue = sortValue(rightRow);
        const order = leftValue < rightValue ? -1 : leftValue > rightValue ? 1 : 0;
        return direction * order
            || downloadModelsRowName(leftRow).localeCompare(downloadModelsRowName(rightRow));
    });
}

// Column names above the rows; Model, Kind and Size sort, a second click
// reverses.
function createDownloadModelsInstalledHeader() {
    const header = createDownloadModelsElement("div", "download-models-installed-header");
    const headerCells = [createDownloadModelsElement("span", "")];
    for (const sortColumn of DOWNLOAD_MODELS_INSTALLED_SORT_COLUMNS) {
        headerCells.push(createDownloadModelsInstalledSortButton(sortColumn));
    }
    headerCells.push(
        createDownloadModelsElement("span", "download-models-installed-column-label", "Status"),
        createDownloadModelsElement("span", "")
    );
    header.replaceChildren(...headerCells);
    return header;
}

function createDownloadModelsInstalledSortButton(sortColumn) {
    const isSorted = downloadModelsInstalledSortKey === sortColumn.key;
    const directionText = isSorted ? (downloadModelsInstalledSortAscending ? " ↑" : " ↓") : "";
    const sortButton = createDownloadModelsElement(
        "button",
        "download-models-installed-sort download-models-installed-sort-" + sortColumn.key
            + (isSorted ? " download-models-installed-sort-active" : ""),
        sortColumn.label + directionText
    );
    sortButton.type = "button";
    sortButton.setAttribute("aria-sort", isSorted ? (downloadModelsInstalledSortAscending ? "ascending" : "descending") : "none");
    sortButton.addEventListener("click", () => {
        downloadModelsInstalledSortAscending = isSorted ? !downloadModelsInstalledSortAscending : true;
        downloadModelsInstalledSortKey = sortColumn.key;
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    });
    return sortButton;
}

function createDownloadModelsInstalledLine(installedRow) {
    const line = createDownloadModelsElement("div", "download-models-installed-line");
    const lineChildren = [
        createDownloadModelsKindIcon(installedRow),
        createDownloadModelsInstalledIdentity(installedRow),
        createDownloadModelsElement("span", "download-models-installed-kind", downloadModelsInstalledKindTitle(installedRow)),
        createDownloadModelsElement(
            "span", "download-models-installed-size", formatDownloadModelsCardSize(installedRow.approximateSizeBytes)
        ),
        createDownloadModelsInstalledStatus(installedRow),
        createDownloadModelsInstalledActions(installedRow)
    ];
    if (downloadModelsOpenMenuHuggingfaceId === installedRow.huggingfaceId) {
        lineChildren.push(createDownloadModelsRowMenu(installedRow));
    }
    const removalStrip = createDownloadModelsRemovalStrip(installedRow);
    if (removalStrip) lineChildren.push(removalStrip);
    line.replaceChildren(...lineChildren);
    return line;
}

// The kind icon is a button only where a click does something of its own: a
// chat model's opens the chat on it, a read-aloud model's plays a short
// sample. Every other kind keeps a plain tile with no hover, so nothing looks
// clickable that is not.
function createDownloadModelsKindIcon(installedRow) {
    const kindClassName = "download-models-kind-icon download-models-kind-icon-" + downloadModelsSectionKey(installedRow);
    if (canChatWithCatalogModel(installedRow)) {
        const chatIcon = createDownloadModelsElement("button", kindClassName + " download-models-kind-icon-action");
        chatIcon.type = "button";
        chatIcon.title = "Open in chat";
        chatIcon.setAttribute("aria-label", "Open " + downloadModelsRowName(installedRow) + " in chat");
        chatIcon.addEventListener("click", () => openThinTalkFromDownloadModels(downloadModelsRowName(installedRow)));
        return chatIcon;
    }
    if (installedRow.supportsSpeech) return createDownloadModelsVoiceSampleIcon(installedRow, kindClassName);
    return createDownloadModelsElement("span", kindClassName);
}

function createDownloadModelsVoiceSampleIcon(installedRow, kindClassName) {
    const phase = downloadModelsVoiceSamplePhase(installedRow.huggingfaceId);
    const voiceIcon = createDownloadModelsElement(
        "button",
        kindClassName + " download-models-kind-icon-action" + (phase ? " download-models-kind-icon-" + phase : "")
    );
    voiceIcon.type = "button";
    voiceIcon.title = downloadModelsVoiceSampleTooltip(installedRow.huggingfaceId);
    voiceIcon.setAttribute(
        "aria-label",
        (phase === "playing" ? "Stop the sample of " : "Hear a sample of ") + downloadModelsRowName(installedRow)
    );
    voiceIcon.setAttribute("aria-pressed", String(phase === "playing"));
    voiceIcon.addEventListener("click", () => void toggleDownloadModelsVoiceSample(installedRow));
    return voiceIcon;
}

function createDownloadModelsInstalledIdentity(installedRow) {
    const identity = createDownloadModelsElement("div", "download-models-installed-identity");
    const identityChildren = [
        createDownloadModelsElement("span", "download-models-installed-name", downloadModelsRowName(installedRow))
    ];
    if (isDownloadModelsRowInHuggingFaceFolder(installedRow)) {
        identityChildren.push(createDownloadModelsElement(
            "span", "download-models-installed-source", "From your Hugging Face folder"
        ));
    }
    // Why a removal was refused, on the line of the model it is about.
    const refusalText = downloadModelsRemovalNoticeFor(installedRow);
    if (refusalText) identityChildren.push(createDownloadModelsNotice(refusalText));
    identity.replaceChildren(...identityChildren);
    return identity;
}

function createDownloadModelsInstalledStatus(installedRow) {
    const status = createDownloadModelsElement("span", "download-models-installed-status");
    if (isDownloadModelsRowInMemory(installedRow)) {
        const inMemoryBadge = createDownloadModelsElement("span", "download-models-in-memory", "In memory");
        inMemoryBadge.title = "Loaded in memory right now";
        status.replaceChildren(inMemoryBadge);
    }
    return status;
}

function createDownloadModelsInstalledActions(installedRow) {
    const actions = createDownloadModelsElement("div", "download-models-installed-actions");
    const actionControls = [];
    if (canChatWithCatalogModel(installedRow)) actionControls.push(createDownloadModelsOpenInChatButton(installedRow));
    actionControls.push(createDownloadModelsMenuButton(installedRow));
    actions.replaceChildren(...actionControls);
    return actions;
}

function createDownloadModelsMenuButton(installedRow) {
    const isOpen = downloadModelsOpenMenuHuggingfaceId === installedRow.huggingfaceId;
    const menuButton = createDownloadModelsElement(
        "button", isOpen ? "download-models-menu-button download-models-menu-button-open" : "download-models-menu-button", "⋯"
    );
    menuButton.type = "button";
    menuButton.setAttribute("aria-label", "More for " + downloadModelsRowName(installedRow));
    menuButton.setAttribute("aria-haspopup", "menu");
    menuButton.setAttribute("aria-expanded", String(isOpen));
    menuButton.addEventListener("click", () => {
        downloadModelsOpenMenuHuggingfaceId = isOpen ? null : installedRow.huggingfaceId;
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    });
    // Keep in the removal question returns focus here, where Remove now lives.
    focusDownloadModelsRemovalControlIfRequested(installedRow, "remove", menuButton);
    return menuButton;
}

function createDownloadModelsRowMenu(installedRow) {
    const menu = createDownloadModelsElement("div", "download-models-row-menu");
    menu.setAttribute("role", "menu");
    menu.setAttribute("aria-label", "Actions for " + downloadModelsRowName(installedRow));
    menu.replaceChildren(
        createDownloadModelsMenuItem("Show in Finder", "finder", () => revealDownloadModelsModelInFinder(installedRow.huggingfaceId)),
        createDownloadModelsMenuItem("Copy model ID", "copy", () => copyDownloadModelsTextToClipboard(downloadModelsRowName(installedRow))),
        createDownloadModelsMenuItem("Details", "details", () => showDownloadModelsRowDetails(installedRow)),
        createDownloadModelsMenuItem("Open on Hugging Face", "external", () => openDownloadModelsOnHuggingFace(installedRow)),
        createDownloadModelsElement("div", "download-models-row-menu-separator"),
        createDownloadModelsRemovalMenuItem(installedRow)
    );
    return menu;
}

function createDownloadModelsMenuItem(label, iconName, onChoose) {
    const menuItem = createDownloadModelsElement(
        "button", "download-models-row-menu-item download-models-menu-icon-" + iconName, label
    );
    menuItem.type = "button";
    menuItem.setAttribute("role", "menuitem");
    menuItem.addEventListener("click", () => {
        downloadModelsOpenMenuHuggingfaceId = null;
        onChoose();
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    });
    return menuItem;
}

// Move to Trash only for a copy the engine reports as removable (one Download
// Models installed, with a Trash to move it to). The menu names a reason only
// when the path proves it, a shared Hugging Face folder copy; any other refusal
// may have several causes, so it just points to Finder.
function createDownloadModelsRemovalMenuItem(installedRow) {
    if (!canRemoveDownloadModelsModelFromDisk(installedRow)) {
        const keptText = isDownloadModelsRowInHuggingFaceFolder(installedRow)
            ? "Managed in your Hugging Face folder"
            : "Remove it in Finder instead";
        const keptNote = createDownloadModelsElement(
            "span", "download-models-row-menu-item download-models-row-menu-note download-models-menu-icon-lock", keptText
        );
        keptNote.setAttribute("role", "menuitem");
        keptNote.setAttribute("aria-disabled", "true");
        return keptNote;
    }
    const trashItem = createDownloadModelsMenuItem(
        "Move to Trash…", "trash", () => askToRemoveDownloadModelsModel(installedRow)
    );
    trashItem.className += " download-models-row-menu-danger download-models-remove-button";
    // One removal at a time, as before: another model's removal must finish first.
    trashItem.disabled = downloadModelsRemovalInFlightId !== null;
    return trashItem;
}

// The removal question (or its progress) takes the whole width under the row.
function createDownloadModelsRemovalStrip(installedRow) {
    const isAsking = downloadModelsRemovalConfirmingId === installedRow.huggingfaceId;
    const isMoving = downloadModelsRemovalInFlightId === installedRow.huggingfaceId;
    if (!isAsking && !isMoving) return null;
    const strip = createDownloadModelsElement("div", "download-models-removal-strip");
    strip.replaceChildren(...createDownloadModelsRemovalControls(installedRow));
    return strip;
}

// Details live on the model's row in Add a model: open it there, unfolding the
// build's lead and clearing any filter that would hide it.
function showDownloadModelsRowDetails(installedRow) {
    downloadModelsActiveFilter = "all";
    downloadModelsSearchQuery = "";
    downloadModelsExpandedModelId = installedRow.modelId;
    if (installedRow.buildOf && !downloadModelsUnfoldedLeadIds.includes(installedRow.buildOf)) {
        downloadModelsUnfoldedLeadIds = [...downloadModelsUnfoldedLeadIds, installedRow.buildOf];
    }
    setTimeout(() => {
        const detailRow = typeof document.getElementById === "function"
            ? document.getElementById("download-models-row-" + installedRow.modelId)
            : null;
        if (detailRow && typeof detailRow.scrollIntoView === "function") {
            detailRow.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }, 0);
}

function openDownloadModelsOnHuggingFace(installedRow) {
    if (typeof window !== "undefined" && typeof window.open === "function") {
        window.open(DOWNLOAD_MODELS_HUGGING_FACE_MODEL_URL + installedRow.huggingfaceId, "_blank", "noopener");
    }
}

// A note under the list, once, when some copy cannot be removed here, so no
// row has to explain a missing action.
function createDownloadModelsInstalledFootnote(installedRows) {
    if (installedRows.every(canRemoveDownloadModelsModelFromDisk)) return null;
    return createDownloadModelsParagraph(
        "download-models-installed-footnote",
        "Some models can only be removed in Finder. A model in your Hugging Face folder is never moved to the Trash from here, so other apps that use it keep working."
    );
}

// Escape or a click outside closes an open ⋯ menu.
function wireDownloadModelsMenuDismissal() {
    if (typeof document.addEventListener !== "function") return;
    document.addEventListener("keydown", (keyboardEvent) => {
        if (keyboardEvent.key === "Escape" && downloadModelsOpenMenuHuggingfaceId !== null) {
            downloadModelsOpenMenuHuggingfaceId = null;
            renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
        }
    });
    document.addEventListener("click", (clickEvent) => {
        const clickedElement = clickEvent.target;
        const isInsideMenu = clickedElement && typeof clickedElement.closest === "function"
            && clickedElement.closest(".download-models-row-menu, .download-models-menu-button");
        if (!isInsideMenu && downloadModelsOpenMenuHuggingfaceId !== null) {
            downloadModelsOpenMenuHuggingfaceId = null;
            renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
        }
    });
}
