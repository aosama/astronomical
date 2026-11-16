// Astronomical Observatory Models page: keeping text selectable while a
// download runs. The page rebuilds its rows whenever what it shows changes,
// and a running download changes every second, so without this a person who
// drags across a model's name or a folder path loses the selection a moment
// later. While text is selected, or the mouse button is held down to select
// it, the rebuild waits, and only the download's own progress moves.

let isDownloadModelsPointerSelecting = false;
let isDownloadModelsSelectionGuardWired = false;

// Pointer state is tracked on the whole document: a drag that starts on the
// page can end outside it, and its mouseup must still end the hold. Wired
// once, since the page itself can be wired again.
function wireDownloadModelsSelectionGuard(catalogContainer) {
    if (isDownloadModelsSelectionGuardWired || typeof catalogContainer?.addEventListener !== "function"
        || typeof document.addEventListener !== "function") {
        return;
    }
    isDownloadModelsSelectionGuardWired = true;
    catalogContainer.addEventListener("pointerdown", (pointerEvent) => {
        if (pointerEvent.button === 0) isDownloadModelsPointerSelecting = true;
    });
    document.addEventListener("pointerup", releaseDownloadModelsPointerSelection);
    document.addEventListener("pointercancel", releaseDownloadModelsPointerSelection);
}

function releaseDownloadModelsPointerSelection() {
    if (!isDownloadModelsPointerSelecting) return;
    isDownloadModelsPointerSelecting = false;
    // A click that selected nothing should not leave the page a second
    // behind; a finished drag that did select something keeps the hold.
    if (typeof downloadModelsCatalogDocument !== "undefined" && downloadModelsCatalogDocument) {
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    }
}

// True while the person is selecting, or has selected, text on the page.
function isDownloadModelsSelectionHeld(catalogContainer) {
    if (isDownloadModelsPointerSelecting) return true;
    const pageSelection = typeof window !== "undefined" && window.getSelection ? window.getSelection() : null;
    if (!pageSelection || pageSelection.isCollapsed || pageSelection.rangeCount === 0) return false;
    const selectedRange = pageSelection.getRangeAt(0);
    return catalogContainer.contains(selectedRange.startContainer)
        || catalogContainer.contains(selectedRange.endContainer);
}

// While a rebuild waits, the download banner's bar and progress text still
// follow the download, unless the selection is inside that text.
function updateDownloadModelsBannerInPlace(catalogContainer) {
    const banner = catalogContainer.querySelector(".download-models-banner");
    if (!banner || downloadModelsCurrentDownload.state !== "downloading") return;
    const progressFill = banner.querySelector(".download-models-download-progress-fill");
    if (progressFill) {
        progressFill.style.width = downloadModelsProgressPercent(downloadModelsCurrentDownload) + "%";
    }
    const bannerDetail = banner.querySelector(".download-models-banner-detail");
    const pageSelection = typeof window !== "undefined" && window.getSelection ? window.getSelection() : null;
    const isDetailSelected = pageSelection && !pageSelection.isCollapsed && bannerDetail
        && pageSelection.containsNode(bannerDetail, true);
    if (bannerDetail && !isDetailSelected) {
        bannerDetail.textContent = downloadModelsTransferProgressText(downloadModelsCurrentDownload);
    }
}
