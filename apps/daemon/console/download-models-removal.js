// Astronomical Observatory Download Models "Remove from disk": the button on a
// downloaded model's card, the in-card confirmation that names what will move,
// and the request. The daemon moves only a folder Download Models itself
// installed, and moves it to the Trash, so a mistaken removal can be undone in
// Finder.
//
// The confirmation is drawn in the card rather than with window.confirm():
// embedded web views (the Mac app's, an in-app browser) suppress that dialog
// and answer Cancel, so the button silently did nothing there.

const DOWNLOAD_MODELS_REMOVE_URL = "/v1/library/remove";

// The outcome, if any, shown on the card of the model it is about; cleared by
// the next attempt.
let downloadModelsRemovalNotice = null;
// The model whose card is asking "Move it to the Trash?", or null.
let downloadModelsRemovalConfirmingId = null;
// The model whose removal request is on its way, or null. A second click on a
// card that is already moving its folder must not send a second request.
let downloadModelsRemovalInFlightId = null;
// Which control to focus once the card is drawn again ("keep" or "remove"), so
// keyboard and VoiceOver users land on the question and come back from it.
let downloadModelsRemovalFocusRequest = null;

// The state the catalog's render fingerprint must include: none of it changes
// a catalog field, yet each changes what a card shows.
function downloadModelsRemovalRenderState() {
    return [downloadModelsRemovalConfirmingId, downloadModelsRemovalInFlightId, downloadModelsRemovalNotice];
}

function canRemoveDownloadModelsModelFromDisk(catalogRow) {
    return Boolean(catalogRow.readyOnThisMac) && Boolean(catalogRow.removableFromDisk);
}

// Names the model, how much will move, where it goes, and how to get it back,
// because the click frees gigabytes and cannot be undone from this page.
function downloadModelsRemovalConfirmationText(catalogRow) {
    return "Move " + catalogRow.displayName + " ("
        + formatDownloadModelsCardSize(catalogRow.approximateSizeBytes)
        + ") to the Trash? You can download it again at any time.";
}

// The card's removal controls: the Remove from disk button; once clicked, the
// question with Move to Trash and Keep; while the request runs, its progress.
function createDownloadModelsRemovalControls(catalogRow) {
    if (downloadModelsRemovalInFlightId === catalogRow.huggingfaceId) {
        const movingButton = createDownloadModelsRemovalButton(
            "Moving to Trash…", "button button-danger download-models-confirm-removal-button", () => {}
        );
        movingButton.disabled = true;
        return [movingButton];
    }
    if (downloadModelsRemovalConfirmingId !== catalogRow.huggingfaceId) {
        const removeButton = createDownloadModelsRemovalButton(
            "Remove from disk", "button button-danger download-models-remove-button",
            () => askToRemoveDownloadModelsModel(catalogRow)
        );
        focusDownloadModelsRemovalControlIfRequested(catalogRow, "remove", removeButton);
        return [removeButton];
    }
    return createDownloadModelsRemovalQuestion(catalogRow);
}

function createDownloadModelsRemovalQuestion(catalogRow) {
    const question = document.createElement("span");
    question.className = "download-models-removal-question";
    question.id = "download-models-removal-question";
    question.textContent = downloadModelsRemovalConfirmationText(catalogRow);
    const confirmButton = createDownloadModelsRemovalButton(
        "Move to Trash", "button button-danger download-models-confirm-removal-button",
        () => confirmDownloadModelsRemoval(catalogRow)
    );
    const keepButton = createDownloadModelsRemovalButton(
        "Keep", "button button-secondary download-models-keep-button", cancelDownloadModelsRemoval
    );
    for (const answerButton of [confirmButton, keepButton]) {
        answerButton.setAttribute("aria-describedby", question.id);
    }
    // Keep is focused rather than Move to Trash, so an accidental Return
    // after the question opens keeps the model.
    focusDownloadModelsRemovalControlIfRequested(catalogRow, "keep", keepButton);
    return [question, confirmButton, keepButton];
}

// Focus waits until the render has put the control in the page; a detached
// element cannot take focus.
function focusDownloadModelsRemovalControlIfRequested(catalogRow, controlName, control) {
    const focusRequest = downloadModelsRemovalFocusRequest;
    if (!focusRequest || focusRequest.huggingfaceId !== catalogRow.huggingfaceId
        || focusRequest.controlName !== controlName) return;
    downloadModelsRemovalFocusRequest = null;
    Promise.resolve().then(() => control.focus());
}

function createDownloadModelsRemovalButton(label, className, onClick) {
    const removalButton = document.createElement("button");
    removalButton.className = className;
    removalButton.type = "button";
    removalButton.textContent = label;
    removalButton.addEventListener("click", onClick);
    return removalButton;
}

function askToRemoveDownloadModelsModel(catalogRow) {
    downloadModelsRemovalNotice = null;
    downloadModelsRemovalConfirmingId = catalogRow.huggingfaceId;
    downloadModelsRemovalFocusRequest = { huggingfaceId: catalogRow.huggingfaceId, controlName: "keep" };
    renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
}

function cancelDownloadModelsRemoval() {
    const keptModelId = downloadModelsRemovalConfirmingId;
    downloadModelsRemovalConfirmingId = null;
    downloadModelsRemovalFocusRequest = { huggingfaceId: keptModelId, controlName: "remove" };
    renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
}

async function confirmDownloadModelsRemoval(catalogRow) {
    if (downloadModelsRemovalInFlightId !== null) return;
    downloadModelsRemovalConfirmingId = null;
    downloadModelsRemovalInFlightId = catalogRow.huggingfaceId;
    renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    try {
        downloadModelsRemovalNotice = await removeDownloadModelsModelFromDisk(catalogRow);
        await refetchDownloadModelsCatalogDocument();
    } finally {
        downloadModelsRemovalInFlightId = null;
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    }
}

// Returns the notice to show on the model's card: what happened, or why not.
async function removeDownloadModelsModelFromDisk(catalogRow) {
    const huggingfaceId = catalogRow.huggingfaceId;
    try {
        const removalResponse = await fetch(DOWNLOAD_MODELS_REMOVE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ huggingface_id: huggingfaceId })
        });
        if (removalResponse.ok) {
            return { huggingfaceId, message: catalogRow.displayName + " was moved to the Trash." };
        }
        // A proxy or a crash can answer with something other than JSON; the
        // card still says the removal did not happen.
        const removalDocument = await removalResponse.json().catch(() => null);
        return {
            huggingfaceId,
            message: removalDocument?.error?.message || "The model could not be removed."
        };
    } catch {
        return { huggingfaceId, message: "The model could not be removed: the local server did not answer." };
    }
}

function downloadModelsRemovalNoticeFor(catalogRow) {
    return downloadModelsRemovalNotice && downloadModelsRemovalNotice.huggingfaceId === catalogRow.huggingfaceId
        ? downloadModelsRemovalNotice.message
        : null;
}
