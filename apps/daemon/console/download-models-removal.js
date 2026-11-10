// Astronomical Observatory Models page "Remove": the button on a downloaded
// model's line under On this Mac, the in-line confirmation that names what
// will move, and the request. The daemon moves only a folder this page itself
// installed, and moves it to the Trash, so a mistaken removal can be undone in
// Finder.
//
// The confirmation is drawn in the line rather than with window.confirm():
// embedded web views (the Mac app's, an in-app browser) suppress that dialog
// and answer Cancel, so the button silently did nothing there.

const DOWNLOAD_MODELS_REMOVE_URL = "/v1/library/remove";

// The outcome, if any, shown on the line of the model it is about:
// { huggingfaceId, message, movedToTrash }. Cleared by the next attempt.
let downloadModelsRemovalNotice = null;
// The model whose line is asking "Move it to the Trash?", or null.
let downloadModelsRemovalConfirmingId = null;
// The model whose removal request is on its way, or null. A second click on a
// line that is already moving its folder must not send a second request.
let downloadModelsRemovalInFlightId = null;
// Which control to focus once the line is drawn again ("keep", "remove" or
// "download"), so keyboard and VoiceOver users land on the question, come back
// from it, and are not dropped to the page top when the controls disappear.
let downloadModelsRemovalFocusRequest = null;

// The state the catalog's render fingerprint must include: none of it changes
// a catalog field, yet each changes what a line shows.
function downloadModelsRemovalRenderState() {
    return [downloadModelsRemovalConfirmingId, downloadModelsRemovalInFlightId, downloadModelsRemovalNotice];
}

function canRemoveDownloadModelsModelFromDisk(catalogRow) {
    return Boolean(catalogRow.readyOnThisMac) && Boolean(catalogRow.removableFromDisk);
}

// Names the model, how much will move, where it goes, and how to get it back,
// because the click frees gigabytes and cannot be undone from this page.
function downloadModelsRemovalConfirmationText(catalogRow) {
    return "Move " + (catalogRow.shortName || catalogRow.displayName) + " to the Trash? This frees "
        + formatDownloadModelsCardSize(catalogRow.approximateSizeBytes)
        + ". You can download it again any time.";
}

// The line's removal controls: the Remove button; once clicked, the
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
            "Remove", "button button-quiet download-models-remove-button",
            () => askToRemoveDownloadModelsModel(catalogRow)
        );
        // One removal at a time: another line's button waits, rather than
        // opening a question whose Move to Trash could not act yet.
        removeButton.disabled = downloadModelsRemovalInFlightId !== null;
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
        // After a move the line offers Get again; after a refusal it
        // offers Remove again. Focus follows to whichever remains.
        downloadModelsRemovalFocusRequest = {
            huggingfaceId: catalogRow.huggingfaceId,
            controlName: downloadModelsRemovalNotice.movedToTrash ? "download" : "remove"
        };
        await refetchDownloadModelsCatalogDocument();
    } catch {
        // The catalog stays as last fetched; the next poll brings it up to date.
    } finally {
        downloadModelsRemovalInFlightId = null;
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    }
}

// Returns the notice to show on the model's line: what happened, or why not.
async function removeDownloadModelsModelFromDisk(catalogRow) {
    const huggingfaceId = catalogRow.huggingfaceId;
    try {
        const removalResponse = await fetch(DOWNLOAD_MODELS_REMOVE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ huggingface_id: huggingfaceId })
        });
        if (removalResponse.ok) {
            return {
                huggingfaceId,
                message: (catalogRow.shortName || catalogRow.displayName) + " was moved to the Trash.",
                movedToTrash: true
            };
        }
        // A proxy or a crash can answer with something other than JSON; the
        // line still says the removal did not happen.
        const removalDocument = await removalResponse.json().catch(() => null);
        return {
            huggingfaceId,
            message: removalDocument?.error?.message || "The model could not be removed.",
            movedToTrash: false
        };
    } catch {
        return {
            huggingfaceId,
            message: "The model could not be removed: the local server did not answer.",
            movedToTrash: false
        };
    }
}

// The notice for this line, or null. It gives way to any download of the model
// (paused, verifying, publishing included), and "moved to the Trash" gives way
// once the model is on this Mac again, so an old outcome never comes back.
function downloadModelsRemovalNoticeFor(catalogRow) {
    const removalNotice = downloadModelsRemovalNotice;
    if (!removalNotice || removalNotice.huggingfaceId !== catalogRow.huggingfaceId) return null;
    if (downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId) return null;
    if (removalNotice.movedToTrash && catalogRow.readyOnThisMac) return null;
    return removalNotice.message;
}
