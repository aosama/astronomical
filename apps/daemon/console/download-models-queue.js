// Astronomical Observatory Models page: the download list. Models download one
// at a time, because one download already fills the connection; Get on any
// other model adds it to the list instead of being greyed out. This file reads
// the list from the download status, draws it under the current download, and
// sends Get, Try again and Remove for listed models.

const LIBRARY_QUEUE_REMOVE_URL = "/v1/library/download/queue/remove";

// The sentence shown when Get was refused (for example, not enough disk for
// the whole list), until the next Get succeeds.
let downloadModelsQueueNotice = null;

function downloadModelsWaitingModels() {
    return downloadModelsCurrentDownload.queue?.waiting || [];
}

function downloadModelsFailedQueueEntries() {
    return downloadModelsCurrentDownload.queue?.failed || [];
}

// 1 for the next model to download, or null when the model is not waiting.
function downloadModelsWaitingPosition(huggingfaceId) {
    const waitingIndex = downloadModelsWaitingModels()
        .findIndex((waitingModel) => waitingModel.huggingface_id === huggingfaceId);
    return waitingIndex < 0 ? null : waitingIndex + 1;
}

function downloadModelsFailedQueueEntry(huggingfaceId) {
    return downloadModelsFailedQueueEntries()
        .find((failedEntry) => failedEntry.huggingface_id === huggingfaceId) || null;
}

function hasDownloadModelsQueueEntries() {
    return downloadModelsWaitingModels().length > 0 || downloadModelsFailedQueueEntries().length > 0;
}

// "2nd", "3rd", "11th": a place in the line as people say it.
function formatDownloadModelsLinePlace(place) {
    const lastTwoDigits = place % 100;
    if (lastTwoDigits >= 11 && lastTwoDigits <= 13) return place + "th";
    const suffixByLastDigit = { 1: "st", 2: "nd", 3: "rd" };
    return place + (suffixByLastDigit[place % 10] || "th");
}

// The row's place in line: the current download is first, so the next model
// to download is second.
function downloadModelsWaitingStatusText(waitingPosition) {
    return "Waiting · " + formatDownloadModelsLinePlace(waitingPosition + 1) + " in line";
}

// "2 models waiting · 21.4 GB · about 25 minutes after this one": the time is
// left out until the transfer rate has been measured, rather than guessed.
function downloadModelsQueueSummaryText(waitingModels) {
    const waitingBytes = waitingModels
        .reduce((totalBytes, waitingModel) => totalBytes + (waitingModel.approximate_size_bytes || 0), 0);
    const countText = waitingModels.length === 1 ? "1 model waiting" : waitingModels.length + " models waiting";
    const summaryParts = [countText, formatDownloadModelsGigabyteCount(waitingBytes) + " GB"];
    if (downloadModelsMeasuredBytesPerSecond > 0) {
        summaryParts.push(formatDownloadModelsRemainingTime(waitingBytes / downloadModelsMeasuredBytesPerSecond)
            .replace(" left", " after this one"));
    }
    return summaryParts.join(" · ");
}

// The list under the current download: who is waiting and what could not
// finish. Null when nothing is listed.
function createDownloadModelsQueueList(catalogRows) {
    if (!hasDownloadModelsQueueEntries()) return null;
    const rowById = new Map(catalogRows.map((catalogRow) => [catalogRow.huggingfaceId, catalogRow]));
    const queueList = createDownloadModelsElement("div", "download-models-queue");
    const queueChildren = [];
    const waitingModels = downloadModelsWaitingModels();
    if (waitingModels.length > 0) {
        queueChildren.push(createDownloadModelsElement("p", "download-models-queue-summary",
            downloadModelsQueueSummaryText(waitingModels)));
        waitingModels.forEach((waitingModel, waitingIndex) => {
            queueChildren.push(createDownloadModelsWaitingLine(waitingModel, waitingIndex + 1, rowById));
        });
    }
    for (const failedEntry of downloadModelsFailedQueueEntries()) {
        queueChildren.push(createDownloadModelsFailedQueueLine(failedEntry, rowById));
    }
    queueList.replaceChildren(...queueChildren);
    return queueList;
}

function downloadModelsQueueModelName(huggingfaceId, rowById) {
    const catalogRow = rowById.get(huggingfaceId);
    return catalogRow ? downloadModelsRowName(catalogRow) : huggingfaceId;
}

function createDownloadModelsWaitingLine(waitingModel, waitingPosition, rowById) {
    const waitingLine = createDownloadModelsElement("div", "download-models-queue-line");
    const sizeText = waitingModel.approximate_size_bytes > 0
        ? " · " + formatDownloadModelsCardSize(waitingModel.approximate_size_bytes) : "";
    const lineText = createDownloadModelsElement("span", "download-models-queue-line-text",
        formatDownloadModelsLinePlace(waitingPosition + 1) + " · "
        + downloadModelsQueueModelName(waitingModel.huggingface_id, rowById) + sizeText);
    waitingLine.replaceChildren(lineText, createDownloadModelsQueueRemoveButton(waitingModel.huggingface_id));
    return waitingLine;
}

function createDownloadModelsFailedQueueLine(failedEntry, rowById) {
    const failedLine = createDownloadModelsElement("div", "download-models-queue-line download-models-queue-line-failed");
    const lineText = createDownloadModelsElement("span", "download-models-queue-line-text");
    lineText.replaceChildren(
        createDownloadModelsElement("span", "download-models-queue-line-title",
            "Couldn't finish " + downloadModelsQueueModelName(failedEntry.huggingface_id, rowById)),
        createDownloadModelsElement("span", "download-models-queue-line-detail",
            downloadModelsFailureExplanation(failedEntry.error_code))
    );
    const lineActions = createDownloadModelsElement("div", "download-models-installed-actions");
    lineActions.replaceChildren(
        downloadModelsActionButton("Try again", "start", failedEntry.huggingface_id),
        createDownloadModelsQueueRemoveButton(failedEntry.huggingface_id)
    );
    failedLine.replaceChildren(lineText, lineActions);
    return failedLine;
}

function createDownloadModelsQueueRemoveButton(huggingfaceId) {
    const removeButton = createDownloadModelsElement("button", "button button-quiet download-models-queue-remove", "Remove from list");
    removeButton.type = "button";
    removeButton.addEventListener("click", () => removeDownloadModelsFromQueue(huggingfaceId));
    return removeButton;
}

// Get: starts the model, or adds it to the list. The status poll is the truth
// for what downloads, so the answer is followed by a fresh reading instead of
// being taken as the current download. A refusal becomes a page notice and
// leaves the running download as it is.
async function startDownloadModelsDownload(huggingfaceId) {
    try {
        const startResponse = await fetch(LIBRARY_DOWNLOAD_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ huggingface_id: huggingfaceId })
        });
        const startDocument = await startResponse.json();
        downloadModelsQueueNotice = startResponse.ok
            ? null
            : startDocument?.error?.message || "That model could not be added to the downloads.";
    } catch {
        downloadModelsQueueNotice = "Astronomical did not answer. Try again in a moment.";
    }
    await refreshDownloadModelsDownloadState();
}

async function removeDownloadModelsFromQueue(huggingfaceId) {
    try {
        await fetch(LIBRARY_QUEUE_REMOVE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ huggingface_id: huggingfaceId })
        });
    } catch {
        // The next status poll shows the list as it really is.
    }
    await refreshDownloadModelsDownloadState();
}

function createDownloadModelsQueueNotice() {
    return downloadModelsQueueNotice
        ? createDownloadModelsNotice(downloadModelsQueueNotice, "download-models-notice download-models-notice-page")
        : null;
}
