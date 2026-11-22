// Astronomical Observatory Models page: the On this Mac section. The download
// in flight sits on top with the only controls it needs, then the models
// already here as rows under column names (download-models-installed-rows.js).

const DOWNLOAD_MODELS_ACTIVE_STATES = [
    "checking_disk", "fetching_manifest", "downloading", "paused", "verifying", "publishing", "failed"
];

// The row whose download holds the page's attention, or null. The status poll
// is the live truth; a catalog row's own state is only a fallback from before
// the first poll (see downloadModelsRowDownloadState).
function downloadModelsActiveDownloadRow(catalogRows) {
    return catalogRows.find((catalogRow) => !catalogRow.readyOnThisMac
        && DOWNLOAD_MODELS_ACTIVE_STATES.includes(downloadModelsRowDownloadState(catalogRow))) || null;
}

function createDownloadModelsInstalledSection(installedRows, activeDownloadRow, catalogRows = []) {
    const section = createDownloadModelsElement("section", "download-models-installed");
    const heading = createDownloadModelsElement("div", "download-models-section-heading");
    const headingChildren = [createDownloadModelsElement("h3", "", "On this Mac")];
    if (installedRows.length > 0) {
        headingChildren.push(createDownloadModelsElement(
            "span", "download-models-section-meta", downloadModelsInstalledTotalText(installedRows)
        ));
    }
    heading.replaceChildren(...headingChildren);
    const lineList = createDownloadModelsElement("div", "download-models-list");
    const lines = downloadModelsInstalledRowsInOrder(installedRows).map(createDownloadModelsInstalledLine);
    if (installedRows.length > 0) lines.unshift(createDownloadModelsInstalledHeader());
    const queueList = createDownloadModelsQueueList(catalogRows);
    if (queueList) lines.unshift(queueList);
    if (activeDownloadRow) lines.unshift(createDownloadModelsDownloadBanner(activeDownloadRow));
    lineList.replaceChildren(...lines);
    const sectionChildren = [heading, lineList];
    const footnote = createDownloadModelsInstalledFootnote(installedRows);
    if (footnote) sectionChildren.push(footnote);
    section.replaceChildren(...sectionChildren);
    return section;
}

// The total is the space the listed models take, which is what removing them frees.
function downloadModelsInstalledTotalText(installedRows) {
    const installedBytes = installedRows.reduce(
        (totalBytes, installedRow) => totalBytes + installedRow.approximateSizeBytes, 0
    );
    const modelCountText = installedRows.length === 1 ? "1 model" : installedRows.length + " models";
    return modelCountText + " · " + formatDownloadModelsCardSize(installedBytes);
}

// One sentence says what is happening and the buttons are the only things the
// person can do next. A stopped download is drawn as a problem; the rest as work.
function createDownloadModelsDownloadBanner(activeDownloadRow) {
    const downloadState = downloadModelsRowDownloadState(activeDownloadRow);
    const banner = createDownloadModelsElement(
        "div", "download-models-banner download-models-banner-" + downloadState.replaceAll("_", "-")
    );
    const bannerLine = createDownloadModelsElement("div", "download-models-banner-line");
    const bannerText = createDownloadModelsElement("div", "download-models-banner-text");
    const bannerCopy = downloadModelsBannerCopy(activeDownloadRow, downloadState);
    bannerText.replaceChildren(
        createDownloadModelsElement("span", "download-models-banner-title", bannerCopy.title),
        createDownloadModelsElement("span", "download-models-banner-detail", bannerCopy.detail)
    );
    const bannerActions = createDownloadModelsElement("div", "download-models-installed-actions");
    bannerActions.replaceChildren(...downloadModelsActionButtons(activeDownloadRow));
    bannerLine.replaceChildren(bannerText, bannerActions);
    const bannerChildren = [bannerLine];
    const progressBar = createDownloadModelsProgressBar(activeDownloadRow, downloadState);
    if (progressBar) bannerChildren.push(progressBar);
    if (downloadState === "downloading") {
        bannerChildren.push(createDownloadModelsParagraph(
            "download-models-banner-reassurance", "You can close this window. The download keeps going."
        ));
    }
    banner.replaceChildren(...bannerChildren);
    return banner;
}

function downloadModelsBannerCopy(activeDownloadRow, downloadState) {
    const modelName = downloadModelsRowName(activeDownloadRow);
    const isCurrentDownload = downloadModelsCurrentDownload.huggingface_id === activeDownloadRow.huggingfaceId;
    const transferText = isCurrentDownload && downloadModelsCurrentDownload.bytes_total > 0
        ? downloadModelsTransferProgressText(downloadModelsCurrentDownload)
        : formatDownloadModelsCardSize(activeDownloadRow.approximateSizeBytes);
    if (downloadState === "paused") {
        return { title: modelName + " is paused", detail: transferText + " is saved. Nothing is lost." };
    }
    if (downloadState === "failed") {
        const errorCode = isCurrentDownload ? downloadModelsCurrentDownload.error_code : undefined;
        return { title: "The " + modelName + " download stopped", detail: downloadModelsFailureExplanation(errorCode) };
    }
    if (downloadState === "verifying" || downloadState === "publishing") {
        return {
            title: "Almost ready: checking " + modelName,
            detail: "The download finished. Making sure every file arrived intact."
        };
    }
    if (downloadState === "downloading") {
        return { title: "Downloading " + modelName, detail: transferText };
    }
    return { title: "Getting " + modelName + " ready to download", detail: "Checking free space and the list of files." };
}

function createDownloadModelsProgressBar(activeDownloadRow, downloadState) {
    const isCurrentDownload = downloadModelsCurrentDownload.huggingface_id === activeDownloadRow.huggingfaceId;
    if (!isCurrentDownload || !(downloadModelsCurrentDownload.bytes_total > 0)
        || !["downloading", "paused", "verifying", "publishing"].includes(downloadState)) {
        return null;
    }
    const percent = downloadState === "publishing" || downloadState === "verifying"
        ? 100
        : downloadModelsProgressPercent(downloadModelsCurrentDownload);
    const track = createDownloadModelsElement("div", "download-models-download-progress");
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(percent));
    track.setAttribute("aria-label", downloadModelsRowName(activeDownloadRow) + " download");
    const fill = createDownloadModelsElement("div", "download-models-download-progress-fill");
    fill.style.width = percent + "%";
    track.appendChild(fill);
    return track;
}

// On this Mac lists every copy the engine serves, the same models any client
// of the engine sees: one the Models page downloaded (ready) and one copied in
// by hand or kept at another revision (served). Get still offers to verify and
// adopt the second kind, so only readiness decides the download controls.
function isDownloadModelsRowOnThisMac(catalogRow) {
    return Boolean(catalogRow.readyOnThisMac) || Boolean(catalogRow.servedOnThisMac);
}

// Thin Talk can only hold a conversation with a chat model: an image model
// draws through a chat model's tool, an embedding model only turns text into
// numbers, a decision model only picks among set answers and a speech model
// only reads text aloud, so offering any of them for a chat would promise
// something that cannot happen.
function canChatWithCatalogModel(catalogRow) {
    return isDownloadModelsRowOnThisMac(catalogRow)
        && !catalogRow.supportsImageGeneration
        && !catalogRow.supportsEmbeddings
        && !catalogRow.supportsDecisions
        && !catalogRow.supportsSpeech;
}

function createDownloadModelsOpenInChatButton(installedRow) {
    const openInChatButton = createDownloadModelsElement(
        "button", "button button-secondary download-models-open-in-chat", "Chat"
    );
    openInChatButton.type = "button";
    // The row already names the model, so the button says only what it does;
    // VoiceOver hears which model it opens.
    openInChatButton.setAttribute("aria-label", "Chat with " + downloadModelsRowName(installedRow));
    openInChatButton.title = "Opens the chat window with this model";
    openInChatButton.addEventListener("click", () => openThinTalkFromDownloadModels(downloadModelsRowName(installedRow)));
    return openInChatButton;
}

// The chat link names the model in its query (?model=<model ID>), so the chat
// window opens on it rather than on whichever model it had. The model ID is
// the name the chat app lists the model under.
function downloadModelsChatLinkForModel(chatLink, modelId) {
    return chatLink + "?model=" + encodeURIComponent(modelId);
}

function openThinTalkFromDownloadModels(modelId) {
    const chatLink = thinTalkChatLink(document.body.dataset.applicationChannel);
    if (chatLink) {
        window.location.assign(downloadModelsChatLinkForModel(chatLink, modelId));
    }
}
