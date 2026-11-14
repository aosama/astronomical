// Astronomical Observatory Models page: the On this Mac section. The download
// in flight sits on top with the only controls it needs, then one line per
// model already here with Open and Remove.

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

function createDownloadModelsInstalledSection(installedRows, activeDownloadRow) {
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
    const lines = installedRows.map(createDownloadModelsInstalledLine);
    if (activeDownloadRow) lines.unshift(createDownloadModelsDownloadBanner(activeDownloadRow));
    lineList.replaceChildren(...lines);
    section.replaceChildren(heading, lineList);
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

function createDownloadModelsInstalledLine(installedRow) {
    const line = createDownloadModelsElement("div", "download-models-installed-line");
    const identity = createDownloadModelsElement("div", "download-models-installed-identity");
    const identityChildren = [
        createDownloadModelsElement("span", "download-models-installed-name", downloadModelsRowName(installedRow)),
        createDownloadModelsElement("span", "download-models-installed-summary", downloadModelsInstalledSummaryText(installedRow))
    ];
    // Why a removal was refused, on the line of the model it is about.
    const refusalText = downloadModelsRemovalNoticeFor(installedRow);
    if (refusalText) identityChildren.push(createDownloadModelsNotice(refusalText));
    identity.replaceChildren(...identityChildren);
    const actions = createDownloadModelsElement("div", "download-models-installed-actions");
    const actionControls = [];
    if (canChatWithCatalogModel(installedRow)) actionControls.push(createDownloadModelsOpenInChatButton());
    if (canRemoveDownloadModelsModelFromDisk(installedRow)) {
        actionControls.push(...createDownloadModelsRemovalControls(installedRow));
    } else {
        // A copy the person placed themself is theirs to manage: the page
        // shows where it is instead of offering to remove it.
        actionControls.push(createDownloadModelsRevealButton(installedRow.huggingfaceId));
    }
    actions.replaceChildren(...actionControls);
    line.replaceChildren(identity, actions);
    return line;
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

// Thin Talk can only hold a conversation with a chat model: an image model
// draws through a chat model's tool, an embedding model only turns text into
// numbers, a decision model only picks among set answers and a speech model
// only reads text aloud, so offering any of them for a chat would promise
// something that cannot happen.
function canChatWithCatalogModel(catalogRow) {
    return Boolean(catalogRow.readyOnThisMac)
        && !catalogRow.supportsImageGeneration
        && !catalogRow.supportsEmbeddings
        && !catalogRow.supportsDecisions
        && !catalogRow.supportsSpeech;
}

function createDownloadModelsOpenInChatButton() {
    const openInChatButton = createDownloadModelsElement(
        "button", "button button-secondary download-models-open-in-chat", "Chat with this model"
    );
    openInChatButton.type = "button";
    openInChatButton.addEventListener("click", openThinTalkFromDownloadModels);
    return openInChatButton;
}

// Thin Talk picks the model inside its own model chooser, so opening the
// conversation window is all the handoff needs.
function openThinTalkFromDownloadModels() {
    const chatLink = thinTalkChatLink(document.body.dataset.applicationChannel);
    if (chatLink) {
        window.location.assign(chatLink);
    }
}
