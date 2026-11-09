// Astronomical Observatory Download Models card footer: the capability tags on
// the left and the card's buttons on the right (Download or Open in Thin Talk,
// download controls, Details). Loaded after download-models-render.js, whose
// card builder calls createDownloadModelsCardFooter.

// What the model can do and what you can do with it share the card's last row:
// the tags on the left read as facts, the buttons on the right as actions, so
// the Download button no longer sits alone on a row of its own.
function createDownloadModelsCardFooter(catalogRow, isExpanded) {
    const footer = document.createElement("div");
    footer.className = "download-models-card-footer";
    footer.replaceChildren(
        createDownloadModelsCapabilityBadges(catalogRow),
        createDownloadModelsPrimaryActions(catalogRow, isExpanded)
    );
    return footer;
}

// Capabilities in the one order every card uses, so the same skill sits in the
// same place and keeps the same colour from card to card; a card shows only the
// ones its model has, and the context size always comes last.
const DOWNLOAD_MODELS_CAPABILITY_ORDER = [
    { kind: "reasoning", label: "Reasoning", isSupported: (catalogRow) => catalogRow.supportsReasoning },
    { kind: "vision", label: "Vision", isSupported: (catalogRow) => catalogRow.supportsVision },
    { kind: "tools", label: "Tools", isSupported: (catalogRow) => catalogRow.supportsToolCalls },
    { kind: "image-generation", label: "Image Generation", isSupported: (catalogRow) => catalogRow.supportsImageGeneration },
    { kind: "embeddings", label: "Embeddings", isSupported: (catalogRow) => catalogRow.supportsEmbeddings }
];

function createDownloadModelsCapabilityBadges(catalogRow) {
    const badgeContainer = document.createElement("div");
    badgeContainer.className = "download-models-capability-badges";
    const badges = DOWNLOAD_MODELS_CAPABILITY_ORDER
        .filter((capability) => capability.isSupported(catalogRow))
        .map((capability) => ({ kind: capability.kind, text: capability.label }));
    const contextLabel = formatDownloadModelsContextTokens(catalogRow.contextWindow);
    if (contextLabel) badges.push({ kind: "context", text: contextLabel });
    for (const badgeDescription of badges) {
        const badge = document.createElement("span");
        badge.className = "download-models-capability-badge";
        badge.dataset.capability = badgeDescription.kind;
        badge.textContent = badgeDescription.text;
        badgeContainer.appendChild(badge);
    }
    return badgeContainer;
}

function createDownloadModelsPrimaryActions(catalogRow, isExpanded) {
    const actions = document.createElement("div");
    actions.className = "download-models-primary-actions";
    if (canChatWithCatalogModel(catalogRow)) {
        actions.appendChild(createDownloadModelsOpenInChatButton());
    }
    for (const button of downloadModelsActionButtons(catalogRow)) {
        actions.appendChild(button);
    }
    const detailsButton = document.createElement("button");
    detailsButton.className = "button button-secondary download-models-details-toggle";
    detailsButton.type = "button";
    detailsButton.textContent = isExpanded ? "Hide details" : "Details";
    detailsButton.setAttribute("aria-expanded", String(isExpanded));
    detailsButton.addEventListener("click", () => {
        downloadModelsExpandedModelId = isExpanded
            ? null
            : catalogRow.huggingfaceId;
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    });
    actions.appendChild(detailsButton);
    return actions;
}

// Thin Talk can only hold a conversation with a chat model: an image model
// draws through a chat model's tool and an embedding model only turns text
// into numbers, so offering either for a chat would promise something that
// cannot happen.
function canChatWithCatalogModel(catalogRow) {
    return Boolean(catalogRow.readyOnThisMac)
        && !catalogRow.supportsImageGeneration
        && !catalogRow.supportsEmbeddings;
}

function createDownloadModelsOpenInChatButton() {
    const openInChatButton = document.createElement("button");
    openInChatButton.className = "button button-primary download-models-open-in-chat";
    openInChatButton.type = "button";
    openInChatButton.textContent = "Open in Thin Talk";
    openInChatButton.addEventListener("click", openThinTalkFromDownloadModels);
    return openInChatButton;
}

// Thin Talk picks the model inside its own model chooser, so opening the
// conversation window is all the handoff needs; there is no in-Observatory
// chat to preselect a model for anymore.
function openThinTalkFromDownloadModels() {
    const chatLink = thinTalkChatLink(document.body.dataset.applicationChannel);
    if (chatLink) {
        window.location.assign(chatLink);
    }
}
