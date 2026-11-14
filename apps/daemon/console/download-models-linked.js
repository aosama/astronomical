// Astronomical Observatory Models page: a model opened from a link.
//
// A web page or a model card can link to one model
// (`astronomical://models/huggingface/OWNER/NAME`); the app turns that into
// this page with `?model=OWNER/NAME`. The page then puts that one model at
// the top with its own Get button, so the person does not have to find it in
// the list. Nothing downloads until Get is pressed, and only a model in the
// catalog is ever offered, whatever the link names.

const DOWNLOAD_MODELS_LINKED_QUERY_PARAMETER = "model";
// An owner and a name in Hugging Face's own characters; anything else is not
// a model name and is ignored rather than shown back to the person.
const DOWNLOAD_MODELS_LINKED_ID_PATTERN = /^[A-Za-z0-9._-]{1,96}\/[A-Za-z0-9._-]{1,96}$/;

// The Hugging Face ID the page was opened on, or null.
let downloadModelsLinkedModelId = null;

function readDownloadModelsLinkedModelId(locationSearch) {
    if (typeof locationSearch !== "string" || locationSearch.length === 0) {
        return null;
    }
    const linkedModelId = new URLSearchParams(locationSearch).get(DOWNLOAD_MODELS_LINKED_QUERY_PARAMETER);
    if (linkedModelId === null || !DOWNLOAD_MODELS_LINKED_ID_PATTERN.test(linkedModelId)) {
        return null;
    }
    return linkedModelId;
}

// Hugging Face treats owner and name case-insensitively when it resolves a
// link, so a link typed in another case still finds its model.
function findDownloadModelsLinkedRow(catalogRows, linkedModelId) {
    const wantedModelId = linkedModelId.toLowerCase();
    for (const catalogRow of catalogRows) {
        if (typeof catalogRow.huggingfaceId === "string"
            && catalogRow.huggingfaceId.toLowerCase() === wantedModelId) {
            return catalogRow;
        }
    }
    return null;
}

function createDownloadModelsLinkedSection(catalogRows) {
    if (downloadModelsLinkedModelId === null) {
        return null;
    }
    const linkedRow = findDownloadModelsLinkedRow(catalogRows, downloadModelsLinkedModelId);
    const section = createDownloadModelsElement("section", "download-models-linked");
    section.setAttribute("aria-label", "The model you opened");
    if (linkedRow === null) {
        section.replaceChildren(
            createDownloadModelsElement("h3", "download-models-linked-title", "That model is not in Astronomical's list"),
            createDownloadModelsParagraph(
                "download-models-linked-lede",
                downloadModelsLinkedModelId + " is not one of the models Astronomical has checked. "
                    + "The models below are."
            )
        );
        return section;
    }
    section.replaceChildren(
        createDownloadModelsElement("h3", "download-models-linked-title", "You opened " + downloadModelsRowName(linkedRow)),
        createDownloadModelsParagraph("download-models-linked-lede", downloadModelsLinkedLede(linkedRow)),
        createDownloadModelsLinkedControls(linkedRow)
    );
    return section;
}

function downloadModelsLinkedLede(linkedRow) {
    const summaryText = linkedRow.summary || linkedRow.description || "";
    if (linkedRow.readyOnThisMac) {
        return (summaryText + " It is already on this Mac.").trim();
    }
    return (summaryText + " It downloads once, then runs entirely on this Mac.").trim();
}

// The same meter and action the list shows for this model, so the link adds
// a shortcut and never a second way to download.
function createDownloadModelsLinkedControls(linkedRow) {
    const controls = createDownloadModelsElement("div", "download-models-linked-controls");
    const controlChildren = [];
    const fitMeter = createDownloadModelsFitMeter(linkedRow);
    if (fitMeter) controlChildren.push(fitMeter);
    controlChildren.push(createDownloadModelsRowAction(linkedRow));
    controls.replaceChildren(...controlChildren);
    return controls;
}
