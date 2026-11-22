// Astronomical Observatory Home: the one next step, the engine's first model
// suggestion for this Mac (the same one the Models page leads with). "Not now"
// hides that suggestion on this Mac until the engine suggests a different
// model; it is a per-viewer convenience, so browser storage that is missing
// or blocked simply means the card shows again.

const OVERVIEW_DISMISSED_NEXT_STEP_KEY = "observatory.dismissedNextStepModel";

function overviewNextStepSuggestion(catalogDocument) {
    const suggestions = Array.isArray(catalogDocument?.suggestions) ? catalogDocument.suggestions : [];
    const firstSuggestion = suggestions.find(
        (suggestion) => typeof suggestion?.huggingface_id === "string" && typeof suggestion.reason === "string"
    );
    if (!firstSuggestion) return null;
    const catalogEntry = (catalogDocument.entries || []).find(
        (entry) => entry?.huggingface_id === firstSuggestion.huggingface_id
    );
    if (!catalogEntry || !(catalogEntry.approximate_size_bytes > 0)) return null;
    return { ...firstSuggestion, approximate_size_bytes: catalogEntry.approximate_size_bytes };
}

function readOverviewDismissedNextStep() {
    try {
        return window.localStorage.getItem(OVERVIEW_DISMISSED_NEXT_STEP_KEY);
    } catch {
        return null;
    }
}

function rememberOverviewDismissedNextStep(huggingfaceId) {
    try {
        window.localStorage.setItem(OVERVIEW_DISMISSED_NEXT_STEP_KEY, huggingfaceId);
    } catch {
        // Without storage the card simply returns on the next visit.
    }
}

// Called on every status poll; it reads the catalog the Models page keeps.
function renderOverviewNextStep() {
    const nextStepCard = document.getElementById("overview-next-step");
    if (!nextStepCard) return;
    const nextStep = overviewNextStepSuggestion(downloadModelsCatalogDocument);
    if (nextStep === null || readOverviewDismissedNextStep() === nextStep.huggingface_id) {
        nextStepCard.hidden = true;
        return;
    }
    nextStepCard.hidden = false;
    setOverviewText("overview-next-step-tag", nextStep.tag || "Next step");
    setOverviewText("overview-next-step-model", nextStep.model_id || nextStep.huggingface_id);
    setOverviewText("overview-next-step-reason", nextStep.reason);
    const getButton = document.getElementById("overview-next-step-get");
    getButton.textContent = "Get · " + formatDownloadModelsCardSize(nextStep.approximate_size_bytes);
    getButton.onclick = () => performDownloadModelsAction("start", nextStep.huggingface_id);
    document.getElementById("overview-next-step-dismiss").onclick = () => {
        rememberOverviewDismissedNextStep(nextStep.huggingface_id);
        nextStepCard.hidden = true;
    };
}
