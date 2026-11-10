// Astronomical Observatory appearance: Auto follows the Mac's light or dark
// setting, and Light or Dark hold one palette. The choice is a per-browser
// convenience, so it lives in localStorage rather than in config.json.
//
// Loaded in <head> without defer so the chosen palette is applied before the
// first paint; a deferred script would flash the other palette on every load.

const OBSERVATORY_APPEARANCE_STORAGE_KEY = "astronomical-observatory-appearance";
const OBSERVATORY_APPEARANCES = ["auto", "light", "dark"];

// Storage can be unavailable (private windows, blocked site data); the page
// then simply follows the Mac.
function readObservatoryAppearance(appearanceStorage) {
    try {
        const storedAppearance = appearanceStorage.getItem(OBSERVATORY_APPEARANCE_STORAGE_KEY);
        return OBSERVATORY_APPEARANCES.includes(storedAppearance) ? storedAppearance : "auto";
    } catch {
        return "auto";
    }
}

function storeObservatoryAppearance(appearanceStorage, appearance) {
    try {
        appearanceStorage.setItem(OBSERVATORY_APPEARANCE_STORAGE_KEY, appearance);
    } catch {
        // The choice still applies to this page view.
    }
}

// Auto removes the attribute, which hands the choice back to the stylesheet's
// prefers-color-scheme rule.
function applyObservatoryAppearance(rootElement, appearance) {
    if (appearance === "auto") {
        rootElement.removeAttribute("data-theme");
    } else {
        rootElement.setAttribute("data-theme", appearance);
    }
}

function markObservatoryAppearanceOptions(appearanceOptions, appearance) {
    for (const appearanceOption of appearanceOptions) {
        appearanceOption.setAttribute(
            "aria-pressed", String(appearanceOption.dataset.appearance === appearance)
        );
    }
}

function wireObservatoryAppearanceSwitch(appearanceDocument, appearanceStorage) {
    const appearanceSwitch = appearanceDocument.getElementById("appearance-switch");
    if (!appearanceSwitch) return;
    const appearanceOptions = [...appearanceSwitch.querySelectorAll("[data-appearance]")];
    markObservatoryAppearanceOptions(appearanceOptions, readObservatoryAppearance(appearanceStorage));
    for (const appearanceOption of appearanceOptions) {
        appearanceOption.addEventListener("click", () => {
            const chosenAppearance = appearanceOption.dataset.appearance;
            storeObservatoryAppearance(appearanceStorage, chosenAppearance);
            applyObservatoryAppearance(appearanceDocument.documentElement, chosenAppearance);
            markObservatoryAppearanceOptions(appearanceOptions, chosenAppearance);
        });
    }
}

if (typeof document !== "undefined" && typeof localStorage !== "undefined") {
    applyObservatoryAppearance(document.documentElement, readObservatoryAppearance(localStorage));
    document.addEventListener("DOMContentLoaded", () => wireObservatoryAppearanceSwitch(document, localStorage));
}
