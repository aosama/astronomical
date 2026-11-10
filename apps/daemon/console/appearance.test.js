// Observatory appearance tests: the stored choice, applying it to the page, and
// the switch that changes it.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const appearanceScriptPath = path.join(__dirname, "appearance.js");
const appearanceScript = fs.readFileSync(appearanceScriptPath, "utf8");

// The script is run without a document, so its load-time wiring stays off and
// each test calls the functions it is about.
function createAppearanceContext() {
    const scriptContext = vm.createContext({});
    vm.runInContext(appearanceScript, scriptContext, { filename: appearanceScriptPath });
    return scriptContext;
}

function createAppearanceStorage(storedAppearance) {
    const storedItems = new Map(storedAppearance === undefined ? [] : [["astronomical-observatory-appearance", storedAppearance]]);
    return {
        storedItems,
        getItem: (itemKey) => (storedItems.has(itemKey) ? storedItems.get(itemKey) : null),
        setItem: (itemKey, itemText) => storedItems.set(itemKey, itemText)
    };
}

function createRootElement() {
    const rootAttributes = new Map();
    return {
        rootAttributes,
        setAttribute: (attributeName, attributeText) => rootAttributes.set(attributeName, attributeText),
        removeAttribute: (attributeName) => rootAttributes.delete(attributeName)
    };
}

function createAppearanceOption(appearance) {
    const optionAttributes = {};
    const clickListeners = [];
    return {
        dataset: { appearance },
        optionAttributes,
        setAttribute: (attributeName, attributeText) => { optionAttributes[attributeName] = attributeText; },
        addEventListener: (eventType, listener) => clickListeners.push(listener),
        click: () => clickListeners.forEach((listener) => listener())
    };
}

test("should_read_a_stored_light_or_dark_choice", () => {
    const scriptContext = createAppearanceContext();
    for (const storedAppearance of ["light", "dark", "auto"]) {
        scriptContext.appearanceStorage = createAppearanceStorage(storedAppearance);
        assert.equal(vm.runInContext("readObservatoryAppearance(appearanceStorage)", scriptContext), storedAppearance);
    }
});

test("should_follow_the_mac_when_nothing_or_something_unknown_is_stored", () => {
    const scriptContext = createAppearanceContext();
    for (const storedAppearance of [undefined, "", "sepia"]) {
        scriptContext.appearanceStorage = createAppearanceStorage(storedAppearance);
        assert.equal(vm.runInContext("readObservatoryAppearance(appearanceStorage)", scriptContext), "auto");
    }
});

test("should_follow_the_mac_when_storage_is_blocked", () => {
    const scriptContext = createAppearanceContext();
    scriptContext.appearanceStorage = {
        getItem() { throw new Error("storage is blocked"); },
        setItem() { throw new Error("storage is blocked"); }
    };

    assert.equal(vm.runInContext("readObservatoryAppearance(appearanceStorage)", scriptContext), "auto");
    assert.doesNotThrow(() => vm.runInContext('storeObservatoryAppearance(appearanceStorage, "dark")', scriptContext));
});

test("should_pin_a_chosen_palette_on_the_root_and_hand_auto_back_to_the_stylesheet", () => {
    const scriptContext = createAppearanceContext();
    const rootElement = createRootElement();
    scriptContext.rootElement = rootElement;

    vm.runInContext('applyObservatoryAppearance(rootElement, "dark")', scriptContext);
    assert.equal(rootElement.rootAttributes.get("data-theme"), "dark");
    vm.runInContext('applyObservatoryAppearance(rootElement, "auto")', scriptContext);
    assert.equal(rootElement.rootAttributes.has("data-theme"), false);
});

test("should_store_apply_and_mark_the_option_that_is_clicked", () => {
    const scriptContext = createAppearanceContext();
    const appearanceOptions = ["auto", "light", "dark"].map(createAppearanceOption);
    const rootElement = createRootElement();
    scriptContext.appearanceStorage = createAppearanceStorage("light");
    scriptContext.appearanceDocument = {
        documentElement: rootElement,
        getElementById: () => ({ querySelectorAll: () => appearanceOptions })
    };

    vm.runInContext("wireObservatoryAppearanceSwitch(appearanceDocument, appearanceStorage)", scriptContext);
    assert.deepEqual(appearanceOptions.map((option) => option.optionAttributes["aria-pressed"]), ["false", "true", "false"]);
    appearanceOptions[2].click();

    assert.equal(scriptContext.appearanceStorage.storedItems.get("astronomical-observatory-appearance"), "dark");
    assert.equal(rootElement.rootAttributes.get("data-theme"), "dark");
    assert.deepEqual(appearanceOptions.map((option) => option.optionAttributes["aria-pressed"]), ["false", "false", "true"]);
});

test("should_do_nothing_on_a_page_without_the_switch", () => {
    const scriptContext = createAppearanceContext();
    scriptContext.appearanceStorage = createAppearanceStorage("dark");
    scriptContext.appearanceDocument = { getElementById: () => null };

    assert.doesNotThrow(
        () => vm.runInContext("wireObservatoryAppearanceSwitch(appearanceDocument, appearanceStorage)", scriptContext)
    );
});
