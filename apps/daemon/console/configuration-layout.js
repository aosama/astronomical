// Astronomical Observatory configuration layout: the pure rules that arrange the
// schema's settings onto the Settings and Tuning pages and that summarise unsaved
// edits. The schema owns the arrangement (its x-astronomical-layout member), so a new
// setting needs a schema entry, not a page change. Nothing here touches the document
// or the network, which is what keeps it testable.

const CONFIGURATION_LAYOUT_KEY = "x-astronomical-layout";
const CONFIGURATION_OTHER_GROUP_TITLE = "Other";
const CONFIGURATION_MODELS_KEY = "models";

function flattenConfigurationLeafDescriptors(descriptors) {
    const leafDescriptors = [];
    for (const descriptor of descriptors) {
        if (descriptor.kind === "group") {
            leafDescriptors.push(...flattenConfigurationLeafDescriptors(descriptor.children));
        } else if (descriptor.kind !== "keyed_objects") {
            leafDescriptors.push(descriptor);
        }
    }
    return leafDescriptors;
}

// Every editable setting outside the per-model entries, keyed by its path in the file.
function configurationRootLeafDescriptors(schema) {
    const rootDescriptors = configurationSettingDescriptors(schema, schema, [])
        .filter((descriptor) => descriptor.key !== CONFIGURATION_MODELS_KEY);
    return new Map(flattenConfigurationLeafDescriptors(rootDescriptors)
        .map((descriptor) => [descriptor.path.join("."), descriptor]));
}

// Every editable setting of one model entry, keyed by its path inside the entry.
function configurationModelLeafDescriptors(schema) {
    const modelEntryNode = schema.properties[CONFIGURATION_MODELS_KEY].additionalProperties;
    return new Map(flattenConfigurationLeafDescriptors(
        configurationSettingDescriptors(schema, modelEntryNode, []))
        .map((descriptor) => [descriptor.path.join("."), descriptor]));
}

// A setting that exists both at the top of the file and inside a model entry is a
// default every model starts from, so it belongs on the Tuning page's All models entry.
function configurationDefaultsForAllModels(schema) {
    const rootLeaves = configurationRootLeafDescriptors(schema);
    const modelLeaves = configurationModelLeafDescriptors(schema);
    return new Map([...modelLeaves].filter(([settingPath]) => rootLeaves.has(settingPath)));
}

function configurationInstanceLeafDescriptors(schema) {
    const modelLeaves = configurationModelLeafDescriptors(schema);
    return new Map([...configurationRootLeafDescriptors(schema)]
        .filter(([settingPath]) => !modelLeaves.has(settingPath)));
}

function resolveConfigurationLayoutGroups(layoutGroups, leafDescriptors) {
    const listedPaths = new Set();
    const groups = layoutGroups.map((layoutGroup) => {
        const descriptors = layoutGroup.settings
            .filter((settingPath) => leafDescriptors.has(settingPath))
            .map((settingPath) => { listedPaths.add(settingPath); return leafDescriptors.get(settingPath); });
        return { title: layoutGroup.title, description: layoutGroup.description || "",
            tier: layoutGroup.tier || "standard",
            note: layoutGroup.note || "", descriptors };
    });
    const unlistedDescriptors = [...leafDescriptors.values()]
        .filter((descriptor) => !listedPaths.has(descriptor.path.join(".")));
    if (unlistedDescriptors.length > 0) {
        groups.push({ title: CONFIGURATION_OTHER_GROUP_TITLE, description: "", tier: "standard", note: "",
            descriptors: unlistedDescriptors });
    }
    return groups.filter((group) => group.descriptors.length > 0);
}

// pageName is "settings" or "tuning". The daemon sorts object members, so the order
// lives in the schema's arrays and never in the order of its properties.
// unavailableSettingPaths are settings the running instance says it cannot honour
// (a sandboxed build cannot open folders of the person's choosing): they are left
// off the page altogether, and a group left with nothing to show goes with them.
function configurationPageGroups(schema, pageName, unavailableSettingPaths) {
    const layout = schema[CONFIGURATION_LAYOUT_KEY] || {};
    const pageLeafDescriptors = pageName === "settings"
        ? configurationInstanceLeafDescriptors(schema)
        : configurationModelLeafDescriptors(schema);
    const unavailablePaths = new Set(unavailableSettingPaths || []);
    const availableLeafDescriptors = new Map([...pageLeafDescriptors]
        .filter(([settingPath]) => !unavailablePaths.has(settingPath)));
    return resolveConfigurationLayoutGroups(layout[pageName] || [], availableLeafDescriptors);
}

// The groups the All models entry shows: only the settings every model starts from.
function configurationGroupsForAllModels(schema, tuningGroups) {
    const defaultPaths = configurationDefaultsForAllModels(schema);
    return tuningGroups
        .map((group) => ({ ...group,
            descriptors: group.descriptors.filter((descriptor) => defaultPaths.has(descriptor.path.join("."))) }))
        .filter((group) => group.descriptors.length > 0);
}

// A path inside the file: relative paths are used as they are for All models and
// below models.<id> for one model.
function configurationScopedPath(modelIdentifier, relativePath) {
    return modelIdentifier === null
        ? relativePath
        : [CONFIGURATION_MODELS_KEY, modelIdentifier].concat(relativePath);
}

// What a setting falls back to when it is unset in this scope: a model inherits the
// All models value when the file sets one, otherwise the schema default.
function configurationBaselineValue(schema, configurationDocument, modelIdentifier, descriptor, stateDirectory) {
    if (modelIdentifier !== null
        && configurationDefaultsForAllModels(schema).has(descriptor.path.join("."))) {
        const inheritedValue = configurationValueAtPath(configurationDocument, descriptor.path);
        if (inheritedValue !== undefined) {
            return { value: inheritedValue, isInherited: true };
        }
    }
    // A folder setting left empty means a named folder inside this instance's own folder.
    // Showing that real path tells a person where things go, which "default" never did.
    if (descriptor.defaultValue === undefined && descriptor.instanceFolderDefault && stateDirectory) {
        return { value: `${stateDirectory}/${descriptor.instanceFolderDefault}`, isInherited: false };
    }
    return { value: descriptor.defaultValue, isInherited: false };
}

// The reset arrow is always shown for a value that changes behaviour. A value that
// equals its baseline only restates it, so the page shows the arrow on hover alone.
function configurationValueDiffersFromBaseline(currentValue, baselineValue) {
    if (currentValue === undefined) { return false; }
    if (baselineValue === undefined) { return true; }
    return canonicalConfigurationJson(currentValue) !== canonicalConfigurationJson(baselineValue);
}

// The arrow's three states as the class the stylesheet keys on: nothing to reset,
// always shown, or shown only while the row is hovered or focused.
function configurationResetArrowState(currentValue, baselineValue) {
    if (currentValue === undefined) { return "is-hidden"; }
    return configurationValueDiffersFromBaseline(currentValue, baselineValue) ? "is-visible" : "is-hover-only";
}

// Switching a setting back to its baseline removes it, so the file stays free of
// restated defaults, unless the saved file already pinned a value there.
function configurationShouldRemoveValue(newValue, baselineValue, savedFileHasValue) {
    if (savedFileHasValue || baselineValue === undefined) { return false; }
    return canonicalConfigurationJson(newValue) === canonicalConfigurationJson(baselineValue);
}

// A value as a person says it: null is "automatic", booleans are on or off.
function configurationDisplayValue(value) {
    if (value === null) { return "automatic"; }
    if (Array.isArray(value)) { return value.length === 0 ? "none" : value.join(", "); }
    if (typeof value === "boolean") { return value ? "on" : "off"; }
    return String(value);
}

// What an unset setting does, in the explanation's words.
function configurationFallbackDescription(descriptor, baselineValue) {
    if (baselineValue !== undefined) { return `Default: ${configurationDisplayValue(baselineValue)}`; }
    if (descriptor.builtinDefault) { return `Default: ${descriptor.builtinDefault}, built in`; }
    if (descriptor.unsetText) { return `Default: ${descriptor.unsetText.toLowerCase()}`; }
    return "No default";
}

function configurationSettingNote(descriptor, baseline) {
    const sentences = [];
    const applyModeDescription = configurationApplyModeDescription(descriptor.applyMode);
    if (applyModeDescription) { sentences.push(`${applyModeDescription}.`); }
    if (baseline.isInherited) {
        sentences.push(`Inherits ${configurationDisplayValue(baseline.value)} from All models.`);
    } else {
        sentences.push(`${configurationFallbackDescription(descriptor, baseline.value)}.`);
    }
    return sentences.join(" ");
}

function flattenConfigurationValues(configurationValue, pathSoFar, flattenedValues) {
    const isBranch = configurationValue !== null && typeof configurationValue === "object"
        && !Array.isArray(configurationValue);
    if (!isBranch) {
        flattenedValues.set(JSON.stringify(pathSoFar), configurationValue);
        return flattenedValues;
    }
    for (const [memberKey, memberValue] of Object.entries(configurationValue)) {
        flattenConfigurationValues(memberValue, pathSoFar.concat(memberKey), flattenedValues);
    }
    return flattenedValues;
}

function configurationChangedPaths(savedDocument, workingDocument) {
    const savedValues = flattenConfigurationValues(savedDocument, [], new Map());
    const workingValues = flattenConfigurationValues(workingDocument, [], new Map());
    const changedPaths = [];
    for (const pathKey of new Set([...savedValues.keys(), ...workingValues.keys()])) {
        if (canonicalConfigurationJson(savedValues.get(pathKey))
            !== canonicalConfigurationJson(workingValues.get(pathKey))) {
            changedPaths.push(JSON.parse(pathKey));
        }
    }
    return changedPaths;
}

function configurationDescriptorForChangedPath(schema, changedPath) {
    if (changedPath[0] === CONFIGURATION_MODELS_KEY) {
        return configurationModelLeafDescriptors(schema).get(changedPath.slice(2).join("."));
    }
    return configurationRootLeafDescriptors(schema).get(changedPath.join("."));
}

function configurationPageOfChangedPath(schema, changedPath) {
    if (changedPath[0] === CONFIGURATION_MODELS_KEY) { return "tuning"; }
    return configurationDefaultsForAllModels(schema).has(changedPath.join(".")) ? "tuning" : "settings";
}

// What the Save bar says: how many edits, which page holds them, and which of them
// restart something. The file is one document, so the count spans both pages.
function configurationChangeSummary(schema, savedDocument, workingDocument) {
    const summary = { changeCount: 0, workerRestartCount: 0, applicationRestartCount: 0,
        editedPages: { settings: false, tuning: false } };
    for (const changedPath of configurationChangedPaths(savedDocument, workingDocument)) {
        summary.changeCount += 1;
        summary.editedPages[configurationPageOfChangedPath(schema, changedPath)] = true;
        const descriptor = configurationDescriptorForChangedPath(schema, changedPath);
        const applyMode = descriptor ? descriptor.applyMode : undefined;
        if (applyMode === "worker-replacement") { summary.workerRestartCount += 1; }
        if (applyMode === "application-restart") { summary.applicationRestartCount += 1; }
    }
    return summary;
}

function configurationChangeSummaryText(summary) {
    const countText = summary.changeCount === 1 ? "1 change" : `${summary.changeCount} changes`;
    const restartPhrases = [];
    if (summary.workerRestartCount > 0) {
        restartPhrases.push(`${summary.workerRestartCount} restarts the inference worker`);
    }
    if (summary.applicationRestartCount > 0) {
        restartPhrases.push(`${summary.applicationRestartCount} needs an app restart`);
    }
    return { countText, detailText: restartPhrases.join(" · ") };
}

function countConfigurationLeavesUnder(configurationDocument, parentPath) {
    const parentValue = configurationValueAtPath(configurationDocument, parentPath);
    if (parentValue === undefined) { return 0; }
    return flattenConfigurationValues(parentValue, [], new Map()).size;
}

// The key of the models entry that configures a model. The file may key it by the
// model's folder name or by its full Hugging Face ID (organization/model); the daemon
// applies the exact folder name first, then the full ID. The console does not know the
// recorded full ID, so it takes the one file key whose last segment is the folder name,
// which is the same entry unless two organisations share a model name.
function configurationEntryKeyForModel(configurationDocument, modelIdentifier) {
    const fileKeys = Object.keys((configurationDocument && configurationDocument.models) || {});
    if (fileKeys.includes(modelIdentifier)) { return modelIdentifier; }
    const aliasKeys = fileKeys.filter((fileKey) => fileKey.includes("/")
        && fileKey.split("/").pop() === modelIdentifier).sort();
    return aliasKeys.length > 0 ? aliasKeys[0] : modelIdentifier;
}

// The Tuning page's list: every model the daemon discovers, plus any key the file
// already holds for a model it does not, so no hand-written entry becomes unreachable.
// A key that configures a discovered model is that model's row, never a second one.
function configurationTuningListEntries(discoveredModels, configurationDocument, addedIdentifiers) {
    const fileModels = (configurationDocument && configurationDocument.models) || {};
    const entriesByIdentifier = new Map();
    const keysOfDiscoveredModels = new Set();
    for (const discoveredModel of discoveredModels) {
        keysOfDiscoveredModels.add(configurationEntryKeyForModel(configurationDocument, discoveredModel.id));
        entriesByIdentifier.set(discoveredModel.id, { id: discoveredModel.id,
            isDiscovered: true, servesChat: discoveredModel.servesChat });
    }
    for (const identifier of Object.keys(fileModels).concat(addedIdentifiers)) {
        if (!entriesByIdentifier.has(identifier) && !keysOfDiscoveredModels.has(identifier)) {
            entriesByIdentifier.set(identifier, { id: identifier, isDiscovered: false, servesChat: true });
        }
    }
    const rankOf = (entry) => (entry.servesChat ? (entry.isDiscovered ? 0 : 1) : 2);
    return [...entriesByIdentifier.values()]
        .map((entry) => ({ ...entry,
            settingCount: countConfigurationLeavesUnder(configurationDocument,
                [CONFIGURATION_MODELS_KEY, configurationEntryKeyForModel(configurationDocument, entry.id)]) }))
        .sort((first, second) => rankOf(first) - rankOf(second) || first.id.localeCompare(second.id));
}

function configurationModelHasUnsavedEdits(savedDocument, workingDocument, modelIdentifier) {
    const modelPath = [CONFIGURATION_MODELS_KEY, configurationEntryKeyForModel(workingDocument, modelIdentifier)];
    return canonicalConfigurationJson(configurationValueAtPath(savedDocument, modelPath))
        !== canonicalConfigurationJson(configurationValueAtPath(workingDocument, modelPath));
}

// /v1/models lists what each model can serve; a model that serves chat has the
// generation, limit and execution settings the Tuning page offers.
function configurationDiscoveredModelFromAdvertisement(advertisedModel) {
    const endpoints = advertisedModel.supported_endpoints || [];
    return { id: advertisedModel.id, servesChat: endpoints.includes("/v1/chat/completions") };
}

// What the Reset everything prompt counts: every value the file holds except the two
// format constants every document carries.
function configurationValueCount(configurationDocument) {
    if (configurationDocument === null || configurationDocument === undefined) { return 0; }
    const flattenedValues = flattenConfigurationValues(configurationDocument, [], new Map());
    return [...flattenedValues.keys()]
        .filter((pathKey) => !["$schema", "schema_version"].includes(JSON.parse(pathKey)[0])).length;
}

// A field shows the value in force, whether the file sets it or not, because a greyed
// hint reads as a disabled field. Only a setting with no concrete value is left empty.
function configurationFieldText(currentValue, baselineValue) {
    const shownValue = currentValue !== undefined ? currentValue : baselineValue;
    return shownValue === undefined || shownValue === null ? "" : String(shownValue);
}

// The words inside a field that has no concrete value to show.
function configurationPlaceholderText(descriptor) {
    return descriptor.unsetText || "Automatic";
}

// The choices of a setting that picks one of the installed chat models. The first is
// "no choice", named after what the app then uses; a saved model that is no longer
// installed stays listed so opening the page never silently changes it.
function configurationModelChoices(descriptor, discoveredModels, currentValue) {
    const chatModelIdentifiers = discoveredModels.filter((model) => model.servesChat)
        .map((model) => model.id).sort((first, second) => first.localeCompare(second));
    const choices = [{ value: "", label: descriptor.builtinDefault
        ? `Built in (${descriptor.builtinDefault})` : "Automatic" }];
    if (typeof currentValue === "string" && !chatModelIdentifiers.includes(currentValue)) {
        choices.push({ value: currentValue, label: `${currentValue} (not installed)` });
    }
    return choices.concat(chatModelIdentifiers.map((identifier) => ({ value: identifier, label: identifier })));
}
