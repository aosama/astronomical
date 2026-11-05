// Astronomical Observatory configuration model: the pure rules behind the
// Configuration page. The page is generated from the bundled JSON Schema, so a
// setting added to the schema appears here without a page change. Nothing in this
// file touches the document or the network, which is what keeps it testable.

const CONFIGURATION_LABEL_ACRONYMS = new Set(["mlx", "ssd", "kv", "moe", "mlp", "gb", "id"]);
const CONFIGURATION_APPLY_MODE_DESCRIPTIONS = {
    "live": "Applies immediately",
    "next-request": "Applies to the next request",
    "model-reload": "Applies when the model next loads",
    "worker-replacement": "Restarts the inference worker",
    "application-restart": "Needs an app restart",
};

function configurationSettingLabel(settingKey) {
    const words = settingKey.split("_").filter((word) => word.length > 0);
    return words
        .map((word, wordIndex) => {
            if (CONFIGURATION_LABEL_ACRONYMS.has(word)) { return word.toUpperCase(); }
            if (wordIndex === 0) { return word.charAt(0).toUpperCase() + word.slice(1); }
            return word;
        })
        .join(" ");
}

function resolveConfigurationSchemaNode(schema, schemaNode) {
    if (schemaNode && typeof schemaNode.$ref === "string") {
        const definitionName = schemaNode.$ref.replace("#/$defs/", "");
        return schema.$defs[definitionName];
    }
    return schemaNode;
}

function configurationSettingKind(schemaNode) {
    if (Array.isArray(schemaNode.enum)) { return "enum"; }
    if (Array.isArray(schemaNode.anyOf)) {
        const hasNull = schemaNode.anyOf.some((choice) => choice.type === "null");
        const integerChoice = schemaNode.anyOf.find((choice) => choice.type === "integer");
        return hasNull && integerChoice ? "nullable_integer" : "unsupported";
    }
    if (schemaNode.type === "array") { return "string_list"; }
    return schemaNode.type || "unsupported";
}

// Bounds for a nullable integer live on its integer choice, not the node itself.
function configurationNumericBounds(schemaNode) {
    const boundedNode = Array.isArray(schemaNode.anyOf)
        ? schemaNode.anyOf.find((choice) => choice.type === "integer") || schemaNode
        : schemaNode;
    return { minimum: boundedNode.minimum, maximum: boundedNode.maximum };
}

// One editable leaf, or a nested group of them. Constants such as `$schema` are
// not settings: changing one could only make the document invalid.
function configurationSettingDescriptors(schema, objectNode, parentPath) {
    const resolvedObject = resolveConfigurationSchemaNode(schema, objectNode);
    const requiredKeys = resolvedObject.required || [];
    const descriptors = [];
    for (const [settingKey, rawNode] of Object.entries(resolvedObject.properties || {})) {
        const schemaNode = resolveConfigurationSchemaNode(schema, rawNode);
        if (Object.prototype.hasOwnProperty.call(schemaNode, "const")) { continue; }
        const path = parentPath.concat(settingKey);
        const shared = {
            key: settingKey,
            path,
            label: configurationSettingLabel(settingKey),
            description: schemaNode.description || "",
            isRequired: requiredKeys.includes(settingKey),
        };
        if (schemaNode.type === "object" && schemaNode.additionalProperties
            && typeof schemaNode.additionalProperties === "object") {
            descriptors.push({ ...shared, kind: "keyed_objects",
                entryNode: schemaNode.additionalProperties });
        } else if (schemaNode.type === "object") {
            descriptors.push({ ...shared, kind: "group",
                children: configurationSettingDescriptors(schema, schemaNode, path) });
        } else {
            descriptors.push({ ...shared,
                kind: configurationSettingKind(schemaNode),
                defaultValue: schemaNode.default,
                enumValues: schemaNode.enum,
                applyMode: schemaNode["x-astronomical-apply-mode"],
                ...configurationNumericBounds(schemaNode) });
        }
    }
    return descriptors;
}

function configurationApplyModeDescription(applyMode) {
    return CONFIGURATION_APPLY_MODE_DESCRIPTIONS[applyMode] || "";
}

function cloneConfigurationDocument(configurationDocument) {
    return JSON.parse(JSON.stringify(configurationDocument));
}

function configurationValueAtPath(configurationDocument, path) {
    let currentValue = configurationDocument;
    for (const pathKey of path) {
        if (currentValue === null || typeof currentValue !== "object") { return undefined; }
        currentValue = currentValue[pathKey];
    }
    return currentValue;
}

function configurationHasValueAtPath(configurationDocument, path) {
    return configurationValueAtPath(configurationDocument, path) !== undefined;
}

function withConfigurationValue(configurationDocument, path, value) {
    const updatedDocument = cloneConfigurationDocument(configurationDocument);
    let currentObject = updatedDocument;
    for (const pathKey of path.slice(0, -1)) {
        if (currentObject[pathKey] === null || typeof currentObject[pathKey] !== "object") {
            currentObject[pathKey] = {};
        }
        currentObject = currentObject[pathKey];
    }
    currentObject[path[path.length - 1]] = value;
    return updatedDocument;
}

// Removing a setting returns it to its default. Parents left empty go too, because
// `"chunking": {}` says nothing, except the one object the format requires.
function withoutConfigurationValue(configurationDocument, path) {
    const updatedDocument = cloneConfigurationDocument(configurationDocument);
    const parentPath = path.slice(0, -1);
    const parentObject = configurationValueAtPath(updatedDocument, parentPath);
    if (parentObject === null || typeof parentObject !== "object") { return updatedDocument; }
    delete parentObject[path[path.length - 1]];
    const isRequiredObject = parentPath.length === 1 && parentPath[0] === "runtime";
    if (parentPath.length > 0 && !isRequiredObject && Object.keys(parentObject).length === 0) {
        return withoutConfigurationValue(updatedDocument, parentPath);
    }
    return updatedDocument;
}

function configurationDocumentsAreEqual(firstDocument, secondDocument) {
    return canonicalConfigurationJson(firstDocument) === canonicalConfigurationJson(secondDocument);
}

// Member order means nothing in this format, so it must not read as an edit.
function canonicalConfigurationJson(configurationValue) {
    if (Array.isArray(configurationValue)) {
        return `[${configurationValue.map(canonicalConfigurationJson).join(",")}]`;
    }
    if (configurationValue !== null && typeof configurationValue === "object") {
        const members = Object.keys(configurationValue).sort().map((memberKey) =>
            `${JSON.stringify(memberKey)}:${canonicalConfigurationJson(configurationValue[memberKey])}`);
        return `{${members.join(",")}}`;
    }
    return JSON.stringify(configurationValue);
}

function configurationBoundsMessage(descriptor, parsedNumber) {
    if (descriptor.minimum !== undefined && parsedNumber < descriptor.minimum) {
        return `Must be at least ${descriptor.minimum}`;
    }
    if (descriptor.maximum !== undefined && parsedNumber > descriptor.maximum) {
        return `Must be at most ${descriptor.maximum}`;
    }
    return "";
}

// A quick answer while typing. The daemon's validation on save stays the authority,
// because some rules cross settings or depend on the models it discovered.
function parseConfigurationInput(descriptor, rawText) {
    const trimmedText = String(rawText).trim();
    if (descriptor.kind === "string" || descriptor.kind === "enum") {
        if (trimmedText.length === 0) { return { isValid: false, message: "Enter a value" }; }
        return { isValid: true, value: trimmedText };
    }
    if (descriptor.kind === "nullable_integer" && trimmedText.toLowerCase() === "null") {
        return { isValid: true, value: null };
    }
    if (trimmedText.length === 0) { return { isValid: false, message: "Enter a number" }; }
    const parsedNumber = Number(trimmedText);
    if (!Number.isFinite(parsedNumber)) { return { isValid: false, message: "Enter a number" }; }
    const needsWholeNumber = descriptor.kind === "integer" || descriptor.kind === "nullable_integer";
    if (needsWholeNumber && !Number.isInteger(parsedNumber)) {
        return { isValid: false, message: "Enter a whole number" };
    }
    const boundsMessage = configurationBoundsMessage(descriptor, parsedNumber);
    if (boundsMessage) { return { isValid: false, message: boundsMessage }; }
    return { isValid: true, value: parsedNumber };
}

function configurationDefaultDescription(descriptor) {
    if (descriptor.defaultValue === undefined) { return "No default"; }
    if (descriptor.kind === "boolean") { return descriptor.defaultValue ? "Default: on" : "Default: off"; }
    if (descriptor.defaultValue === null) { return "Default: automatic"; }
    if (Array.isArray(descriptor.defaultValue)) { return "Default: none"; }
    return `Default: ${descriptor.defaultValue}`;
}

// The save itself and applying it are separate outcomes: a valid file stays saved
// even when a running generation or a restart delays when it takes effect.
function configurationSaveOutcomeMessage(saveOutcome) {
    const reload = saveOutcome.reload || {};
    if (saveOutcome.reload_http_status === 409) {
        return { tone: "progress",
            text: "Saved. It applies once the current request finishes; press Apply then." };
    }
    if (reload.rest_api_restart_required) {
        const restartFields = (reload.restart_required_fields || []).join(", ");
        return { tone: "progress",
            text: `Saved. Restart the app to apply: ${restartFields}.` };
    }
    if (saveOutcome.reload_http_status >= 400) {
        return { tone: "error",
            text: `Saved, but it could not be applied: ${reload.message || "see local diagnostics"}.` };
    }
    return { tone: "success", text: "Saved and applied." };
}

function neuralEnginePrefillStatusMessage(neuralEnginePrefill) {
    if (!neuralEnginePrefill) { return "Off for the loaded model, or no model is loaded."; }
    if (neuralEnginePrefill.state === "active") {
        const memoryGigabytes = neuralEnginePrefill.estimated_memory_bytes / 1073741824;
        return `Active: ${neuralEnginePrefill.neural_engine_tokens_per_chunk} of every `
            + `${neuralEnginePrefill.chunk_token_count} prompt tokens run on the Neural Engine, across `
            + `${neuralEnginePrefill.layer_program_count} layers, using about `
            + `${Math.ceil(memoryGigabytes)} GB.`;
    }
    if (neuralEnginePrefill.state === "unavailable") {
        return `Enabled but not running: ${neuralEnginePrefill.reason}.`;
    }
    return "Off for the loaded model.";
}
