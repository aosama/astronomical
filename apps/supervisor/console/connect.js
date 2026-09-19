// Observatory connection material: the values a user needs to point an
// OpenAI-compatible coding agent at this Mac.
//
// Every published value is derived from the console that renders it. The
// loopback port belongs to the instance serving the page, so a hardwired sample
// would silently rot on any machine that runs the other channel, and a copied
// sample that disagrees with the live port is worse than no sample at all.

const CONNECTION_PROVIDER_IDENTIFIER = "astronomical";
const CONNECTION_PROVIDER_DISPLAY_NAME = "Astronomical (local)";

// Astronomical's loopback API performs no authorization check, yet coding agents
// commonly keep a model unavailable until some key exists. An inert, obviously
// non-secret placeholder keeps those agents usable without inviting users to
// paste a real credential into a loopback configuration file.
const CONNECTION_CREDENTIAL_PLACEHOLDER = "astronomical";

// Publishing a concrete identifier is impossible before a model is ready, and
// inventing a plausible one would teach users a value the daemon rejects, so the
// fallback is named to read as an example rather than as an installable model.
const CONNECTION_EXAMPLE_MODEL_IDENTIFIER = "example-model-not-downloaded";

const CONNECTION_MODEL_LIST_SENTENCE =
    "GET /v1/models lists what this Mac can serve, and a request naming anything else is rejected.";

const CONNECTION_ENDPOINT_PATHS = [
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/models",
    "/v1/embeddings",
    "/v1/images/generations"
];

const CONNECTION_ADDRESS_EXPLANATION =
    "This page is served by the running instance on this Mac, so this address and port are the ones it is listening on right now. The API is loopback-only: nothing outside this Mac can reach it.";

const CONNECTION_CREDENTIAL_GUIDANCE =
    "The loopback API performs no authorization check, so no key can be wrong and none is required. An agent that insists on a key can be given this inert placeholder; it is not a secret and never leaves this Mac.";

const CONNECTION_MODEL_GUIDANCE_WITHOUT_MODELS =
    "No model is installed on this Mac yet. Download one from the Library, then replace the example identifier in the samples below.";

function connectionConsoleOrigin() {
    return typeof window !== "undefined" && window.location
        ? String(window.location.origin || "")
        : "";
}

// The console may be reached at any observatory path, and a trailing slash is
// common in copied URLs, so the origin is taken from the leading authority and
// every path segment is discarded.
function connectionApiBaseUrl(consoleOrigin) {
    const consoleAddress = String(consoleOrigin || "").trim();
    const authorityMatch = consoleAddress.match(/^https?:\/\/[^/]+/);
    const authority = authorityMatch ? authorityMatch[0] : consoleAddress.replace(/\/+$/, "");
    return `${authority}/v1`;
}

function connectionModelIdentifiers(modelsResponse) {
    const advertisedModels = (modelsResponse && modelsResponse.data) || [];
    return advertisedModels
        .map((advertisedModel) => advertisedModel && advertisedModel.id)
        .filter((modelIdentifier) => typeof modelIdentifier === "string" && modelIdentifier.length > 0);
}

// The samples follow the model this console has already selected, so the value a
// user copies matches the one the Chat and Model views are showing instead of an
// arbitrary entry of the advertised list.
function connectionModelIdentifier({ modelIdentifiers, selectedModelIdentifier }) {
    if (typeof selectedModelIdentifier === "string" && selectedModelIdentifier.length > 0) {
        return selectedModelIdentifier;
    }
    return modelIdentifiers.length > 0
        ? modelIdentifiers[0]
        : CONNECTION_EXAMPLE_MODEL_IDENTIFIER;
}

// The guidance names the same model the samples embed, so a reader is never told
// one provenance while the copyable block shows another.
function connectionModelGuidanceText({ modelIdentifiers, selectedModelIdentifier }) {
    if (modelIdentifiers.length === 0) {
        return CONNECTION_MODEL_GUIDANCE_WITHOUT_MODELS;
    }
    const sampleProvenance = selectedModelIdentifier
        ? "The samples below use the model this console has selected"
        : "The samples below use the first advertised model";
    return `${sampleProvenance}, and any identifier in this list works. ${CONNECTION_MODEL_LIST_SENTENCE}`;
}

function opencodeProviderConfigurationSnippet(apiBaseUrl, modelIdentifier) {
    return JSON.stringify({
        "$schema": "https://opencode.ai/config.json",
        provider: {
            [CONNECTION_PROVIDER_IDENTIFIER]: {
                npm: "@ai-sdk/openai-compatible",
                name: CONNECTION_PROVIDER_DISPLAY_NAME,
                options: {
                    baseURL: apiBaseUrl,
                    apiKey: CONNECTION_CREDENTIAL_PLACEHOLDER
                },
                models: {
                    [modelIdentifier]: { name: modelIdentifier }
                }
            }
        }
    }, null, 2);
}

function piProviderConfigurationSnippet(apiBaseUrl, modelIdentifier) {
    return JSON.stringify({
        providers: {
            [CONNECTION_PROVIDER_IDENTIFIER]: {
                baseUrl: apiBaseUrl,
                api: "openai-completions",
                apiKey: CONNECTION_CREDENTIAL_PLACEHOLDER,
                models: [
                    {
                        id: modelIdentifier,
                        name: modelIdentifier
                    }
                ]
            }
        }
    }, null, 2);
}

// A model identifier reaches the shell through a JSON body, so the value is
// single-quoted with the standard escape rather than interpolated bare.
function shellSingleQuotedValue(value) {
    return `'${String(value).replace(/'/g, "'\\''")}'`;
}

// A reasoning model spends budget on thinking before it emits any answer, so a
// stingy cap returns empty content and reads as a broken endpoint. Measured on a
// thinking model: 1024 tokens completes in about 11 seconds, 4096 takes about 50
// seconds, and neither is guaranteed to reach visible prose, because the contract
// resolves `reasoning_effort` off/none to model-default thinking rather than
// disabling it. The verification therefore publishes a JSON reply, not a finished
// sentence, as the signal that the endpoint is reachable.
const CONNECTION_VERIFICATION_TOKEN_BUDGET = 1024;

function connectionVerificationCommand(apiBaseUrl, modelIdentifier) {
    const requestBody = JSON.stringify({
        model: modelIdentifier,
        messages: [{ role: "user", content: "Say hello in one short sentence." }],
        max_tokens: CONNECTION_VERIFICATION_TOKEN_BUDGET
    });
    return [
        `curl -s ${shellSingleQuotedValue(`${apiBaseUrl}/chat/completions`)} \\`,
        `  -H ${shellSingleQuotedValue("Content-Type: application/json")} \\`,
        `  -d ${shellSingleQuotedValue(requestBody)}`
    ].join("\n");
}

function renderConnectionText(elementId, text) {
    const element = document.getElementById(elementId);
    if (!element) { return; }
    element.textContent = text;
}

function renderConnectionListItems(elementId, listItemTexts) {
    const container = document.getElementById(elementId);
    if (!container) { return; }
    const listItemElements = listItemTexts.map((listItemText) => {
        const listItemElement = document.createElement("li");
        listItemElement.textContent = listItemText;
        return listItemElement;
    });
    container.replaceChildren(...listItemElements);
}

function renderConnectionMaterial({ consoleOrigin, modelIdentifiers, selectedModelIdentifier }) {
    const apiBaseUrl = connectionApiBaseUrl(consoleOrigin);
    const modelIdentifier = connectionModelIdentifier({ modelIdentifiers, selectedModelIdentifier });

    renderConnectionText("connect-api-base-url", apiBaseUrl);
    renderConnectionText("connect-port-explanation", CONNECTION_ADDRESS_EXPLANATION);
    renderConnectionListItems("connect-endpoint-list", CONNECTION_ENDPOINT_PATHS);
    renderConnectionText("connect-credential-guidance", CONNECTION_CREDENTIAL_GUIDANCE);
    renderConnectionText("connect-credential-placeholder", CONNECTION_CREDENTIAL_PLACEHOLDER);
    renderConnectionText(
        "connect-model-guidance",
        connectionModelGuidanceText({ modelIdentifiers, selectedModelIdentifier })
    );
    renderConnectionListItems("connect-model-identifiers", modelIdentifiers);
    renderConnectionText(
        "connect-opencode-snippet",
        opencodeProviderConfigurationSnippet(apiBaseUrl, modelIdentifier)
    );
    renderConnectionText(
        "connect-pi-snippet",
        piProviderConfigurationSnippet(apiBaseUrl, modelIdentifier)
    );
    renderConnectionText(
        "connect-verification-command",
        connectionVerificationCommand(apiBaseUrl, modelIdentifier)
    );
}

// Between the first paint and the first model poll the address is already knowable
// from the console itself, so the view never shows an empty endpoint.
function wireConnectMaterial() {
    renderConnectionMaterial({
        consoleOrigin: connectionConsoleOrigin(),
        modelIdentifiers: [],
        selectedModelIdentifier: ""
    });
}