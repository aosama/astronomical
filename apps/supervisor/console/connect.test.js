const assert = require("node:assert/strict");
const { exec } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ASYNC_TEST_OPTIONS = { timeout: 5_000 };
const observatoryShellPath = path.join(__dirname, "index.html");

const connectScriptPath = path.join(__dirname, "connect.js");
const connectScript = fs.readFileSync(connectScriptPath, "utf8");

function createConnectContext() {
    const scriptContext = vm.createContext({ console: { log() {} } });
    vm.runInContext(connectScript, scriptContext, { filename: connectScriptPath });
    return scriptContext;
}

// The connect view writes only through textContent, so a fixture captures the
// visible contract a user reads instead of asserting on source layout.
function createConnectDocumentFixture() {
    const elements = {
        "connect-api-base-url": { textContent: "" },
        "connect-port-explanation": { textContent: "" },
        "connect-credential-guidance": { textContent: "" },
        "connect-credential-placeholder": { textContent: "" },
        "connect-endpoint-list": { replaceChildren(...children) { this.children = children; } },
        "connect-model-identifiers": { replaceChildren(...children) { this.children = children; } },
        "connect-model-guidance": { textContent: "" },
        "connect-opencode-snippet": { textContent: "" },
        "connect-pi-snippet": { textContent: "" },
        "connect-verification-command": { textContent: "" }
    };
    const connectDocument = {
        createElement(tagName) {
            return { tagName, textContent: "", className: "" };
        },
        getElementById(elementId) {
            return elements[elementId] || null;
        }
    };
    return { connectDocument, elements };
}

function renderedLineTexts(container) {
    return (container.children || []).map((child) => child.textContent);
}

// The published command is executed asynchronously so the stub server can answer
// on the same event loop the user's terminal would reach.
function runPublishedCommand(command) {
    return new Promise((resolve, reject) => {
        exec(command, { shell: "/bin/bash", timeout: 10000 }, (commandError, standardOutput, standardError) => {
            if (commandError) {
                reject(new Error(`published command failed: ${commandError.message} ${standardError}`));
                return;
            }
            resolve(standardOutput);
        });
    });
}

// The shell boot test exercises the whole shipped console, so its elements carry
// every property the other Observatory scripts touch. The Library-named fixture
// module is deliberately not reused here: coupling the connection view's coverage
// to another view's fixture would break it whenever the Library view changes.
function createObservatoryShellElement(tagName) {
    return {
        tagName: String(tagName).toUpperCase(),
        className: "",
        dataset: {},
        attributes: {},
        textContent: "",
        children: [],
        hidden: false,
        style: {},
        value: "",
        classList: {
            add() {},
            remove() {},
            toggle() {},
            contains() { return false; }
        },
        addEventListener() {},
        appendChild(childElement) { this.children.push(childElement); return childElement; },
        querySelectorAll() { return []; },
        querySelector() { return null; },
        removeAttribute(attributeName) { delete this.attributes[attributeName]; },
        setAttribute(attributeName, attributeValue) {
            this.attributes[attributeName] = attributeValue;
        },
        replaceChildren(...replacementChildren) { this.children = replacementChildren; }
    };
}

// Boots every script the shipped shell declares and returns the elements the
// connection view writes into, so the journey is verified through the artifact the
// daemon actually serves rather than through one module loaded in isolation.
async function bootShippedObservatoryShell({ origin, modelsResponse }) {
    const domReadyHandlers = [];
    const requestedPaths = [];
    const elementsById = new Map();
    const elementFor = (elementId) => {
        if (!elementsById.has(elementId)) {
            elementsById.set(elementId, createObservatoryShellElement("div"));
        }
        return elementsById.get(elementId);
    };
    const shellDocument = {
        body: createObservatoryShellElement("body"),
        addEventListener(eventName, eventHandler) {
            if (eventName === "DOMContentLoaded") { domReadyHandlers.push(eventHandler); }
        },
        createElement: createObservatoryShellElement,
        createTextNode(text) { return { tagName: "#text", textContent: text, children: [] }; },
        getElementById: elementFor,
        querySelectorAll() { return []; }
    };
    const scriptContext = vm.createContext({
        AbortController,
        clearInterval() {},
        clearTimeout,
        console: { log() {} },
        document: shellDocument,
        fetch: async (requestPath) => {
            requestedPaths.push(requestPath);
            if (requestPath === "/v1/models") {
                return { ok: true, async json() { return modelsResponse; } };
            }
            return { ok: false, async json() { return {}; } };
        },
        history: { pushState() {}, replaceState() {}, state: null },
        setInterval() {},
        setTimeout,
        TextEncoder,
        window: {
            addEventListener() {},
            location: { origin, pathname: "/connect" }
        }
    });
    const observatoryShell = fs.readFileSync(observatoryShellPath, "utf8");
    const scriptSources = Array.from(
        observatoryShell.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/g),
        (scriptMatch) => scriptMatch[1]
    );
    assert.ok(scriptSources.length > 0, "the shell must declare the Observatory scripts");
    for (const scriptSource of scriptSources) {
        const shippedScriptPath = path.join(__dirname, path.basename(scriptSource));
        vm.runInContext(
            fs.readFileSync(shippedScriptPath, "utf8"),
            scriptContext,
            { filename: shippedScriptPath }
        );
    }

    assert.equal(domReadyHandlers.length, 1, "the shell must boot exactly once");
    domReadyHandlers[0]();
    // Two turns let the boot-time model poll resolve its stubbed fetch and render.
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));

    return { elementFor, requestedPaths };
}

test("points a coding agent at the live local endpoint using only what the connect view shows", () => {
    const scriptContext = createConnectContext();
    const { connectDocument, elements } = createConnectDocumentFixture();
    scriptContext.document = connectDocument;

    scriptContext.renderConnectionMaterial({
        consoleOrigin: "http://127.0.0.1:6733",
        modelIdentifiers: ["example-unselected-model", "example-selected-model"]
    });

    assert.equal(elements["connect-api-base-url"].textContent, "http://127.0.0.1:6733/v1");
    assert.notEqual(elements["connect-port-explanation"].textContent, "");
    assert.match(elements["connect-credential-guidance"].textContent, /authorization/i);
    assert.equal(elements["connect-credential-placeholder"].textContent, "astronomical");
    assert.ok(renderedLineTexts(elements["connect-endpoint-list"]).includes("/v1/chat/completions"));
    assert.deepEqual(
        renderedLineTexts(elements["connect-model-identifiers"]),
        ["example-unselected-model", "example-selected-model"]
    );
    assert.match(elements["connect-model-guidance"].textContent, /\/v1\/models/);
    assert.match(elements["connect-opencode-snippet"].textContent, /127\.0\.0\.1:6733\/v1/);
    assert.match(elements["connect-opencode-snippet"].textContent, /example-unselected-model/);
    assert.match(elements["connect-pi-snippet"].textContent, /127\.0\.0\.1:6733\/v1/);
    assert.match(elements["connect-pi-snippet"].textContent, /example-unselected-model/);
    assert.match(elements["connect-verification-command"].textContent, /127\.0\.0\.1:6733\/v1\/chat\/completions/);
});

test("derives the API base URL from the console origin so it cannot drift from the live port", () => {
    const scriptContext = createConnectContext();

    assert.equal(
        scriptContext.connectionApiBaseUrl("http://127.0.0.1:6733"),
        "http://127.0.0.1:6733/v1"
    );
    assert.equal(
        scriptContext.connectionApiBaseUrl("http://127.0.0.1:6733/"),
        "http://127.0.0.1:6733/v1"
    );
    assert.equal(
        scriptContext.connectionApiBaseUrl("http://127.0.0.1:6732/connect"),
        "http://127.0.0.1:6732/v1"
    );
});

test("builds an opencode provider block that names the local endpoint and a servable model", () => {
    const scriptContext = createConnectContext();
    const snippet = scriptContext.opencodeProviderConfigurationSnippet(
        "http://127.0.0.1:6733/v1",
        "example-selected-model"
    );
    const configuration = JSON.parse(snippet);

    assert.equal(configuration.$schema, "https://opencode.ai/config.json");
    const provider = configuration.provider.astronomical;
    assert.equal(provider.npm, "@ai-sdk/openai-compatible");
    assert.equal(provider.options.baseURL, "http://127.0.0.1:6733/v1");
    assert.ok(provider.options.apiKey.length > 0);
    assert.deepEqual(Object.keys(provider.models), ["example-selected-model"]);
});

test("builds a pi provider block that names the local endpoint and a servable model", () => {
    const scriptContext = createConnectContext();
    const snippet = scriptContext.piProviderConfigurationSnippet(
        "http://127.0.0.1:6733/v1",
        "example-selected-model"
    );
    const configuration = JSON.parse(snippet);

    const provider = configuration.providers.astronomical;
    assert.equal(provider.baseUrl, "http://127.0.0.1:6733/v1");
    assert.equal(provider.api, "openai-completions");
    assert.ok(provider.apiKey.length > 0, "pi keeps keyless local servers available with an inert key");
    assert.deepEqual(provider.models.map((model) => model.id), ["example-selected-model"]);
});

test("states that the local endpoint performs no authorization check and needs only an inert placeholder", () => {
    const scriptContext = createConnectContext();
    const { connectDocument, elements } = createConnectDocumentFixture();
    scriptContext.document = connectDocument;

    scriptContext.renderConnectionMaterial({
        consoleOrigin: "http://127.0.0.1:6732",
        modelIdentifiers: ["example-selected-model"]
    });

    const advertisedPlaceholder = elements["connect-credential-placeholder"].textContent;
    assert.notEqual(advertisedPlaceholder, "");
    assert.equal(
        advertisedPlaceholder.includes("sk-"),
        false,
        "the placeholder must not look like a real credential"
    );
    assert.match(elements["connect-credential-guidance"].textContent, /loopback/i);
    assert.equal(
        JSON.parse(elements["connect-opencode-snippet"].textContent).provider.astronomical.options.apiKey,
        advertisedPlaceholder,
        "the key a user pastes must be the placeholder the view advertises"
    );
    assert.equal(
        JSON.parse(elements["connect-pi-snippet"].textContent).providers.astronomical.apiKey,
        advertisedPlaceholder
    );
});

// This is the published command, executed the way a user would paste it. A stub
// OpenAI-compatible server keeps the check hermetic: no model weights are loaded.
test("publishes a verification command that reaches an OpenAI-compatible endpoint", async () => {
    const scriptContext = createConnectContext();
    const receivedRequests = [];
    let receivedBody = "";
    const stubServer = http.createServer((request, response) => {
        receivedRequests.push({ method: request.method, url: request.url });
        request.on("data", (requestChunk) => { receivedBody += requestChunk; });
        request.on("end", () => {
            response.writeHead(200, { "Content-Type": "application/json" });
            response.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "hello" } }] }));
        });
    });
    await new Promise((resolve) => stubServer.listen(0, "127.0.0.1", resolve));
    const stubPort = stubServer.address().port;

    try {
        const verificationCommand = scriptContext.connectionVerificationCommand(
            `http://127.0.0.1:${stubPort}/v1`,
            "example-selected-model"
        );
        const commandOutput = await runPublishedCommand(verificationCommand);

        assert.equal(receivedRequests.length, 1);
        assert.equal(receivedRequests[0].method, "POST");
        assert.equal(receivedRequests[0].url, "/v1/chat/completions");
        assert.match(commandOutput, /hello/);

        // A reasoning model spends budget before it emits content, so a stingy cap
        // returns an empty answer and makes a healthy endpoint look broken.
        const publishedRequestBody = JSON.parse(receivedBody);
        assert.ok(
            publishedRequestBody.max_tokens >= 1024,
            `the published verification budget must survive reasoning, got ${publishedRequestBody.max_tokens}`
        );
    } finally {
        await new Promise((resolve) => stubServer.close(resolve));
    }
});

test("keeps the verification command shell-safe for a model identifier containing a quote", async () => {
    const scriptContext = createConnectContext();
    let receivedBody = "";
    const stubServer = http.createServer((request, response) => {
        request.on("data", (chunk) => { receivedBody += chunk; });
        request.on("end", () => {
            response.writeHead(200, { "Content-Type": "application/json" });
            response.end(JSON.stringify({ choices: [{ message: { content: "ok" } }] }));
        });
    });
    await new Promise((resolve) => stubServer.listen(0, "127.0.0.1", resolve));
    const stubPort = stubServer.address().port;

    try {
        const verificationCommand = scriptContext.connectionVerificationCommand(
            `http://127.0.0.1:${stubPort}/v1`,
            "model'with\"quotes"
        );
        await runPublishedCommand(verificationCommand);

        assert.equal(JSON.parse(receivedBody).model, "model'with\"quotes");
    } finally {
        await new Promise((resolve) => stubServer.close(resolve));
    }
});

test("uses the model the console has selected in the published samples", () => {
    const scriptContext = createConnectContext();
    const { connectDocument, elements } = createConnectDocumentFixture();
    scriptContext.document = connectDocument;

    scriptContext.renderConnectionMaterial({
        consoleOrigin: "http://127.0.0.1:6733",
        modelIdentifiers: ["example-unselected-model", "example-selected-model"],
        selectedModelIdentifier: "example-selected-model"
    });

    assert.match(elements["connect-opencode-snippet"].textContent, /"example-selected-model"/);
    assert.match(elements["connect-pi-snippet"].textContent, /"example-selected-model"/);
    assert.match(elements["connect-verification-command"].textContent, /example-selected-model/);
    assert.equal(
        elements["connect-opencode-snippet"].textContent.includes("example-unselected-model"),
        false,
        "an unselected identifier must not leak into the copyable sample"
    );
    assert.deepEqual(
        renderedLineTexts(elements["connect-model-identifiers"]),
        ["example-unselected-model", "example-selected-model"],
        "the advertised list keeps the daemon's own order"
    );
    const guidanceWithSelection = elements["connect-model-guidance"].textContent;

    scriptContext.renderConnectionMaterial({
        consoleOrigin: "http://127.0.0.1:6733",
        modelIdentifiers: ["example-unselected-model", "example-selected-model"],
        selectedModelIdentifier: ""
    });
    const guidanceWithoutSelection = elements["connect-model-guidance"].textContent;

    assert.notEqual(guidanceWithSelection, "");
    assert.notEqual(
        guidanceWithoutSelection,
        guidanceWithSelection,
        "the guidance must describe the provenance of the model the samples embed"
    );
});

test("names an obviously fictional example model when nothing is installed", () => {
    const scriptContext = createConnectContext();
    const { connectDocument, elements } = createConnectDocumentFixture();
    scriptContext.document = connectDocument;

    scriptContext.renderConnectionMaterial({
        consoleOrigin: "http://127.0.0.1:6733",
        modelIdentifiers: [],
        selectedModelIdentifier: ""
    });

    const exampleModelIdentifier = "example-model-not-downloaded";
    assert.match(elements["connect-opencode-snippet"].textContent, new RegExp(exampleModelIdentifier));
    assert.match(elements["connect-pi-snippet"].textContent, new RegExp(exampleModelIdentifier));
    assert.match(elements["connect-verification-command"].textContent, new RegExp(exampleModelIdentifier));
    assert.match(elements["connect-model-guidance"].textContent, /Library/);
    assert.deepEqual(renderedLineTexts(elements["connect-model-identifiers"]), []);
});

test("boots the shipped console shell into a readable connection view", ASYNC_TEST_OPTIONS, async () => {
    const { elementFor, requestedPaths } = await bootShippedObservatoryShell({
        origin: "http://127.0.0.1:6733",
        modelsResponse: { data: [{ id: "example-selected-model" }, { id: "example-other-model" }] }
    });

    assert.ok(requestedPaths.includes("/v1/models"), "the boot must ask the daemon for its models");
    assert.equal(elementFor("connect-api-base-url").textContent, "http://127.0.0.1:6733/v1");
    assert.deepEqual(
        elementFor("connect-model-identifiers").children.map((child) => child.textContent),
        ["example-selected-model", "example-other-model"]
    );
    assert.match(elementFor("connect-opencode-snippet").textContent, /example-selected-model/);
    assert.match(elementFor("connect-pi-snippet").textContent, /example-selected-model/);
    assert.match(elementFor("connect-verification-command").textContent, /example-selected-model/);
    assert.equal(
        elementFor("connect-endpoint-list").children.length,
        5,
        "the served endpoint list must be published on boot"
    );
    assert.notEqual(elementFor("connect-credential-placeholder").textContent, "");
});

test("lists only the model identifiers the running instance advertises", () => {
    const scriptContext = createConnectContext();

    assert.deepEqual(
        JSON.parse(JSON.stringify(scriptContext.connectionModelIdentifiers({
            data: [{ id: "example-unselected-model" }, { id: "example-selected-model" }, { id: null }]
        }))),
        ["example-unselected-model", "example-selected-model"]
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(scriptContext.connectionModelIdentifiers({}))),
        []
    );
});