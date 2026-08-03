const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const consoleScriptPath = path.join(__dirname, "console.js");
const consoleScript = fs.readFileSync(consoleScriptPath, "utf8");
const memoryControlScriptPath = path.join(__dirname, "memory-control.js");
const memoryControlScript = fs.readFileSync(memoryControlScriptPath, "utf8");
const playgroundScriptPath = path.join(__dirname, "playground.js");
const playgroundScript = fs.readFileSync(playgroundScriptPath, "utf8");

function createConsoleContext() {
    const scriptContext = vm.createContext({
        console: { log() {} },
        document: { addEventListener() {} },
        setInterval() {},
        TextEncoder
    });
    vm.runInContext(memoryControlScript, scriptContext, { filename: memoryControlScriptPath });
    vm.runInContext(consoleScript, scriptContext, { filename: consoleScriptPath });
    vm.runInContext(playgroundScript, scriptContext, { filename: playgroundScriptPath });
    return scriptContext;
}

test("selects the ready model metadata instead of the first advertised model", () => {
    const scriptContext = createConsoleContext();
    const selectedModel = vm.runInContext(
        'selectAdvertisedModel([{ id: "first" }, { id: "ready" }], "ready")',
        scriptContext
    );

    assert.equal(selectedModel.id, "ready");
});

test("computes whole decimal gigabyte bounds from exact status bytes", () => {
    const scriptContext = createConsoleContext();
    const bounds = vm.runInContext(
        "wholeDecimalGigabyteBounds({ minimum_mlx_memory_ceiling_bytes: 32000000001, machine_mlx_memory_ceiling_bytes: 40999999999 })",
        scriptContext
    );
    assert.deepEqual(JSON.parse(JSON.stringify(bounds)), { minimumGigabytes: 33, maximumGigabytes: 40 });
});

test("rejects a serialized chat request after base64 expansion exceeds the HTTP body limit", () => {
    const scriptContext = createConsoleContext();
    const oversizedSerializedRequest = "x".repeat(32 * 1024 * 1024 + 1);
    scriptContext.oversizedSerializedRequest = oversizedSerializedRequest;

    assert.equal(
        vm.runInContext(
            "chatRequestFitsHttpBodyLimit(oversizedSerializedRequest)",
            scriptContext
        ),
        false
    );
});

test("retains partial assistant output in conversation history after interruption", () => {
    const scriptContext = createConsoleContext();
    const assistantHistoryMessage = vm.runInContext(
        'assistantHistoryMessage({ assistantText: "partial", reasoningText: "working" })',
        scriptContext
    );

    assert.deepEqual(
        JSON.parse(JSON.stringify(assistantHistoryMessage)),
        {
            role: "assistant",
            content: "partial",
            reasoning_content: "working"
        }
    );
});

test("describes an attached image in the visible user transcript", () => {
    const scriptContext = createConsoleContext();
    const visibleMessage = vm.runInContext(
        'visibleUserMessageText({ content: [{ type: "text", text: "inspect" }, { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } }] })',
        scriptContext
    );

    assert.equal(visibleMessage, "inspect\n[Image attached]");
});
