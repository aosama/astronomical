// Shared Observatory console test fixtures: the vm context that loads the
// shipped console scripts, and a document whose elements record what was written.

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const consoleScriptPath = path.join(__dirname, "console.js");
const consoleScript = fs.readFileSync(consoleScriptPath, "utf8");
const memoryControlScriptPath = path.join(__dirname, "memory-control.js");
const memoryControlScript = fs.readFileSync(memoryControlScriptPath, "utf8");
const neuralEnginePanelScriptPath = path.join(__dirname, "neural-engine-panel.js");
const neuralEnginePanelScript = fs.readFileSync(neuralEnginePanelScriptPath, "utf8");
const overviewCompactScriptPath = path.join(__dirname, "overview-compact.js");
const overviewCompactScript = fs.readFileSync(overviewCompactScriptPath, "utf8");

function createConsoleContext() {
    const scriptContext = vm.createContext({
        console: { log() {} },
        document: { addEventListener() {} },
        setInterval() {},
        history: {
            pushState() {},
            replaceState() {},
            state: null
        },
        TextEncoder
    });
    vm.runInContext(memoryControlScript, scriptContext, { filename: memoryControlScriptPath });
    vm.runInContext(neuralEnginePanelScript, scriptContext, { filename: neuralEnginePanelScriptPath });
    vm.runInContext(consoleScript, scriptContext, { filename: consoleScriptPath });
    vm.runInContext(overviewCompactScript, scriptContext, { filename: overviewCompactScriptPath });
    return scriptContext;
}

function attachRecordingElements(scriptContext) {
    const elementsById = {};
    scriptContext.document = {
        addEventListener() {},
        getElementById(elementId) {
            if (!elementsById[elementId]) {
                elementsById[elementId] = { id: elementId, textContent: "", style: {}, dataset: {} };
            }
            return elementsById[elementId];
        }
    };
    return elementsById;
}

module.exports = { createConsoleContext, attachRecordingElements };
