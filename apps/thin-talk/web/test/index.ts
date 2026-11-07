// Every test file registers itself with node's test runner on import; the
// bundler concatenates them and the runner discovers the registrations.
// The shim tests use plain fakes (no DOM), so they never touch `window`/storage
// globals and can run in any order relative to the DOM-dependent suites.
// The globals module must come first: it installs the DOM that DOMPurify and
// the other window-reading modules bind to at import time.
import "./support/globals";
import "./appConfig.test";
import "./sessionBridge.test";
import "./sessionDocument.test";
import "./sessionPreferences.test";
import "./chatJourney.test";
import "./modelSelection.test";
import "./reasoningDisclosure.test";
import "./stableIdentifierContract.test";
import "./previewBridgeShim.test";
import "./copyAnswer.test";
import "./transcriptRenderSkip.test";
import "./attachmentTrust.test";
import "./hostCommand.test";
import "./chatCompletionChunk.test";
import "./daemonClient.test";
import "./assistantEventAssembler.test";
import "./daemonStreamFunction.test";
import "./agentTurnRunner.test";
import "./conversationLog.test";
import "./toolStep.test";
import "./toolStepTracker.test";
import "./modelCatalogEntry.test";
import "./toolStepRendering.test";
import "./generateImageTool.test";
import "./stallWatchdog.test";
import "./agentToolAdapter.test";
import "./sessionBridgeImages.test";
import "./chatController.test";
import "./toolLimits.test";
import "./toolCatalog.test";
import "./currentTimeTool.test";
import "./agentHistoryReplay.test";
import "./toolJourney.test";
