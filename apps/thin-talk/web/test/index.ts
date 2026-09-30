// Every test file registers itself with node's test runner on import; the
// bundler concatenates them and the runner discovers the registrations.
// The shim tests use plain fakes (no DOM), so they never touch `window`/storage
// globals and can run in any order relative to the DOM-dependent suites.
// The globals module must come first: it installs the DOM that DOMPurify and
// the other window-reading modules bind to at import time.
import "./support/globals";
import "./sessionBridge.test";
import "./sessionDocument.test";
import "./sessionPreferences.test";
import "./chatJourney.test";
import "./previewBridgeShim.test";
