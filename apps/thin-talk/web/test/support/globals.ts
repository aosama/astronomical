import { installDom } from "./dom";

// Installs the happy-dom globals at module-eval time, before any client module
// is imported: DOMPurify binds to the global window when its module evaluates,
// so the DOM must already exist by then.
installDom();
