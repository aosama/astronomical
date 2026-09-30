import { installPreviewShim } from "./previewShim";

/**
 * Preview-shim bundle entrypoint. The dev server loads this as an external
 * same-origin script (`/preview/preview-shim.js`) so it passes the app's strict
 * CSP, which forbids inline scripts. Installing over the real `window` — whose
 * `webkit`, `__thintalkBridge`, and `localStorage` are declared in
 * `vendor-globals.d.ts` — wires the native-side message bridge used by the
 * preview shell before the production bundle boots.
 */
installPreviewShim(window, window.localStorage);
