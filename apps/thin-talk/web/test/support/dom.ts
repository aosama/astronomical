import { readFileSync } from "node:fs";
import { Window } from "happy-dom";

/**
 * Boots a happy-dom window carrying the real index.html body (its script tags
 * are inert through innerHTML) and installs the globals the client reads at
 * call time: window, document, the DOM constructors, and the host-injected
 * config. Returns the window so tests can drive it.
 */
export function installDom(): Window {
  const indexPath = process.env["THINTALK_INDEX_HTML"];
  if (!indexPath) {
    throw new Error("THINTALK_INDEX_HTML must point at the real index.html");
  }
  const html = readFileSync(indexPath, "utf8");
  const domWindow = new Window({ url: "http://localhost/" });
  const bodyStart = html.indexOf("<body");
  const bodyEnd = html.lastIndexOf("</body>");
  if (bodyStart < 0 || bodyEnd < 0) {
    throw new Error("index.html has no body to install");
  }
  const bodyOpenEnd = html.indexOf(">", bodyStart);
  domWindow.document.body.innerHTML = html.slice(bodyOpenEnd + 1, bodyEnd);

  const globalTarget = globalThis as unknown as Record<string, unknown>;
  const defineGlobal = (key: string, value: unknown): void => {
    Object.defineProperty(globalTarget, key, { value, writable: true, configurable: true });
  };
  defineGlobal("window", domWindow);
  defineGlobal("document", domWindow.document);
  for (const key of [
    "HTMLElement", "HTMLTextAreaElement", "HTMLButtonElement", "HTMLInputElement",
    "Element", "Node", "Event", "CustomEvent", "KeyboardEvent", "MouseEvent",
    "MutationObserver", "getComputedStyle", "navigator", "location",
  ]) {
    const value = (domWindow as unknown as Record<string, unknown>)[key];
    if (value !== undefined) {
      defineGlobal(key, value);
    }
  }
  return domWindow;
}

/** The config the Swift host would inject at document start. */
export function installConfig(supervisorBaseURL: string): void {
  window.__thintalkConfig = {
    supervisorBaseURL,
    expectedChannel: "development",
    expectedStateDirectory: "~/.astronomical-dev",
  };
}

/** Waits until the predicate holds, draining microtasks and timers. */
export async function waitFor(predicate: () => boolean, label: string): Promise<void> {
  const deadline = Date.now() + 5_000;
  while (!predicate()) {
    if (Date.now() > deadline) {
      throw new Error(`timed out waiting for ${label}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
