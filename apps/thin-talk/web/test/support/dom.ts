import assert from "node:assert/strict";
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
    "HTMLDetailsElement", "Element", "Node", "Event", "CustomEvent", "KeyboardEvent",
    "MouseEvent", "MutationObserver", "getComputedStyle", "navigator", "location", "CSS",
  ]) {
    const value = (domWindow as unknown as Record<string, unknown>)[key];
    if (value !== undefined) {
      defineGlobal(key, value);
    }
  }
  return domWindow;
}

/** The config the Swift host would inject at document start. */
export function installConfig(daemonBaseURL: string, titlebarInset?: number, canRestartEngine?: boolean): void {
  window.__thintalkConfig = {
    daemonBaseURL,
    expectedChannel: "development",
    expectedStateDirectory: "~/.astronomical-dev",
    ...(titlebarInset === undefined ? {} : { titlebarInset }),
    ...(canRestartEngine === undefined ? {} : { canRestartEngine }),
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

/** A few words naming an element for a failure message: "button#send.primary". */
function describeElement(element: Element | null | undefined): string {
  if (element == null) {
    return "nothing";
  }
  const idPart = element.id.length > 0 ? `#${element.id}` : "";
  const classPart = element.classList.length > 0 ? `.${[...element.classList].join(".")}` : "";
  return `${element.tagName.toLowerCase()}${idPart}${classPart}`;
}

/*
 * Never hand a DOM node to assert.equal or assert.deepEqual. When such an
 * assert fails, Node prints both values, and printing a happy-dom node walks
 * the whole document through its parent and owner links. One failing assert
 * of that kind grew the test process past every limit and took the machine's
 * memory with it. The two helpers below compare first and only ever put a
 * short description in the failure message.
 */

/** Fails when `root` holds an element matching `selector`. */
export function assertNoElement(root: ParentNode | null | undefined, selector: string, message?: string): void {
  const foundElement = root != null ? root.querySelector(selector) : null;
  assert.ok(foundElement === null, `${message ?? `expected no "${selector}"`} (found ${describeElement(foundElement)})`);
}

/** Fails unless keyboard focus is on `expectedElement`. */
export function assertFocusOn(expectedElement: Element | null, message?: string): void {
  const focusedElement = document.activeElement;
  assert.ok(
    focusedElement === expectedElement,
    `${message ?? "keyboard focus"}: expected ${describeElement(expectedElement)}, focus is on ${describeElement(focusedElement)}`,
  );
}
