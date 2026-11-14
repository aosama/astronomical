/** The part of `window` this needs: somewhere to hear that focus came back. */
export interface FocusSource {
  addEventListener(eventName: "focus", listener: () => void): void;
}

/** The part of `document` this needs: whether the page is showing, and when
 * that changes. */
export interface VisibilitySource {
  readonly visibilityState: string;
  addEventListener(eventName: "visibilitychange", listener: () => void): void;
}

/**
 * Runs `refresh` each time the person comes back to the chat window.
 *
 * Models are added outside this window: "Choose a different model" opens the
 * Models page in the browser, and a model link or the terminal can install
 * one too. The window used to list its models once at start-up and again only
 * after its own download finished, so someone who downloaded a model in the
 * browser came back to "No chat model is available yet" until they reopened
 * the window. Coming back is the moment the list can have changed, so that is
 * when it is read again. A page that becomes hidden asks for nothing.
 */
export function refreshOnReturnToWindow(focusSource: FocusSource, visibilitySource: VisibilitySource, refresh: () => void): void {
  focusSource.addEventListener("focus", refresh);
  visibilitySource.addEventListener("visibilitychange", () => {
    if (visibilitySource.visibilityState === "visible") {
      refresh();
    }
  });
}
