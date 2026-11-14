import assert from "node:assert/strict";
import { test } from "node:test";
import { refreshOnReturnToWindow } from "../src/WindowReturnRefresh";

/** Stands in for `window` or `document`: remembers listeners and fires them. */
class EventSourceDouble {
  public visibilityState = "visible";
  private readonly listenersByEventName = new Map<string, Array<() => void>>();

  public addEventListener(eventName: string, listener: () => void): void {
    const registeredListeners = this.listenersByEventName.get(eventName) ?? [];
    registeredListeners.push(listener);
    this.listenersByEventName.set(eventName, registeredListeners);
  }

  public fire(eventName: string): void {
    for (const listener of this.listenersByEventName.get(eventName) ?? []) {
      listener();
    }
  }
}

function wiredSources(): { windowDouble: EventSourceDouble; documentDouble: EventSourceDouble; refreshCount: () => number } {
  const windowDouble = new EventSourceDouble();
  const documentDouble = new EventSourceDouble();
  let refreshCallCount = 0;
  refreshOnReturnToWindow(windowDouble, documentDouble, () => {
    refreshCallCount += 1;
  });
  return { windowDouble, documentDouble, refreshCount: () => refreshCallCount };
}

test("should_refresh_when_the_window_regains_focus", () => {
  const { windowDouble, refreshCount } = wiredSources();

  windowDouble.fire("focus");
  windowDouble.fire("focus");

  assert.equal(refreshCount(), 2);
});

test("should_refresh_when_the_page_becomes_visible_again", () => {
  const { documentDouble, refreshCount } = wiredSources();

  documentDouble.visibilityState = "visible";
  documentDouble.fire("visibilitychange");

  assert.equal(refreshCount(), 1);
});

test("should_not_refresh_when_the_page_becomes_hidden", () => {
  const { documentDouble, refreshCount } = wiredSources();

  documentDouble.visibilityState = "hidden";
  documentDouble.fire("visibilitychange");

  assert.equal(refreshCount(), 0);
});

test("should_not_refresh_before_the_person_has_left_and_come_back", () => {
  const { refreshCount } = wiredSources();

  assert.equal(refreshCount(), 0);
});
