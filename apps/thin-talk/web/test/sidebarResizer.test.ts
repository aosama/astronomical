import assert from "node:assert/strict";
import { test } from "node:test";
import { MAX_SIDEBAR_WIDTH_PIXELS, MIN_SIDEBAR_WIDTH_PIXELS } from "../src/SessionPreferences";
import { waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";
import type { FakeSessionHost } from "./support/fakeHost";

function resizeHandle(): HTMLElement {
  const handle = document.querySelector<HTMLElement>('[data-testid="sidebar-resize-handle"]');
  if (handle == null) {
    throw new Error("the sidebar has a resize handle");
  }
  return handle;
}

function dragHandleTo(pointerX: number, button = 0): void {
  const handle = resizeHandle();
  handle.dispatchEvent(new window.PointerEvent("pointerdown", { button, clientX: 0, pointerId: 1, bubbles: true, cancelable: true }));
  handle.dispatchEvent(new window.PointerEvent("pointermove", { clientX: pointerX, pointerId: 1, bubbles: true }));
  handle.dispatchEvent(new window.PointerEvent("pointerup", { clientX: pointerX, pointerId: 1, bubbles: true }));
}

function savedSidebarWidth(host: FakeSessionHost): unknown {
  const savedPreferences = [...host.receivedRequests].reverse().find((request) => request.op === "savePrefs")?.payload;
  return (savedPreferences as { sidebarWidthPixels?: unknown } | undefined)?.sidebarWidthPixels;
}

function appliedSidebarWidth(): string {
  return document.documentElement.style.getPropertyValue("--sidebar-width");
}

test("should_resize_the_sidebar_by_dragging_its_edge_and_remember_the_width", async () => {
  const { host } = bootJourney();
  await waitUntilReady();
  dragHandleTo(320);
  await waitFor(() => savedSidebarWidth(host) === 320, "the dragged width to be saved");
  assert.equal(appliedSidebarWidth(), "320px");
}, { timeout: 30_000 });

test("should_stop_the_sidebar_at_its_narrowest_and_widest", async () => {
  const { host } = bootJourney();
  await waitUntilReady();
  dragHandleTo(10);
  await waitFor(() => savedSidebarWidth(host) === MIN_SIDEBAR_WIDTH_PIXELS, "the narrowest width");
  dragHandleTo(4000);
  await waitFor(() => savedSidebarWidth(host) === MAX_SIDEBAR_WIDTH_PIXELS, "the widest width");
  assert.equal(appliedSidebarWidth(), `${MAX_SIDEBAR_WIDTH_PIXELS}px`);
}, { timeout: 30_000 });

test("should_return_to_the_default_width_on_double_click", async () => {
  const { host } = bootJourney();
  await waitUntilReady();
  dragHandleTo(400);
  await waitFor(() => savedSidebarWidth(host) === 400, "the dragged width");
  resizeHandle().dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  await waitFor(() => savedSidebarWidth(host) === null, "the default width to be saved");
  assert.equal(appliedSidebarWidth(), "", "the stylesheet's default width applies again");
}, { timeout: 30_000 });

test("should_ignore_a_right_button_press_on_the_edge", async () => {
  const { host } = bootJourney();
  await waitUntilReady();
  dragHandleTo(400, 2);
  await new Promise((settle) => setTimeout(settle, 20));
  assert.equal(host.receivedRequests.some((request) => request.op === "savePrefs"), false);
  assert.equal(appliedSidebarWidth(), "");
}, { timeout: 30_000 });

test("should_open_at_the_width_saved_last_time", async () => {
  bootJourney(undefined, { seedPreferences: { sidebarWidthPixels: 360 } });
  await waitUntilReady();
  await waitFor(() => appliedSidebarWidth() === "360px", "the saved width to apply on boot");
}, { timeout: 30_000 });

test("should_save_nothing_for_a_click_on_the_edge_that_moves_nothing", async () => {
  const { host } = bootJourney();
  await waitUntilReady();
  const handle = resizeHandle();
  handle.dispatchEvent(new window.PointerEvent("pointerdown", { button: 0, clientX: 248, pointerId: 1, bubbles: true, cancelable: true }));
  handle.dispatchEvent(new window.PointerEvent("pointerup", { clientX: 248, pointerId: 1, bubbles: true }));
  await new Promise((settle) => setTimeout(settle, 20));
  assert.equal(host.receivedRequests.some((request) => request.op === "savePrefs"), false);
}, { timeout: 30_000 });

test("should_keep_following_the_pointer_when_the_page_republishes_mid_drag", async () => {
  bootJourney(undefined, { seedPreferences: { sidebarWidthPixels: 300 } });
  await waitUntilReady();
  await waitFor(() => appliedSidebarWidth() === "300px", "the saved width");
  const handle = resizeHandle();
  handle.dispatchEvent(new window.PointerEvent("pointerdown", { button: 0, clientX: 300, pointerId: 1, bubbles: true, cancelable: true }));
  handle.dispatchEvent(new window.PointerEvent("pointermove", { clientX: 420, pointerId: 1, bubbles: true }));
  // Sending a message makes the controller publish (with the saved 300px)
  // while the pointer is still down.
  sendUserMessage("Hark, what light through yonder window breaks?");
  await new Promise((settle) => setTimeout(settle, 30));
  assert.equal(appliedSidebarWidth(), "420px", "the dragged width is not pulled back");
  handle.dispatchEvent(new window.PointerEvent("pointerup", { clientX: 420, pointerId: 1, bubbles: true }));
}, { timeout: 30_000 });
