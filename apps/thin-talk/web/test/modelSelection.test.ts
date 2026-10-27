import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";

/**
 * Model selection behaviour.
 *
 * The daemon advertises a model catalog; the composer renders it as a menu and
 * the chosen id rides both the outgoing chat request and the persisted
 * preferences, so a relaunch resumes on the model the user picked.
 */

const TWO_CHAT_MODELS = [
  {
    id: "model-alpha",
    input_modalities: ["text"],
    supported_endpoints: ["/v1/chat/completions"],
  },
  {
    id: "model-beta",
    input_modalities: ["text"],
    supported_endpoints: ["/v1/chat/completions"],
  },
];

function modelPill(): HTMLButtonElement {
  const pill = document.querySelector<HTMLButtonElement>('[data-testid="composer-model"]');
  if (!pill) {
    throw new Error("the composer model pill exists");
  }
  return pill;
}

function modelChoice(modelID: string): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>(`[data-testid="composer-model-${modelID}"]`);
}

function modelLabel(): string | null {
  const label = document.querySelector<HTMLElement>('[data-testid="composer-model-label"]');
  return label != null ? label.textContent : null;
}

function chooseModel(modelID: string): void {
  modelPill().dispatchEvent(new window.Event("click", { bubbles: true }));
  const choice = modelChoice(modelID);
  if (!choice) {
    throw new Error(`the model choice ${modelID} exists`);
  }
  choice.dispatchEvent(new window.Event("click", { bubbles: true }));
}

async function waitForSelectedModel(modelID: string): Promise<void> {
  await waitFor(() => modelChoice(modelID)?.getAttribute("aria-checked") === "true",
    `the model ${modelID} to be checked in the menu`);
}

test("should_list_every_catalog_model_in_the_composer_model_menu", async () => {
  bootJourney(undefined, { models: TWO_CHAT_MODELS });
  await waitUntilReady();
  await waitFor(
    () => modelChoice("model-alpha") !== null && modelChoice("model-beta") !== null,
    "the model menu to list both models",
  );
  // The first catalog model is checked until the user chooses otherwise.
  assert.equal(modelChoice("model-alpha")?.getAttribute("aria-checked"), "true");
  assert.equal(modelChoice("model-beta")?.getAttribute("aria-checked"), "false");
}, { timeout: 30_000 });

test("should_send_chat_requests_to_the_model_chosen_from_the_menu", async () => {
  const { daemon } = bootJourney(undefined, { models: TWO_CHAT_MODELS });
  await waitUntilReady();

  // Opening the pill shows the menu; choosing a model checks it and updates
  // the pill label.
  modelPill().dispatchEvent(new window.Event("click", { bubbles: true }));
  await waitFor(
    () => modelPill().getAttribute("aria-expanded") === "true",
    "the model menu to open",
  );
  chooseModel("model-beta");
  await waitForSelectedModel("model-beta");
  assert.equal(modelLabel(), "model-beta");

  sendUserMessage("Compare the fates of Romeo and Juliet.");
  await waitFor(() => daemon.chatRequestBody() !== null, "the chat request to leave the composer");
  assert.equal(daemon.chatRequestBody()?.["model"], "model-beta");
}, { timeout: 30_000 });

test("should_restore_the_chosen_model_after_a_relaunch", async () => {
  const first = bootJourney(undefined, { models: TWO_CHAT_MODELS });
  await waitUntilReady();
  chooseModel("model-beta");
  await waitForSelectedModel("model-beta");
  await waitFor(
    () => first.host.receivedRequests.some((request) => request.op === "savePrefs"),
    "the model choice to be persisted",
  );

  // Second life: a fresh page, same host files, so the saved preference returns.
  const second = bootJourney(first, { models: TWO_CHAT_MODELS });
  await waitUntilReady();
  assert.equal(modelLabel(), "model-beta");
  const requestsBeforeSecondSend: number = second.daemon.requests.length;
  sendUserMessage("What does the chorus foretell in the prologue?");
  await waitFor(
    () => second.daemon.requests
      .slice(requestsBeforeSecondSend)
      .some((request) => request.url.endsWith("/v1/chat/completions")),
    "the relaunched chat request to leave the composer",
  );
  const relaunchedChat = second.daemon.requests
    .slice(requestsBeforeSecondSend)
    .find((request) => request.url.endsWith("/v1/chat/completions"));
  assert.ok(relaunchedChat && relaunchedChat.body, "the relaunched chat request carried a body");
  const relaunchedBody = JSON.parse(relaunchedChat.body) as Record<string, unknown>;
  assert.equal(relaunchedBody["model"], "model-beta");
}, { timeout: 30_000 });

test("should_treat_choosing_the_selected_model_again_as_a_no_op", async () => {
  const { host } = bootJourney(undefined, { models: TWO_CHAT_MODELS });
  await waitUntilReady();
  chooseModel("model-beta");
  await waitForSelectedModel("model-beta");
  await waitFor(
    () => host.receivedRequests.some((request) => request.op === "savePrefs"),
    "the first model choice to be persisted",
  );
  const savePrefsCountAfterFirstChoice = host.receivedRequests.filter(
    (request) => request.op === "savePrefs",
  ).length;

  // The same model chosen again must not re-persist or disturb the label.
  chooseModel("model-beta");
  await new Promise((resolve) => setTimeout(resolve, 64));
  assert.equal(modelLabel(), "model-beta");
  assert.equal(
    host.receivedRequests.filter((request) => request.op === "savePrefs").length,
    savePrefsCountAfterFirstChoice,
    "no duplicate savePrefs for an unchanged selection",
  );
}, { timeout: 30_000 });

test("should_fall_back_to_the_first_model_when_the_stored_preference_is_unknown", async () => {
  const { daemon } = bootJourney(undefined, {
    models: TWO_CHAT_MODELS,
    seedPreferences: { modelID: "ghost-model" },
  });
  await waitUntilReady();

  // The stored id is not in the catalog, so the first catalog model takes over
  // and the unknown id never reaches the daemon.
  assert.equal(modelLabel(), "model-alpha");
  sendUserMessage("Warn the star-crossed lovers about their fate.");
  await waitFor(() => daemon.chatRequestBody() !== null, "the chat request to leave the composer");
  assert.equal(daemon.chatRequestBody()?.["model"], "model-alpha");
}, { timeout: 30_000 });
