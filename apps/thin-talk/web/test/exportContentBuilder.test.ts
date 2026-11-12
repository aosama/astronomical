import assert from "node:assert/strict";
import { assertNoElement } from "./support/dom";
import { test } from "node:test";
import { aiNoteText, EXPORT_PICTURE_CLASS, ExportContentBuilder, messagesInScope, suggestedExportName } from "../src/export/ExportContentBuilder";
import type { ExportableMessage, ExportPictureData } from "../src/export/ExportEnvironment";
import { defaultExportOptions } from "../src/export/ExportOptions";
import type { ExportOptions } from "../src/export/ExportOptions";

const PICTURE_NAME = "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.png";

function exportableMessage(messageID: string, isQuestion: boolean, html: string, generatedImageNames: string[] = []): ExportableMessage {
  const renderedBody = document.createElement("div");
  renderedBody.innerHTML = html;
  return {
    messageID,
    isQuestion,
    renderedBody,
    generatedImageNames,
    generatedImageDescriptions: new Map(generatedImageNames.map((imageName) => [imageName, "Generated image: a rocket"])),
  };
}

/** A conversation of two turns; the second answer drew a picture. Bodies are
 * rebuilt on every read, as the transcript clones them afresh. */
function mitosisConversation(): ExportableMessage[] {
  return [
    exportableMessage("q1", true, "<p>What is mitosis?</p>"),
    exportableMessage("a1", false, "<p>Cell division.</p>"),
    exportableMessage("q2", true, "<p>Draw a cell</p>"),
    exportableMessage("a2", false, "<p>Here it is.</p>", [PICTURE_NAME]),
  ];
}

function builderFor(
  assistantName: string | null,
  pictureAnswers: Map<string, ExportPictureData | null>,
  title = "Mitosis explained",
  conversation: () => ExportableMessage[] = mitosisConversation,
  paperDrawing: string | null = null,
): { builder: ExportContentBuilder; pictureLoads: string[]; redrawnSources: string[] } {
  const redrawnSources: string[] = [];
  const pictureLoads: string[] = [];
  const builder = new ExportContentBuilder(
    {
      conversationTitle: () => title,
      assistantName: () => assistantName,
      exportableMessages: conversation,
      drawDiagramForPaper: async (diagramSource) => {
        redrawnSources.push(diagramSource);
        return paperDrawing;
      },
      loadPicture: async (imageName) => {
        pictureLoads.push(imageName);
        return pictureAnswers.get(imageName) ?? null;
      },
    },
    false,
    () => new Date(Date.UTC(2026, 10, 10, 12)),
  );
  return { builder, pictureLoads, redrawnSources };
}

const PICTURE_ANSWERS = new Map([[PICTURE_NAME, { base64: "iVBORw0KGgo=", mimeType: "image/png" }]]);

function entryTexts(entries: readonly { body: HTMLElement }[]): string[] {
  return entries.map((entry) => (entry.body.textContent ?? "").trim());
}

test("should_export_one_answer_with_the_question_right_before_it", () => {
  const scoped = messagesInScope(mitosisConversation(), { kind: "oneAnswer", messageID: "a2" });

  assert.deepEqual(scoped.map((message) => message.messageID), ["q2", "a2"]);
});

test("should_export_an_answer_alone_when_no_question_comes_before_it", () => {
  const answerFirst = mitosisConversation().slice(1);

  assert.deepEqual(messagesInScope(answerFirst, { kind: "oneAnswer", messageID: "a1" }).map((message) => message.messageID), ["a1"]);
});

test("should_keep_picked_messages_in_conversation_order_and_ignore_unknown_ids", () => {
  const scoped = messagesInScope(mitosisConversation(), { kind: "pickedMessages", messageIDs: ["a2", "gone", "q1"] });

  assert.deepEqual(scoped.map((message) => message.messageID), ["q1", "a2"]);
  assert.deepEqual(messagesInScope(mitosisConversation(), { kind: "oneAnswer", messageID: "gone" }), []);
});

test("should_build_the_whole_chat_with_the_title_date_and_a_note_naming_astro", async () => {
  const { builder } = builderFor("Astro", PICTURE_ANSWERS);

  const content = await builder.build(defaultExportOptions({ kind: "wholeChat" }));

  assert.equal(content.title, "Mitosis explained");
  assert.match(content.dateText, /2026/);
  assert.deepEqual(entryTexts(content.entries), ["What is mitosis?", "Cell division.", "Draw a cell", "Here it is."]);
  assert.deepEqual(content.entries.map((entry) => entry.isQuestion), [true, false, true, false]);
  assert.equal(content.aiNote, "Written with help from Astro, an AI on this Mac.");
});

test("should_leave_out_questions_pictures_and_the_note_when_asked", async () => {
  const { builder, pictureLoads } = builderFor("Astro", PICTURE_ANSWERS);
  const options: ExportOptions = { ...defaultExportOptions({ kind: "wholeChat" }), includesQuestions: false, includesPictures: false, includesAiNote: false };

  const content = await builder.build(options);

  assert.deepEqual(entryTexts(content.entries), ["Cell division.", "Here it is."]);
  assertNoElement(content.entries[1]?.body, "img");
  assert.deepEqual(pictureLoads, [], "no picture is fetched when none is wanted");
  assert.equal(content.aiNote, null);
});

test("should_put_each_picture_above_its_answer_and_fetch_it_only_once", async () => {
  const { builder, pictureLoads } = builderFor(null, PICTURE_ANSWERS);
  const options = defaultExportOptions({ kind: "wholeChat" });

  await builder.build(options);
  const content = await builder.build(options);

  const pictureFigure = content.entries[3]?.body.firstElementChild;
  assert.equal(pictureFigure?.className, EXPORT_PICTURE_CLASS);
  assert.equal(pictureFigure?.querySelector("img")?.getAttribute("src"), "data:image/png;base64,iVBORw0KGgo=");
  assert.equal(pictureFigure?.querySelector("img")?.getAttribute("alt"), "Generated image: a rocket");
  assert.deepEqual(pictureLoads, [PICTURE_NAME], "a rebuild for the preview reuses the picture");
});

test("should_export_the_rest_of_an_answer_whose_picture_is_gone", async () => {
  const { builder } = builderFor(null, new Map());

  const content = await builder.build(defaultExportOptions({ kind: "oneAnswer", messageID: "a2" }));

  assert.deepEqual(entryTexts(content.entries), ["Here it is."]);
  assertNoElement(content.entries[0]?.body, "img");
});

test("should_start_a_single_answer_without_its_question_and_name_no_assistant_in_the_plain_chat", async () => {
  const { builder } = builderFor(null, PICTURE_ANSWERS, "   ");

  const content = await builder.build(defaultExportOptions({ kind: "oneAnswer", messageID: "a1" }));

  assert.deepEqual(entryTexts(content.entries), ["Cell division."]);
  assert.equal(content.title, "Conversation");
  assert.equal(content.aiNote, aiNoteText(null));
  assert.equal(aiNoteText(null), "Written with help from an AI on this Mac.");
});

test("should_offer_a_file_name_any_computer_can_save", () => {
  assert.equal(suggestedExportName("Mitosis: phases / steps?"), "Mitosis phases steps");
  assert.equal(suggestedExportName("  \n "), "Conversation");
  assert.equal(suggestedExportName("a".repeat(120)).length, 80);
  assert.equal(suggestedExportName("Café 日本語"), "Café 日本語");
});

const DARK_SCREEN_DIAGRAM = `<div class="code-block code-block--diagram"><figure class="diagram" data-diagram-source="flowchart LR; A-->B"><svg id="screen"></svg></figure></div>`;

function diagramConversation(): ExportableMessage[] {
  return [exportableMessage("a1", false, DARK_SCREEN_DIAGRAM)];
}

// A diagram drawn in the dark appearance printed as a black slab.
test("should_draw_a_diagram_again_for_paper_once_and_give_it_no_dark_ground", async () => {
  const { builder, redrawnSources } = builderFor(null, new Map(), "Diagram", diagramConversation, `<svg id="paper"></svg>`);
  const darkBuilderOptions = defaultExportOptions({ kind: "wholeChat" });

  await builder.build(darkBuilderOptions);
  const content = await builder.build(darkBuilderOptions);

  const diagramFigure = content.entries[0]?.body.querySelector("figure");
  assert.equal(diagramFigure?.querySelector("svg")?.id, "paper");
  assert.equal(diagramFigure?.className, "export-diagram");
  assert.deepEqual(redrawnSources, ["flowchart LR; A-->B"], "a preview rebuild reuses the drawing");
});

test("should_keep_the_screen_drawing_when_it_cannot_be_drawn_again", async () => {
  const { builder } = builderFor(null, new Map(), "Diagram", diagramConversation, null);

  const content = await builder.build(defaultExportOptions({ kind: "wholeChat" }));

  assert.equal(content.entries[0]?.body.querySelector("svg")?.id, "screen");
});

function headedConversation(): ExportableMessage[] {
  return [
    exportableMessage("q1", true, "<p>What is mitosis?</p>"),
    exportableMessage("a1", false, "<p>Cell division.</p>"),
    exportableMessage("q2", true, "<p>Explain the water cycle</p>"),
    exportableMessage("a2", false, "<h1>The Water Cycle</h1><p>Water moves in a loop.</p><h2>Stages</h2>"),
  ];
}

// A chat is named after its first question, so a later answer exported alone
// was titled "What is mitosis?" above a page about the water cycle.
test("should_title_a_single_answer_with_its_own_opening_heading", async () => {
  const { builder } = builderFor("Astro", new Map(), "What is mitosis?", headedConversation);

  const content = await builder.build(defaultExportOptions({ kind: "oneAnswer", messageID: "a2" }));

  assert.equal(content.title, "The Water Cycle");
  assertNoElement(content.entries[0]?.body, "h1", "the heading is not repeated under the title");
  assert.equal(content.entries[0]?.body.querySelector("h2")?.textContent, "Stages", "later headings stay");
  assert.match(content.entries[0]?.body.textContent ?? "", /Water moves in a loop\./);
});

test("should_keep_the_chats_name_for_a_single_answer_that_opens_with_prose", async () => {
  const { builder } = builderFor("Astro", new Map(), "What is mitosis?", headedConversation);

  const content = await builder.build(defaultExportOptions({ kind: "oneAnswer", messageID: "a1" }));

  assert.equal(content.title, "What is mitosis?");
  assert.match(content.entries[0]?.body.textContent ?? "", /Cell division\./);
});

test("should_keep_the_chats_name_and_every_heading_when_exporting_more_than_one_answer", async () => {
  const { builder } = builderFor("Astro", new Map(), "What is mitosis?", headedConversation);

  const wholeChat = await builder.build(defaultExportOptions({ kind: "wholeChat" }));
  const pickedTwoAnswers = await builder.build(defaultExportOptions({ kind: "pickedMessages", messageIDs: ["a1", "a2"] }));

  assert.equal(wholeChat.title, "What is mitosis?");
  assert.equal(wholeChat.entries[3]?.body.querySelector("h1")?.textContent, "The Water Cycle");
  assert.equal(pickedTwoAnswers.title, "What is mitosis?", "two picked answers are part of a chat");
  assert.equal(pickedTwoAnswers.entries.at(-1)?.body.querySelector("h1")?.textContent, "The Water Cycle");
});

// Picking only the water-cycle answer from a chat named after black holes
// printed "Explain black holes" above the report.
test("should_title_one_picked_answer_with_its_own_opening_heading", async () => {
  const { builder } = builderFor("Astro", new Map(), "What is mitosis?", headedConversation);

  const pickedAlone = await builder.build(defaultExportOptions({ kind: "pickedMessages", messageIDs: ["a2"] }));
  const pickedWithItsQuestion = await builder.build({ ...defaultExportOptions({ kind: "pickedMessages", messageIDs: ["q2", "a2"] } as const), includesQuestions: true });

  assert.equal(pickedAlone.title, "The Water Cycle");
  assertNoElement(pickedAlone.entries[0]?.body, "h1", "the heading is not repeated under the title");
  assert.equal(pickedWithItsQuestion.title, "The Water Cycle");
  assert.deepEqual(pickedWithItsQuestion.entries.map((entry) => entry.isQuestion), [true, false]);
});

test("should_keep_the_chats_name_when_only_questions_are_picked_or_the_picked_answer_opens_with_prose", async () => {
  const { builder } = builderFor("Astro", new Map(), "What is mitosis?", headedConversation);

  const onlyAQuestion = await builder.build({ ...defaultExportOptions({ kind: "pickedMessages", messageIDs: ["q2"] } as const), includesQuestions: true });
  const proseAnswer = await builder.build(defaultExportOptions({ kind: "pickedMessages", messageIDs: ["a1"] }));

  assert.equal(onlyAQuestion.title, "What is mitosis?");
  assert.equal(proseAnswer.title, "What is mitosis?");
});

test("should_keep_a_whole_chats_name_even_when_it_holds_one_headed_answer", async () => {
  const oneExchange = () => [exportableMessage("q1", true, "Explain the water cycle"), exportableMessage("a1", false, "<h1>The Water Cycle</h1><p>Water moves in a loop.</p>")];
  const { builder } = builderFor("Astro", new Map(), "Explain the water cycle", oneExchange);

  const wholeChat = await builder.build(defaultExportOptions({ kind: "wholeChat" }));

  assert.equal(wholeChat.title, "Explain the water cycle");
  assert.equal(wholeChat.entries.at(-1)?.body.querySelector("h1")?.textContent, "The Water Cycle");
});

test("should_take_the_heading_from_the_answer_not_from_the_question_before_it", async () => {
  const { builder } = builderFor("Astro", new Map(), "What is mitosis?", headedConversation);
  const withQuestion = { ...defaultExportOptions({ kind: "oneAnswer", messageID: "a2" } as const), includesQuestions: true };

  const content = await builder.build(withQuestion);

  assert.equal(content.title, "The Water Cycle");
  assert.deepEqual(content.entries.map((entry) => entry.isQuestion), [true, false]);
  assert.match(content.entries[0]?.body.textContent ?? "", /Explain the water cycle/);
});
