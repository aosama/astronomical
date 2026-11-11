import assert from "node:assert/strict";
import { test } from "node:test";
import { strFromU8, unzipSync } from "fflate";
import { writeWordDocument } from "../src/export/WordDocumentWriter";
import type { ExportContent } from "../src/export/ExportContentBuilder";
import type { DiagramRasteriser } from "../src/export/DiagramRasteriser";

// A one-pixel PNG, so Word has a real picture to embed.
const ONE_PIXEL_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

function answerContent(answerHtml: string): ExportContent {
  const answerBody = document.createElement("div");
  answerBody.innerHTML = answerHtml;
  return { title: "Mitosis explained", dateText: "November 10, 2026", entries: [{ isQuestion: false, body: answerBody }], aiNote: "Written with help from Astro, an AI on this Mac." };
}

async function wordFileParts(content: ExportContent, rasteriseDiagram: DiagramRasteriser): Promise<Record<string, string>> {
  const base64Document = await writeWordDocument(content, rasteriseDiagram);
  const wordFile = unzipSync(new Uint8Array(Buffer.from(base64Document, "base64")));
  return Object.fromEntries(Object.entries(wordFile).map(([partName, partBytes]) => [partName, partName.endsWith(".xml") ? strFromU8(partBytes) : `${partBytes.length} bytes`]));
}

// A different one-pixel PNG: Word keeps identical pictures as one file.
const ONE_VIOLET_PIXEL_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGPIsj8CAAKEAW76LM8WAAAAAElFTkSuQmCC";

const NEVER_DRAWS: DiagramRasteriser = async () => null;

test("should_write_a_word_file_with_real_headings_lists_tables_code_and_equations", async () => {
  const parts = await wordFileParts(
    answerContent(
      `<h2>The phases</h2><ol><li>Prophase</li><li>Metaphase</li></ol>` +
        `<table><tr><th>Phase</th><th>What happens</th></tr><tr><td>Prophase</td><td>Chromosomes condense</td></tr></table>` +
        `<pre><code>print("phase")</code></pre>` +
        `<p><math><mfrac><mi>a</mi><mi>b</mi></mfrac></math></p>`,
    ),
    NEVER_DRAWS,
  );

  const documentXml = parts["word/document.xml"] ?? "";
  assert.match(documentXml, /Mitosis explained/);
  assert.match(documentXml, /<w:tbl>/, "the table stays a table Word can edit");
  assert.match(documentXml, /<w:numPr>/, "the list keeps its numbers");
  assert.match(documentXml, /<m:oMath>/, "the maths is one of Word's own equations");
  assert.match(documentXml, /<m:f>/);
  assert.match(documentXml, /print\(&quot;phase&quot;\)|print\("phase"\)/);
  assert.match(documentXml, /Written with help from Astro/);
});

test("should_keep_headings_in_the_apps_ink_instead_of_words_blue", async () => {
  const parts = await wordFileParts(answerContent("<h2>The phases</h2>"), NEVER_DRAWS);

  const stylesXml = parts["word/styles.xml"] ?? "";
  const heading2Style = stylesXml.slice(stylesXml.indexOf('w:styleId="Heading2"'));
  assert.match(heading2Style.slice(0, 800), /<w:color w:val="1C1B19"\/>/);
});

test("should_embed_a_picture_and_a_drawn_diagram_as_pictures", async () => {
  const parts = await wordFileParts(
    answerContent(`<figure class="export-picture"><img src="${ONE_PIXEL_PNG}" alt="A rocket"></figure><figure class="export-diagram"><svg viewBox="0 0 10 10"></svg></figure>`),
    async () => ONE_VIOLET_PIXEL_PNG,
  );

  const embeddedMedia = Object.keys(parts).filter((partName) => partName.startsWith("word/media/"));
  assert.equal(embeddedMedia.length, 2);
  assert.equal((parts["word/document.xml"] ?? "").match(/<w:drawing>/g)?.length, 2);
});

test("should_draw_a_dark_diagram_on_its_dark_ground_and_a_light_one_on_white", async () => {
  const groundColours: string[] = [];
  await wordFileParts(
    answerContent(
      `<figure class="export-diagram export-diagram--dark"><svg viewBox="0 0 10 10"></svg></figure>` +
        `<figure class="export-diagram"><svg viewBox="0 0 10 10"></svg></figure>`,
    ),
    async (_diagram, groundColour) => {
      groundColours.push(groundColour);
      return ONE_PIXEL_PNG;
    },
  );

  assert.deepEqual(groundColours, ["#1f1f1d", "#ffffff"], "light lines would vanish on Word's white page");
});

test("should_leave_out_a_diagram_that_could_not_be_drawn", async () => {
  const parts = await wordFileParts(answerContent(`<p>Before</p><figure class="export-diagram"><svg viewBox="0 0 10 10"></svg></figure><p>After</p>`), NEVER_DRAWS);

  assert.equal(Object.keys(parts).filter((partName) => partName.startsWith("word/media/")).length, 0);
  assert.match(parts["word/document.xml"] ?? "", /Before[\s\S]*After/);
});

test("should_keep_nothing_about_the_person_in_the_files_properties", async () => {
  const parts = await wordFileParts(answerContent("<p>Hi</p>"), NEVER_DRAWS);

  const coreProperties = parts["docProps/core.xml"] ?? "";
  assert.match(coreProperties, /<dc:creator>Astronomical<\/dc:creator>/);
  assert.match(coreProperties, /<dc:title>Mitosis explained<\/dc:title>/);
});
