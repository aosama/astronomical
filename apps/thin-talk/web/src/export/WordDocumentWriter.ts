import { Document, Packer } from "docx";
import type { IBaseParagraphStyleOptions } from "docx";
import { htmlToContent } from "docx/html";
import { exportArticleHtml, SHARED_EXPORT_RULES } from "./ExportArticle";
import { EXPORT_DIAGRAM_CLASS, EXPORT_DARK_DIAGRAM_CLASS } from "./ExportBodyCleaner";
import type { ExportContent } from "./ExportContentBuilder";
import type { DiagramRasteriser } from "./DiagramRasteriser";

/**
 * Writes the exported document as a Word file.
 *
 * The `docx` library reads the same article HTML the other formats use and
 * turns it into Word's own headings, lists, tables and equations, so a
 * student can keep editing it. Two alternatives were measured and not used:
 * macOS's own Word writer (NSAttributedString) dropped every table, picture,
 * list number and code background, and hand-written Word XML would mean
 * owning the whole file format.
 */

const INK_COLOUR = "1C1B19";
/** Word's built-in headings are blue; the file keeps the app's ink colour so
 * an exported chat reads like the window it came from. */
const HEADING_STYLE: IBaseParagraphStyleOptions = { run: { color: INK_COLOUR, bold: true }, paragraph: { spacing: { before: 240, after: 80 } } };

/** The ground a diagram drawn in the dark appearance keeps in Word: its lines
 * are light, and on Word's white page they would vanish. */
const DARK_DIAGRAM_GROUND = "#1f1f1d";
const LIGHT_DIAGRAM_GROUND = "#ffffff";

/** Word cannot show an SVG without a PNG beside it, and older versions cannot
 * show one at all, so each diagram becomes a picture before conversion. A
 * diagram that cannot be drawn is left out rather than shown as nothing. */
async function replaceDiagramsWithPictures(article: HTMLElement, rasteriseDiagram: DiagramRasteriser): Promise<void> {
  for (const diagramFigure of [...article.querySelectorAll(`figure.${EXPORT_DIAGRAM_CLASS}`)]) {
    const diagram = diagramFigure.querySelector<SVGSVGElement>("svg");
    const groundColour = diagramFigure.classList.contains(EXPORT_DARK_DIAGRAM_CLASS) ? DARK_DIAGRAM_GROUND : LIGHT_DIAGRAM_GROUND;
    const pictureData = diagram !== null ? await rasteriseDiagram(diagram, groundColour) : null;
    if (pictureData === null) {
      diagramFigure.remove();
      continue;
    }
    const picture = document.createElement("img");
    picture.setAttribute("src", pictureData);
    picture.setAttribute("alt", "Diagram");
    diagramFigure.replaceChildren(picture);
  }
}

export async function writeWordDocument(content: ExportContent, rasteriseDiagram: DiagramRasteriser): Promise<string> {
  const article = document.createElement("div");
  article.innerHTML = exportArticleHtml(content);
  await replaceDiagramsWithPictures(article, rasteriseDiagram);
  const wordContent = htmlToContent(article.innerHTML, { stylesheet: SHARED_EXPORT_RULES });
  const wordDocument = new Document({
    title: content.title,
    // Nothing about the person or the Mac goes into the file's properties: a
    // document handed in is read by others.
    creator: "Astronomical",
    description: content.aiNote ?? "",
    styles: {
      default: {
        document: { run: { color: INK_COLOUR, size: 22 } },
        heading1: HEADING_STYLE,
        heading2: HEADING_STYLE,
        heading3: HEADING_STYLE,
        heading4: HEADING_STYLE,
      },
    },
    sections: [{ children: [...wordContent] }],
  });
  return Packer.toBase64String(wordDocument);
}
