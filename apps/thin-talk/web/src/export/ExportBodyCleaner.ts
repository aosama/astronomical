/**
 * Turns a rendered answer body into markup that can leave the app.
 *
 * The transcript's markup leans on the page around it: the app's stylesheet
 * and fonts, KaTeX's stylesheet, the private image scheme, and buttons that
 * only work inside the window. An exported file has none of those, so each
 * dependent piece is replaced by its self-contained equivalent, and every
 * attribute the page added for its own use is dropped. What is left is plain
 * HTML that a browser, the PDF printer and the Word writer all read the same
 * way, which is what lets one document feed all three formats.
 */

/** The classes of a checklist box as the transcript draws it; the values are
 * the Renderer's, repeated because this script shares no code with the page. */
const TASK_BOX_CLASS = "task-box";
const DONE_TASK_BOX_CLASS = "task-box--done";

/** The class of an exported diagram's figure. */
export const EXPORT_DIAGRAM_CLASS = "export-diagram";
/** Added when the diagram was drawn in the dark appearance, so the file can
 * put the same dark ground behind its light lines. */
export const EXPORT_DARK_DIAGRAM_CLASS = "export-diagram--dark";
/** Set on a transcript diagram figure once it has been drawn again for
 * paper, so it is not also given a dark ground. */
export const PAPER_DRAWN_ATTRIBUTE = "data-paper-drawn";
/** A checklist item, which shows its own box instead of a bullet. */
export const EXPORT_TASK_CLASS = "export-task";

/** The attributes an exported element may keep, by tag name. Everything else
 * (classes, ids, test ids, data attributes) served the page, not the reader. */
const KEPT_ATTRIBUTES_BY_TAG: Readonly<Record<string, readonly string[]>> = {
  A: ["href"],
  IMG: ["src", "alt"],
  TD: ["colspan", "rowspan"],
  TH: ["colspan", "rowspan"],
  OL: ["start"],
};

const OPENABLE_LINK = /^(?:https?:|mailto:)/i;
const EMBEDDED_IMAGE = /^data:image\//i;

function replaceElement(staleElement: Element, replacement: Node): void {
  staleElement.parentNode?.replaceChild(replacement, staleElement);
}

/** A code block becomes a plain `pre`: its Copy button and language label are
 * page chrome, and its syntax colours come from a stylesheet the file does
 * not carry. A drawn diagram keeps its drawing instead of its source. */
function replaceCodeBlocks(body: HTMLElement, isDarkAppearance: boolean): void {
  body.querySelectorAll(".code-block").forEach((codeBlockWrapper) => {
    const drawnDiagram = codeBlockWrapper.querySelector("figure.diagram svg");
    if (drawnDiagram !== null) {
      const isPaperDrawn = codeBlockWrapper.querySelector(`figure.diagram[${PAPER_DRAWN_ATTRIBUTE}]`) !== null;
      const diagramFigure = document.createElement("figure");
      diagramFigure.className = isDarkAppearance && !isPaperDrawn ? `${EXPORT_DIAGRAM_CLASS} ${EXPORT_DARK_DIAGRAM_CLASS}` : EXPORT_DIAGRAM_CLASS;
      diagramFigure.appendChild(drawnDiagram);
      replaceElement(codeBlockWrapper, diagramFigure);
      return;
    }
    const plainBlock = document.createElement("pre");
    const plainCode = document.createElement("code");
    plainCode.textContent = codeBlockWrapper.querySelector("pre > code")?.textContent ?? "";
    plainBlock.appendChild(plainCode);
    replaceElement(codeBlockWrapper, plainBlock);
  });
}

/** KaTeX draws maths twice: as positioned spans that need its stylesheet and
 * fonts, and as MathML for screen readers. Only the MathML is kept, because
 * browsers draw it without any stylesheet and Word reads it as an equation.
 * Maths with no MathML falls back to the LaTeX it was written in. */
function replaceMaths(body: HTMLElement): void {
  body.querySelectorAll(".katex").forEach((katexSpan) => {
    const displayWrapper = katexSpan.closest(".katex-display");
    const staleElement = displayWrapper ?? katexSpan;
    const mathElement = katexSpan.querySelector("math");
    if (mathElement === null) {
      const latexSource = katexSpan.querySelector("annotation")?.textContent ?? katexSpan.textContent ?? "";
      replaceElement(staleElement, document.createTextNode(latexSource));
      return;
    }
    if (displayWrapper !== null) {
      mathElement.setAttribute("display", "block");
    }
    replaceElement(staleElement, mathElement);
  });
}

/** An image the file cannot carry (anything but inline image data) is replaced
 * by its description, so the reader sees that a picture was there instead of
 * a broken-image box. */
function replaceUnembeddableImages(body: HTMLElement): void {
  body.querySelectorAll("img").forEach((image) => {
    if (EMBEDDED_IMAGE.test(image.getAttribute("src") ?? "")) {
      return;
    }
    const description = (image.getAttribute("alt") ?? "").trim();
    replaceElement(image, document.createTextNode(description.length > 0 ? `[Picture: ${description}]` : ""));
  });
}

/** A checklist box becomes a character. The transcript draws its boxes with
 * a stylesheet the file does not carry, while ☑ and ☐ are text everywhere,
 * Word included. The item drops its bullet, as the box is its marker. */
function replaceTaskBoxes(body: HTMLElement): void {
  body.querySelectorAll(`.${TASK_BOX_CLASS}`).forEach((taskBox) => {
    taskBox.closest("li")?.classList.add(EXPORT_TASK_CLASS);
    replaceElement(taskBox, document.createTextNode(taskBox.classList.contains(DONE_TASK_BOX_CLASS) ? "☑ " : "☐ "));
  });
}

/** A folded section is opened flat: a printed page cannot unfold anything. */
function flattenDisclosures(body: HTMLElement): void {
  body.querySelectorAll("details").forEach((disclosure) => {
    const flatSection = document.createElement("div");
    const summary = disclosure.querySelector(":scope > summary");
    if (summary !== null) {
      const summaryHeading = document.createElement("p");
      const summaryEmphasis = document.createElement("strong");
      summaryEmphasis.textContent = summary.textContent ?? "";
      summaryHeading.appendChild(summaryEmphasis);
      flatSection.appendChild(summaryHeading);
      summary.remove();
    }
    flatSection.append(...disclosure.childNodes);
    replaceElement(disclosure, flatSection);
  });
}

function stripPageAttributes(body: HTMLElement): void {
  body.querySelectorAll("*").forEach((element) => {
    // Maths and drawings are whole documents of their own; their attributes
    // are their content.
    if (element.closest("math, svg") !== null) {
      return;
    }
    const isExportFigure = element.classList.contains(EXPORT_DIAGRAM_CLASS) || element.classList.contains(EXPORT_TASK_CLASS);
    const keptAttributeNames = KEPT_ATTRIBUTES_BY_TAG[element.tagName] ?? [];
    for (const attributeName of element.getAttributeNames()) {
      const isKept = keptAttributeNames.includes(attributeName) || (isExportFigure && attributeName === "class");
      if (!isKept) {
        element.removeAttribute(attributeName);
      }
    }
    if (element.tagName === "A" && !OPENABLE_LINK.test(element.getAttribute("href") ?? "")) {
      element.removeAttribute("href");
    }
  });
}

/** Cleans a detached copy of a rendered body in place and returns it. */
export function cleanExportBody(renderedBody: HTMLElement, isDarkAppearance: boolean): HTMLElement {
  replaceCodeBlocks(renderedBody, isDarkAppearance);
  replaceMaths(renderedBody);
  replaceUnembeddableImages(renderedBody);
  replaceTaskBoxes(renderedBody);
  flattenDisclosures(renderedBody);
  stripPageAttributes(renderedBody);
  for (const attributeName of renderedBody.getAttributeNames()) {
    renderedBody.removeAttribute(attributeName);
  }
  return renderedBody;
}
