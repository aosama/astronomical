import { EXPORT_FORMAT_OFFERS, exportFormatOffer } from "./ExportFormat";
import type { ExportFormat } from "./ExportFormat";
import type { ExportOptions, ExportScope } from "./ExportOptions";
import { PREVIEW_FRAME_CLASS, PREVIEW_SCALE, WHOLE_BLOCK_SELECTOR } from "./ExportArticle";
import { wholeBlocksIn } from "./PageEstimate";
import type { WholeBlock } from "./PageEstimate";

/** The scope choices the sheet can show; "oneAnswer" only when it was opened
 * from an answer. */
export type ExportScopeKind = ExportScope["kind"];

/** The three switches under "Include". */
export type ExportToggle = "includesQuestions" | "includesPictures" | "includesAiNote";

/** What the sheet draws besides the options themselves. */
export interface ExportSheetFacts {
  readonly canExportOneAnswer: boolean;
  readonly hasQuestions: boolean;
  readonly hasPictures: boolean;
  readonly pickedMessageCount: number;
}

export interface ExportSheetCallbacks {
  chooseScope(scopeKind: ExportScopeKind): void;
  chooseFormat(format: ExportFormat): void;
  toggle(toggle: ExportToggle, isOn: boolean): void;
  save(): void;
  cancel(): void;
}

const SCOPE_LABELS: Readonly<Record<ExportScopeKind, string>> = {
  wholeChat: "Whole chat",
  oneAnswer: "This answer",
  pickedMessages: "Pick messages…",
};

const TOGGLE_LABELS: Readonly<Record<ExportToggle, string>> = {
  includesQuestions: "My questions",
  includesPictures: "Pictures",
  includesAiNote: "A line saying an AI helped",
};

function makeElement<K extends keyof HTMLElementTagNameMap>(tagName: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const createdElement = document.createElement(tagName);
  createdElement.className = className;
  if (text !== undefined) {
    createdElement.textContent = text;
  }
  return createdElement;
}

/** A button with an id equal to its test id, as every Thin Talk control has. */
function makeButton(testID: string, className: string, text: string): HTMLButtonElement {
  const createdButton = makeElement("button", className, text);
  createdButton.type = "button";
  createdButton.id = testID;
  createdButton.dataset.testid = testID;
  return createdButton;
}

const NEXT_RADIO_KEYS = ["ArrowRight", "ArrowDown"];
const PREVIOUS_RADIO_KEYS = ["ArrowLeft", "ArrowUp"];

/**
 * Makes a group of radio buttons one stop for the Tab key, moved through
 * with the arrow keys, as a radio group is everywhere else on the Mac.
 *
 * The arrows move the focus without choosing: one choice here ("Pick
 * messages…") leaves the sheet, and arrowing past it must not. Space or
 * Return chooses, as on any button.
 */
function enableArrowKeys(radioGroup: HTMLElement): void {
  radioGroup.addEventListener("keydown", (event) => {
    const step = NEXT_RADIO_KEYS.includes(event.key) ? 1 : PREVIOUS_RADIO_KEYS.includes(event.key) ? -1 : 0;
    const shownRadios = [...radioGroup.querySelectorAll<HTMLElement>('[role="radio"]')].filter((radio) => !radio.hidden);
    const focusedPosition = shownRadios.indexOf(event.target as HTMLElement);
    if (step === 0 || focusedPosition < 0) {
      return;
    }
    event.preventDefault();
    shownRadios[(focusedPosition + step + shownRadios.length) % shownRadios.length]?.focus();
  });
}

/** Leaves only the chosen radio of a group in the Tab order. */
function setTabStop(radioButtons: Iterable<HTMLElement>): void {
  for (const radioButton of radioButtons) {
    radioButton.tabIndex = radioButton.getAttribute("aria-checked") === "true" ? 0 : -1;
  }
}

/**
 * Draws the export sheet: a live preview of the real document on the left,
 * and on the right what to export, as what, and what to include.
 *
 * It is app chrome built from fixed strings; the only model text it shows is
 * inside the preview, which is the already-cleaned document in a shadow root
 * so neither the app's styles nor the document's can reach the other.
 */
export class ExportSheetView {
  private readonly sheetElement: HTMLElement;
  private readonly callbacks: ExportSheetCallbacks;
  private readonly previewRoot: ShadowRoot;
  private readonly pageCount: HTMLElement;
  private readonly scopeButtons = new Map<ExportScopeKind, HTMLButtonElement>();
  private readonly formatButtons = new Map<ExportFormat, HTMLButtonElement>();
  private readonly toggleRows = new Map<ExportToggle, { row: HTMLElement; checkbox: HTMLInputElement }>();
  private readonly status: HTMLElement;
  private readonly saveButton: HTMLButtonElement;
  private readonly cancelButton: HTMLButtonElement;

  public constructor(sheetElement: HTMLElement, callbacks: ExportSheetCallbacks) {
    this.sheetElement = sheetElement;
    this.callbacks = callbacks;
    const panel = makeElement("div", "export-sheet__panel");
    const previewColumn = makeElement("div", "export-sheet__preview-column");
    const paper = makeElement("div", "export-sheet__paper");
    paper.dataset.testid = "export-preview";
    paper.setAttribute("role", "img");
    paper.setAttribute("aria-label", "Preview of the exported file");
    this.previewRoot = paper.attachShadow({ mode: "open" });
    this.pageCount = makeElement("p", "export-sheet__pages");
    this.pageCount.dataset.testid = "export-page-count";
    previewColumn.append(paper, this.pageCount);
    this.status = makeElement("p", "export-sheet__status");
    this.status.setAttribute("role", "status");
    this.status.dataset.testid = "export-status";
    this.saveButton = makeButton("export-save", "export-sheet__save", "");
    this.cancelButton = makeButton("export-cancel", "export-sheet__cancel", "Cancel");
    panel.append(previewColumn, this.controlsColumn());
    this.sheetElement.replaceChildren(panel);
    this.sheetElement.addEventListener("keydown", (event) => this.handleKey(event));
  }

  private controlsColumn(): HTMLElement {
    const controls = makeElement("div", "export-sheet__controls");
    const heading = makeElement("h2", "export-sheet__title", "Export");
    heading.id = "export-sheet-title";
    this.sheetElement.setAttribute("aria-labelledby", heading.id);
    const actions = makeElement("div", "export-sheet__actions");
    actions.append(this.cancelButton, this.saveButton);
    this.cancelButton.addEventListener("click", () => this.callbacks.cancel());
    this.saveButton.addEventListener("click", () => this.callbacks.save());
    controls.append(
      heading,
      makeElement("p", "export-sheet__label", "What"),
      this.scopeGroup(),
      makeElement("p", "export-sheet__label", "As"),
      this.formatGroup(),
      makeElement("p", "export-sheet__label", "Include"),
      this.toggleGroup(),
      this.status,
      actions,
    );
    return controls;
  }

  private scopeGroup(): HTMLElement {
    const scopeGroup = makeElement("div", "export-sheet__choices");
    scopeGroup.setAttribute("role", "radiogroup");
    scopeGroup.setAttribute("aria-label", "What to export");
    for (const scopeKind of Object.keys(SCOPE_LABELS) as ExportScopeKind[]) {
      const scopeButton = makeButton(`export-scope-${scopeKind}`, "export-sheet__choice", SCOPE_LABELS[scopeKind]);
      scopeButton.setAttribute("role", "radio");
      scopeButton.addEventListener("click", () => this.callbacks.chooseScope(scopeKind));
      this.scopeButtons.set(scopeKind, scopeButton);
      scopeGroup.appendChild(scopeButton);
    }
    enableArrowKeys(scopeGroup);
    return scopeGroup;
  }

  private formatGroup(): HTMLElement {
    const formatGroup = makeElement("div", "export-sheet__formats");
    formatGroup.setAttribute("role", "radiogroup");
    formatGroup.setAttribute("aria-label", "File type");
    for (const offer of EXPORT_FORMAT_OFFERS) {
      const formatButton = makeButton(`export-format-${offer.format}`, "export-sheet__format", "");
      formatButton.setAttribute("role", "radio");
      formatButton.append(makeElement("span", "export-sheet__format-name", offer.label), makeElement("span", "export-sheet__format-hint", offer.hint));
      formatButton.addEventListener("click", () => this.callbacks.chooseFormat(offer.format));
      this.formatButtons.set(offer.format, formatButton);
      formatGroup.appendChild(formatButton);
    }
    enableArrowKeys(formatGroup);
    return formatGroup;
  }

  private toggleGroup(): HTMLElement {
    const toggleGroup = makeElement("div", "export-sheet__toggles");
    for (const toggle of Object.keys(TOGGLE_LABELS) as ExportToggle[]) {
      const row = makeElement("label", "export-sheet__toggle");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.id = `export-include-${toggle}`;
      checkbox.dataset.testid = checkbox.id;
      checkbox.addEventListener("change", () => this.callbacks.toggle(toggle, checkbox.checked));
      row.append(checkbox, document.createTextNode(TOGGLE_LABELS[toggle]));
      this.toggleRows.set(toggle, { row, checkbox });
      toggleGroup.appendChild(row);
    }
    return toggleGroup;
  }

  /** Esc closes like Cancel, as every Mac sheet does. */
  private handleKey(event: KeyboardEvent): void {
    if (event.key === "Escape") {
      event.preventDefault();
      this.callbacks.cancel();
    }
  }

  /** The sheet is modal: while it is open the window behind it takes no
   * focus and is not read by VoiceOver. */
  private setWindowBehindInert(isInert: boolean): void {
    const windowRoot = this.sheetElement.parentElement;
    for (const sibling of windowRoot !== null ? [...windowRoot.children] : []) {
      if (sibling !== this.sheetElement) {
        sibling.toggleAttribute("inert", isInert);
      }
    }
  }

  public show(): void {
    this.sheetElement.hidden = false;
    this.setWindowBehindInert(true);
    this.saveButton.focus();
  }

  public hide(): void {
    this.sheetElement.hidden = true;
    this.setWindowBehindInert(false);
  }

  public get isShown(): boolean {
    return !this.sheetElement.hidden;
  }

  public render(options: ExportOptions, facts: ExportSheetFacts): void {
    for (const [scopeKind, scopeButton] of this.scopeButtons) {
      scopeButton.setAttribute("aria-checked", String(options.scope.kind === scopeKind));
      scopeButton.hidden = scopeKind === "oneAnswer" && !facts.canExportOneAnswer;
    }
    const pickedButton = this.scopeButtons.get("pickedMessages");
    if (pickedButton !== undefined) {
      pickedButton.textContent = facts.pickedMessageCount > 0 ? `${facts.pickedMessageCount} picked…` : SCOPE_LABELS.pickedMessages;
    }
    for (const [format, formatButton] of this.formatButtons) {
      formatButton.setAttribute("aria-checked", String(options.format === format));
    }
    setTabStop(this.scopeButtons.values());
    setTabStop(this.formatButtons.values());
    this.renderToggle("includesQuestions", options.includesQuestions, facts.hasQuestions);
    this.renderToggle("includesPictures", options.includesPictures, facts.hasPictures);
    this.renderToggle("includesAiNote", options.includesAiNote, true);
    this.saveButton.textContent = exportFormatOffer(options.format).saveLabel;
  }

  /** A switch with nothing to switch (no pictures in the chat) is hidden
   * rather than shown disabled, so the sheet only asks real questions. */
  private renderToggle(toggle: ExportToggle, isOn: boolean, isRelevant: boolean): void {
    const toggleRow = this.toggleRows.get(toggle);
    if (toggleRow !== undefined) {
      toggleRow.checkbox.checked = isOn;
      toggleRow.row.hidden = !isRelevant;
    }
  }

  public showPreview(previewMarkup: string, pageCountText: string): void {
    this.previewRoot.innerHTML = previewMarkup;
    this.pageCount.textContent = pageCountText;
  }

  /** Gives the preview's frame the scaled page's height, so the paper
   * scrolls through the whole document and no further. */
  public fitPreviewFrame(): void {
    const previewFrame = this.previewRoot.querySelector<HTMLElement>(`.${PREVIEW_FRAME_CLASS}`);
    const previewArticle = this.previewArticle();
    if (previewFrame !== null && previewArticle !== null) {
      previewFrame.style.height = `${Math.ceil(previewArticle.offsetHeight * PREVIEW_SCALE)}px`;
    }
  }

  public setPageCount(pageCountText: string): void {
    this.pageCount.textContent = pageCountText;
  }

  /** Settles once every picture in the preview has loaded or failed. */
  public async previewPicturesLoaded(): Promise<void> {
    const previewPictures = [...this.previewRoot.querySelectorAll("img")];
    await Promise.all(previewPictures.map((previewPicture) => (typeof previewPicture.decode === "function" ? previewPicture.decode().catch(() => undefined) : undefined)));
  }

  /** The preview's article, for measuring how many pages it makes. */
  public previewArticle(): HTMLElement | null {
    return this.previewRoot.querySelector<HTMLElement>(".export-article");
  }

  /** Where the preview's uncut blocks sit in the text at full size. */
  public previewWholeBlocks(textInsetPixels: number): WholeBlock[] {
    const article = this.previewArticle();
    return article !== null ? wholeBlocksIn(article, WHOLE_BLOCK_SELECTOR, textInsetPixels) : [];
  }

  public setSaving(isSaving: boolean, idleLabel: string): void {
    this.saveButton.disabled = isSaving;
    this.cancelButton.disabled = isSaving;
    this.saveButton.textContent = isSaving ? "Saving…" : idleLabel;
  }

  public setCanSave(canSave: boolean): void {
    this.saveButton.disabled = !canSave;
  }

  public setStatus(statusText: string): void {
    this.status.textContent = statusText;
  }
}
