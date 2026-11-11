import { cleanExportBody } from "./ExportBodyCleaner";
import type { ExportEnvironment, ExportableMessage, ExportPictureData } from "./ExportEnvironment";
import type { ExportOptions, ExportScope } from "./ExportOptions";

/** The class of a generated picture's figure in an exported body. */
export const EXPORT_PICTURE_CLASS = "export-picture";

/** One message in the exported document: whose it is, and its cleaned body
 * with its pictures already inside. */
export interface ExportEntry {
  readonly isQuestion: boolean;
  readonly body: HTMLElement;
}

/** A conversation, or part of one, ready to be written in any format. */
export interface ExportContent {
  readonly title: string;
  readonly dateText: string;
  readonly entries: readonly ExportEntry[];
  /** The closing line saying an AI helped; null when the person left it out. */
  readonly aiNote: string | null;
}

const UNTITLED_CONVERSATION_TITLE = "Conversation";
const MAXIMUM_FILE_NAME_CHARACTERS = 80;
/** Characters a file name cannot hold on macOS or Windows; a Word file is
 * often opened on a school's Windows machine. */
const UNSAFE_FILE_NAME_CHARACTERS = /[\\/:*?"<>|\u0000-\u001f]/g;

/** The file name a document with this title is offered under, without an
 * extension. */
export function suggestedExportName(title: string): string {
  const safeName = title.replace(UNSAFE_FILE_NAME_CHARACTERS, " ").replace(/\s+/g, " ").trim().slice(0, MAXIMUM_FILE_NAME_CHARACTERS).trim();
  return safeName.length > 0 ? safeName : UNTITLED_CONVERSATION_TITLE;
}

export function aiNoteText(assistantName: string | null): string {
  return assistantName !== null ? `Written with help from ${assistantName}, an AI on this Mac.` : "Written with help from an AI on this Mac.";
}

/** The messages a scope covers, in conversation order. One answer brings the
 * question right before it, so "My questions" has something to show or hide. */
export function messagesInScope(messages: readonly ExportableMessage[], scope: ExportScope): ExportableMessage[] {
  if (scope.kind === "wholeChat") {
    return [...messages];
  }
  if (scope.kind === "pickedMessages") {
    return messages.filter((message) => scope.messageIDs.includes(message.messageID));
  }
  const answerIndex = messages.findIndex((message) => message.messageID === scope.messageID);
  if (answerIndex < 0) {
    return [];
  }
  const precedingMessage = answerIndex > 0 ? messages[answerIndex - 1] : undefined;
  const leadingQuestion = precedingMessage !== undefined && precedingMessage.isQuestion ? [precedingMessage] : [];
  return [...leadingQuestion, messages[answerIndex] as ExportableMessage];
}

type ContentSource = Pick<ExportEnvironment, "conversationTitle" | "assistantName" | "exportableMessages" | "loadPicture">;

/**
 * Builds the document a person asked for from the conversation on screen.
 *
 * It is rebuilt every time an option changes, because the sheet previews the
 * real document. Pictures are megabytes each and cross the host bridge, so
 * they are fetched once and kept for as long as the sheet is open.
 */
export class ExportContentBuilder {
  private readonly source: ContentSource;
  private readonly isDarkAppearance: boolean;
  private readonly currentDate: () => Date;
  private readonly picturesByName = new Map<string, Promise<ExportPictureData | null>>();

  public constructor(source: ContentSource, isDarkAppearance: boolean, currentDate: () => Date = () => new Date()) {
    this.source = source;
    this.isDarkAppearance = isDarkAppearance;
    this.currentDate = currentDate;
  }

  public async build(options: ExportOptions): Promise<ExportContent> {
    const scopedMessages = messagesInScope(this.source.exportableMessages(), options.scope);
    const includedMessages = scopedMessages.filter((message) => options.includesQuestions || !message.isQuestion);
    const entries: ExportEntry[] = [];
    for (const message of includedMessages) {
      const body = cleanExportBody(message.renderedBody, this.isDarkAppearance);
      if (options.includesPictures) {
        await this.prependPictures(body, message);
      }
      if ((body.textContent ?? "").trim().length > 0 || body.querySelector("img, svg, math") !== null) {
        entries.push({ isQuestion: message.isQuestion, body });
      }
    }
    const title = this.source.conversationTitle().trim();
    return {
      title: title.length > 0 ? title : UNTITLED_CONVERSATION_TITLE,
      dateText: new Intl.DateTimeFormat(undefined, { dateStyle: "long" }).format(this.currentDate()),
      entries,
      aiNote: options.includesAiNote ? aiNoteText(this.source.assistantName()) : null,
    };
  }

  /** Pictures go above the text, where the transcript shows them. A picture
   * the host no longer has is left out; the rest of the answer still exports. */
  private async prependPictures(body: HTMLElement, message: ExportableMessage): Promise<void> {
    const pictureFigures: HTMLElement[] = [];
    for (const imageName of message.generatedImageNames) {
      const pictureData = await this.loadPictureOnce(imageName);
      if (pictureData === null) {
        continue;
      }
      const pictureFigure = document.createElement("figure");
      pictureFigure.className = EXPORT_PICTURE_CLASS;
      const picture = document.createElement("img");
      picture.setAttribute("src", `data:${pictureData.mimeType};base64,${pictureData.base64}`);
      picture.setAttribute("alt", message.generatedImageDescriptions.get(imageName) ?? "Generated image");
      pictureFigure.appendChild(picture);
      pictureFigures.push(pictureFigure);
    }
    body.prepend(...pictureFigures);
  }

  private loadPictureOnce(imageName: string): Promise<ExportPictureData | null> {
    let pendingPicture = this.picturesByName.get(imageName);
    if (pendingPicture === undefined) {
      pendingPicture = this.source.loadPicture(imageName).catch(() => null);
      this.picturesByName.set(imageName, pendingPicture);
    }
    return pendingPicture;
  }
}
