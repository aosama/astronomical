import { ChatMessageState } from "./ChatMessageState";
import { ChatRole } from "./ChatRole";
import { generatedImageDescription } from "./Renderer";
import type { TranscriptMessage } from "./TranscriptMessage";
import type { TranscriptView } from "./TranscriptView";
import type { ExportableMessage } from "./export/ExportEnvironment";

/** Whether a message is finished enough to export. A reply still arriving, a
 * queued question, and a question that was never sent are not; a reply the
 * person stopped is, since what arrived is what they read. */
export function isExportableMessage(message: TranscriptMessage): boolean {
  if (message.role === ChatRole.USER) {
    return message.state === ChatMessageState.COMPLETE;
  }
  return message.state === ChatMessageState.COMPLETE || message.state === ChatMessageState.STOPPED;
}

/** The finished messages on screen, as the export script reads them. */
export function exportableMessages(transcript: TranscriptView): ExportableMessage[] {
  const finishedMessages: ExportableMessage[] = [];
  for (const message of transcript.latestMessages()) {
    const renderedBody = isExportableMessage(message) ? transcript.renderedBodyClone(message.id) : null;
    if (renderedBody === null) {
      continue;
    }
    const pictureSteps = message.toolSteps.filter((toolStep) => toolStep.generatedImageName != null);
    finishedMessages.push({
      messageID: message.id,
      isQuestion: message.role === ChatRole.USER,
      renderedBody,
      generatedImageNames: pictureSteps.map((toolStep) => toolStep.generatedImageName as string),
      generatedImageDescriptions: new Map(pictureSteps.map((toolStep) => [toolStep.generatedImageName as string, generatedImageDescription(toolStep.argumentsJSON)])),
    });
  }
  return finishedMessages;
}
