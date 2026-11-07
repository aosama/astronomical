import type { AgentMessage } from "@earendil-works/pi-agent-core";

/** Share of the model's context window a request may fill. The rest is left
 * for the answer, for the template's own tokens, and for the estimate below
 * being rough. */
export const CONTEXT_BUDGET_SHARE = 0.8;

/** Characters per token used to estimate a request's size; English text runs
 * near four. */
const ESTIMATED_CHARACTERS_PER_TOKEN = 4;

/**
 * Keeps a long conversation inside the model's context window by dropping its
 * oldest whole turns. A turn is a user message and everything after it up to
 * the next user message (answers, tool calls and their results), so a tool
 * call is never separated from its result. Messages before the first user
 * message (the system instructions and tool declarations) are always kept, and
 * so is the latest turn, even when it alone is too long: the daemon then
 * refuses it with a clear context-length error instead of an answer to a
 * question the model never fully saw.
 *
 * Each message is measured once and turns are dropped from a running total,
 * since this runs before every model request of a turn.
 */
export function trimToContextBudget(transcript: readonly AgentMessage[], contextWindowTokens: number): AgentMessage[] {
  const turnStartIndexes = transcript.flatMap((agentMessage, messageIndex) => (agentMessage.role === "user" ? [messageIndex] : []));
  const firstTurnStart = turnStartIndexes[0];
  if (firstTurnStart == null) {
    return [...transcript];
  }
  const budgetTokens = Math.floor(Math.max(0, contextWindowTokens) * CONTEXT_BUDGET_SHARE);
  const messageTokens = transcript.map((agentMessage) => estimatedTokens([agentMessage]));
  let keptTokens = messageTokens.reduce((tokenTotal, tokenCount) => tokenTotal + tokenCount, 0);
  let keptStart = firstTurnStart;
  for (let turnNumber = 1; turnNumber < turnStartIndexes.length && keptTokens > budgetTokens; turnNumber += 1) {
    const nextTurnStart = turnStartIndexes[turnNumber] as number;
    for (let messageIndex = keptStart; messageIndex < nextTurnStart; messageIndex += 1) {
      keptTokens -= messageTokens[messageIndex] as number;
    }
    keptStart = nextTurnStart;
  }
  return [...transcript.slice(0, firstTurnStart), ...transcript.slice(keptStart)];
}

/** A rough token count for messages, from the text the model reads: message
 * text, reasoning, tool-call arguments and tool results. Bookkeeping fields
 * (usage, timestamps, model names) never reach the model and are not counted. */
export function estimatedTokens(agentMessages: readonly AgentMessage[]): number {
  let characterCount = 0;
  for (const agentMessage of agentMessages) {
    characterCount += readableCharacterCount(agentMessage);
  }
  return Math.ceil(characterCount / ESTIMATED_CHARACTERS_PER_TOKEN);
}

function readableCharacterCount(agentMessage: AgentMessage): number {
  // A system message also carries the tool declarations, which are sent in
  // full, so it is measured whole.
  if (agentMessage.role === "system") {
    return JSON.stringify(agentMessage).length;
  }
  const content = (agentMessage as { content?: unknown }).content;
  if (typeof content === "string") {
    return content.length;
  }
  if (!Array.isArray(content)) {
    return 0;
  }
  let characterCount = 0;
  for (const contentBlock of content as Record<string, unknown>[]) {
    if (typeof contentBlock["text"] === "string") {
      characterCount += contentBlock["text"].length;
    } else if (typeof contentBlock["thinking"] === "string") {
      characterCount += contentBlock["thinking"].length;
    } else if (contentBlock["type"] === "toolCall") {
      characterCount += JSON.stringify(contentBlock["arguments"] ?? {}).length + String(contentBlock["name"] ?? "").length;
    }
  }
  return characterCount;
}
