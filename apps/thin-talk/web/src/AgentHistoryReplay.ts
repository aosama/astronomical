import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Api, AssistantMessage, JsonObject, Model, ToolCall, ToolResultMessage } from "@earendil-works/pi-ai";
import { ChatMessage } from "./ChatMessage";
import { ChatRole } from "./ChatRole";
import { emptyAssistantMessage } from "./DaemonModelDescription";
import { ToolStep, ToolStepState, UNFINISHED_TOOL_CALL_RESULT } from "./ToolStep";

/**
 * Turns the saved conversation back into Pi's transcript for the next turn.
 *
 * An answer that used tools is replayed round by round: for each model
 * request that called tools, an assistant message with its calls and one
 * result per call; then the answer text. Text the model wrote between rounds
 * is saved as part of that one answer, so it is replayed after the last round.
 * That lets a follow-up question lean on what a tool returned. When the turn
 * offers no tools (the switch is off, or the model cannot call tools), earlier
 * steps are left out: a model that sees tool calls but has no tools declared
 * may try to call them anyway.
 */
export function agentHistory(
  earlierMessages: readonly ChatMessage[],
  model: Model<Api>,
  replaysToolSteps: boolean,
): AgentMessage[] {
  const history: AgentMessage[] = [];
  for (const earlierMessage of earlierMessages) {
    if (earlierMessage.role === ChatRole.USER) {
      history.push({ role: "user", content: earlierMessage.content, timestamp: Date.now() });
      continue;
    }
    if (replaysToolSteps && earlierMessage.toolSteps.length > 0) {
      for (const roundSteps of stepsByRound(earlierMessage.toolSteps)) {
        history.push(toolCallMessage(roundSteps, model));
        history.push(...roundSteps.map((toolStep) => toolResultMessage(toolStep)));
      }
      if (earlierMessage.content.length === 0) {
        continue;
      }
    }
    history.push(textMessage(earlierMessage.content, model));
  }
  return history;
}

/** Steps grouped by the round that asked for them, rounds in order. */
function stepsByRound(toolSteps: readonly ToolStep[]): ToolStep[][] {
  const roundIndexes = [...new Set(toolSteps.map((toolStep) => toolStep.roundIndex))].sort((left, right) => left - right);
  return roundIndexes.map((roundIndex) => toolSteps.filter((toolStep) => toolStep.roundIndex === roundIndex));
}

function textMessage(answerText: string, model: Model<Api>): AssistantMessage {
  const assistantMessage = emptyAssistantMessage(model);
  assistantMessage.content = [{ type: "text", text: answerText }];
  return assistantMessage;
}

function toolCallMessage(toolSteps: readonly ToolStep[], model: Model<Api>): AssistantMessage {
  const assistantMessage = emptyAssistantMessage(model);
  assistantMessage.content = toolSteps.map((toolStep): ToolCall => ({
    type: "toolCall",
    id: toolStep.toolCallID,
    name: toolStep.toolName,
    arguments: savedArguments(toolStep.argumentsJSON),
  }));
  assistantMessage.stopReason = "toolUse";
  return assistantMessage;
}

/** A step that never finished (the turn was stopped mid-call) is replayed as
 * a failed call, so the model does not assume it saw a result. */
function toolResultMessage(toolStep: ToolStep): ToolResultMessage {
  const isSuccessful = toolStep.state === ToolStepState.SUCCEEDED;
  const resultText = toolStep.state === ToolStepState.RUNNING ? UNFINISHED_TOOL_CALL_RESULT : toolStep.resultText;
  return {
    role: "toolResult",
    toolCallId: toolStep.toolCallID,
    toolName: toolStep.toolName,
    content: [{ type: "text", text: resultText }],
    isError: !isSuccessful,
    timestamp: Date.now(),
  };
}

function savedArguments(argumentsJSON: string): JsonObject {
  try {
    const decodedArguments: unknown = JSON.parse(argumentsJSON);
    if (typeof decodedArguments === "object" && decodedArguments !== null && !Array.isArray(decodedArguments)) {
      return decodedArguments as JsonObject;
    }
  } catch {
    // A damaged saved argument string replays as no arguments.
  }
  return {};
}
