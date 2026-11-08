import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { ConversationTool, ToolCallContext, ToolOutput } from "./ConversationTool";
import { executeWithinLimits } from "./ToolLimits";

/**
 * Adapts one of Thin Talk's tools to Pi's agent loop. The tool runs under the
 * shared timeout and result cap, and the turn's stop signal reaches it; Pi
 * reports a thrown error to the model as a failed call.
 */
export function agentTool(conversationTool: ConversationTool, context: ToolCallContext): AgentTool {
  return {
    name: conversationTool.name,
    label: conversationTool.label,
    description: conversationTool.description,
    // Pi accepts a plain JSON Schema here and validates arguments against it
    // without generating code, which the canvas's no-eval policy requires.
    parameters: conversationTool.parameters as unknown as AgentTool["parameters"],
    // A tool that asks the user runs alone, so two sheets never compete for
    // the conversation window; other tools keep running in parallel.
    executionMode: conversationTool.needsUserAttention === true ? "sequential" : undefined,
    execute: async (_toolCallID, toolArguments, signal) => {
      const toolSignal = signal != null ? signal : new AbortController().signal;
      const toolOutput = await executeWithinLimits(conversationTool, toolArguments as Record<string, unknown>, toolSignal, context);
      // The image name and its model's name ride in the result's details,
      // which Pi hands to the end-of-call report; the model never sees them.
      // An image for the model rides in the content, which
      // DaemonStreamFunction sends on.
      const details = generatedImageDetails(toolOutput);
      const imageForModel = toolOutput.imageForModel;
      const resultContent: (TextContent | ImageContent)[] = [{ type: "text", text: toolOutput.text }];
      if (imageForModel != null) {
        resultContent.push({ type: "image", data: imageForModel.base64, mimeType: imageForModel.mimeType });
      }
      return { content: resultContent, details };
    },
  };
}

/** The keys under which a result's details carry a generated image's name and
 * its model's name; written and read only in this file, so the two sides
 * cannot drift. */
const GENERATED_IMAGE_DETAIL = "generatedImageName";
const GENERATED_IMAGE_MODEL_DETAIL = "generatedImageModelName";

function generatedImageDetails(toolOutput: ToolOutput): Record<string, string> | undefined {
  if (toolOutput.generatedImageName == null) {
    return undefined;
  }
  const details: Record<string, string> = { [GENERATED_IMAGE_DETAIL]: toolOutput.generatedImageName };
  if (toolOutput.generatedImageModelName != null) {
    details[GENERATED_IMAGE_MODEL_DETAIL] = toolOutput.generatedImageModelName;
  }
  return details;
}

/** The generated image name a finished call reported, or null. Reads Pi's
 * result shape here so the rest of Thin Talk receives a plain value. */
export function generatedImageNameFromToolResult(toolResult: unknown): string | null {
  return stringDetail(toolResult, GENERATED_IMAGE_DETAIL);
}

/** The name of the model that made a finished call's image, or null. */
export function generatedImageModelNameFromToolResult(toolResult: unknown): string | null {
  return stringDetail(toolResult, GENERATED_IMAGE_MODEL_DETAIL);
}

function stringDetail(toolResult: unknown, detailKey: string): string | null {
  const resultRecord = typeof toolResult === "object" && toolResult !== null ? (toolResult as Record<string, unknown>) : {};
  const details = resultRecord["details"];
  const detailsRecord = typeof details === "object" && details !== null ? (details as Record<string, unknown>) : {};
  const detailValue = detailsRecord[detailKey];
  return typeof detailValue === "string" ? detailValue : null;
}
