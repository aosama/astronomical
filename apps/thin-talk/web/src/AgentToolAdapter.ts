import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ConversationTool } from "./ConversationTool";
import { executeWithinLimits } from "./ToolLimits";

/**
 * Adapts one of Thin Talk's tools to Pi's agent loop. The tool runs under the
 * shared timeout and result cap, and the turn's stop signal reaches it; Pi
 * reports a thrown error to the model as a failed call.
 */
export function agentTool(conversationTool: ConversationTool): AgentTool {
  return {
    name: conversationTool.name,
    label: conversationTool.label,
    description: conversationTool.description,
    // Pi accepts a plain JSON Schema here and validates arguments against it
    // without generating code, which the canvas's no-eval policy requires.
    parameters: conversationTool.parameters as unknown as AgentTool["parameters"],
    execute: async (_toolCallID, toolArguments, signal) => {
      const toolSignal = signal != null ? signal : new AbortController().signal;
      const toolOutput = await executeWithinLimits(conversationTool, toolArguments as Record<string, unknown>, toolSignal);
      // The image name rides in the result's details, which Pi hands to the
      // end-of-call report; the model never sees it because
      // DaemonStreamFunction sends only a result's text.
      const details = toolOutput.generatedImageName != null ? { [GENERATED_IMAGE_DETAIL]: toolOutput.generatedImageName } : undefined;
      return { content: [{ type: "text", text: toolOutput.text }], details };
    },
  };
}

/** The key under which a result's details carry a generated image's name;
 * written and read only in this file, so the two sides cannot drift. */
const GENERATED_IMAGE_DETAIL = "generatedImageName";

/** The generated image name a finished call reported, or null. Reads Pi's
 * result shape here so the rest of Thin Talk receives a plain value. */
export function generatedImageNameFromToolResult(toolResult: unknown): string | null {
  const resultRecord = typeof toolResult === "object" && toolResult !== null ? (toolResult as Record<string, unknown>) : {};
  const details = resultRecord["details"];
  const detailsRecord = typeof details === "object" && details !== null ? (details as Record<string, unknown>) : {};
  const imageName = detailsRecord[GENERATED_IMAGE_DETAIL];
  return typeof imageName === "string" ? imageName : null;
}
