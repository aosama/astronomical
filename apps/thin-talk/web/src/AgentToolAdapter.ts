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
      const resultText = await executeWithinLimits(conversationTool, toolArguments as Record<string, unknown>, toolSignal);
      return { content: [{ type: "text", text: resultText }], details: undefined };
    },
  };
}
