import type { ToolCallContext } from "./ConversationTool";
import { SessionBridge, type HostToolUse } from "./SessionBridge";

/**
 * Runs one use of a host tool and turns every way it can end without a
 * result into an error the agent loop reports to the model. The messages keep
 * apart a refusal the user made from a tool that was never offered to them,
 * and tell the model to carry on rather than ask again, so one refusal does
 * not become a string of sheets.
 */
export async function runHostToolUse(
  bridge: SessionBridge,
  toolName: string,
  toolLabel: string,
  context: ToolCallContext,
  signal: AbortSignal,
): Promise<Exclude<HostToolUse, { outcome: "refused" | "nothingChosen" | "cancelled" | "unavailable" }>> {
  if (context.conversationID == null) {
    throw new Error(`"${toolLabel}" is not available because no conversation is open. Continue without it.`);
  }
  const hostToolUse = await bridge.useHostTool(toolName, context.conversationID, signal);
  switch (hostToolUse.outcome) {
    case "refused":
      throw new Error(`The user did not allow "${toolLabel}". Do not call it again in this answer; continue without it.`);
    case "nothingChosen":
      throw new Error(`The user allowed "${toolLabel}" but did not choose anything. Continue without it.`);
    case "cancelled":
      throw new Error(`"${toolLabel}" was stopped.`);
    case "unavailable":
      throw new Error(`"${toolLabel}" is not available right now (${hostToolUse.reason}). The user was not asked. Continue without it.`);
    default:
      return hostToolUse;
  }
}
