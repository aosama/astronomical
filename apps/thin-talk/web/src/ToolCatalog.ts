import type { ConversationTool } from "./ConversationTool";

/**
 * Decides which tools a turn offers. A model that does not declare tool
 * support gets none, so its turn is a plain completion with no tool
 * declarations to read; the user's Tools switch can turn them off for any
 * model.
 */
export class ToolCatalog {
  private readonly availableTools: readonly ConversationTool[];

  public constructor(availableTools: readonly ConversationTool[]) {
    const seenToolNames = new Set<string>();
    for (const availableTool of availableTools) {
      if (seenToolNames.has(availableTool.name)) {
        throw new Error(`two tools share the name ${availableTool.name}`);
      }
      seenToolNames.add(availableTool.name);
    }
    this.availableTools = availableTools;
  }

  public toolsForTurn(init: { modelSupportsToolCalls: boolean; toolsEnabled: boolean }): readonly ConversationTool[] {
    if (!init.modelSupportsToolCalls || !init.toolsEnabled) {
      return [];
    }
    return this.availableTools;
  }

  /** The label for a tool name the model used, falling back to the name
   * itself when the model called a tool the catalog does not have. */
  public labelFor(toolName: string): string {
    const knownTool = this.availableTools.find((availableTool) => availableTool.name === toolName);
    return knownTool != null ? knownTool.label : toolName;
  }
}
