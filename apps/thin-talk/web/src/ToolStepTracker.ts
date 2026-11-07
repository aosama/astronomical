import { ToolStep, ToolStepState, UNKNOWN_TOOL_LABEL } from "./ToolStep";

/**
 * Builds the tool steps of one turn from the agent loop's start and end
 * reports. The loop reports a call's arguments only when it starts and its
 * result only when it ends, so the tracker keeps the arguments in between and
 * hands the conversation complete steps; it also numbers the model requests
 * (rounds) the calls came from.
 */
export class ToolStepTracker {
  private readonly labelFor: (toolName: string) => string;
  private readonly argumentsJSONByToolCallID = new Map<string, string>();
  private readonly roundIndexByToolCallID = new Map<string, number>();
  private completedRoundCount = 0;

  public constructor(labelFor: (toolName: string) => string) {
    this.labelFor = labelFor;
  }

  /** How many rounds of tool calls have finished in this turn. */
  public get completedRounds(): number {
    return this.completedRoundCount;
  }

  public started(toolCallID: string, toolName: string, toolArguments: unknown): ToolStep {
    const argumentsJSON = ToolStepTracker.argumentsJSON(toolArguments);
    this.argumentsJSONByToolCallID.set(toolCallID, argumentsJSON);
    this.roundIndexByToolCallID.set(toolCallID, this.completedRoundCount);
    return this.step(toolCallID, toolName, argumentsJSON, "", ToolStepState.RUNNING);
  }

  /** `generatedImageName` is the image the call produced, if any; the
   * transcript shows it. */
  public finished(toolCallID: string, toolName: string, toolResult: unknown, isError: boolean, generatedImageName: string | null = null): ToolStep {
    const recordedArgumentsJSON = this.argumentsJSONByToolCallID.get(toolCallID);
    const argumentsJSON = recordedArgumentsJSON != null ? recordedArgumentsJSON : "{}";
    const resultState = isError ? ToolStepState.FAILED : ToolStepState.SUCCEEDED;
    return this.step(toolCallID, toolName, argumentsJSON, ToolStepTracker.resultText(toolResult), resultState, generatedImageName);
  }

  /** Marks the end of one model request whose tool calls have all finished. */
  public completeRound(): void {
    this.completedRoundCount += 1;
  }

  private step(
    toolCallID: string,
    toolName: string,
    argumentsJSON: string,
    resultText: string,
    state: ToolStepState,
    generatedImageName: string | null = null,
  ): ToolStep {
    const recordedRoundIndex = this.roundIndexByToolCallID.get(toolCallID);
    return new ToolStep({
      toolCallID,
      toolName,
      label: toolName.length > 0 ? this.labelFor(toolName) : UNKNOWN_TOOL_LABEL,
      argumentsJSON,
      resultText,
      state,
      roundIndex: recordedRoundIndex != null ? recordedRoundIndex : this.completedRoundCount,
      generatedImageName,
    });
  }

  private static argumentsJSON(toolArguments: unknown): string {
    try {
      const encodedArguments = JSON.stringify(toolArguments);
      return encodedArguments != null ? encodedArguments : "{}";
    } catch {
      return "{}";
    }
  }

  /** The text blocks of a tool result, joined. Results carry text only. */
  private static resultText(toolResult: unknown): string {
    const resultRecord = typeof toolResult === "object" && toolResult !== null ? (toolResult as Record<string, unknown>) : {};
    const resultContent: unknown[] = Array.isArray(resultRecord["content"]) ? resultRecord["content"] : [];
    const resultTexts: string[] = [];
    for (const contentBlock of resultContent) {
      if (typeof contentBlock !== "object" || contentBlock === null) {
        continue;
      }
      const blockRecord = contentBlock as Record<string, unknown>;
      if (blockRecord["type"] === "text" && typeof blockRecord["text"] === "string") {
        resultTexts.push(blockRecord["text"]);
      }
    }
    return resultTexts.join("\n");
  }
}
