/** Where one tool call stands within an assistant turn. */
export enum ToolStepState {
  RUNNING = "running",
  SUCCEEDED = "succeeded",
  FAILED = "failed",
}

/** The result recorded for a call that never finished: the turn was stopped,
 * or the app quit, while the tool was still running. Shown in the row and
 * replayed to the model, so neither sees an empty result. */
export const UNFINISHED_TOOL_CALL_RESULT = "The call was stopped before it finished.";

/** The label for a call to a tool name the model made up or left empty. */
export const UNKNOWN_TOOL_LABEL = "Unknown tool";

/** A stored image name exactly as the app's image store produces it: a
 * lowercase UUID with a `.png` or `.jpg` extension. The store checks the same
 * pattern before reading a file; checking here as well keeps a damaged or
 * crafted name out of the transcript markup. */
const GENERATED_IMAGE_NAME = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(?:png|jpg)$/;

export function isGeneratedImageName(candidate: string): boolean {
  return GENERATED_IMAGE_NAME.test(candidate);
}

/**
 * One tool call the model made while answering: what it asked for and what
 * came back. Steps are saved with the assistant message so a reloaded
 * conversation shows them, and so a follow-up question can be answered with
 * the tool results the model already saw.
 */
export class ToolStep {
  /** The model's tool call id; it pairs the call with its result on replay. */
  public readonly toolCallID: string;
  /** The tool the model named; empty when the model sent no name. */
  public readonly toolName: string;
  /** The human label shown in the transcript row ("Current time"). */
  public readonly label: string;
  /** The model's arguments as JSON text, as the model sent them and before
   * they were checked against the tool's schema; `{}` when they could not be
   * read. */
  public readonly argumentsJSON: string;
  /** The result text the model saw, already capped by the tool limits. */
  public readonly resultText: string;
  public readonly state: ToolStepState;
  /** Which model request in the turn asked for this call, counting from 0.
   * Replay groups calls by round so the model sees its turn as it happened. */
  public readonly roundIndex: number;
  /** The stored image the call produced (the image tool), shown in the answer. */
  public readonly generatedImageName: string | null;
  /** The display name of the model that made the image, shown on the row;
   * null for other tools and for steps saved before it was recorded. */
  public readonly generatedImageModelName: string | null;

  public constructor(init: {
    toolCallID: string;
    toolName: string;
    label: string;
    argumentsJSON: string;
    resultText: string;
    state: ToolStepState;
    roundIndex?: number;
    generatedImageName?: string | null;
    generatedImageModelName?: string | null;
  }) {
    this.toolCallID = init.toolCallID;
    this.toolName = init.toolName;
    this.label = init.label;
    this.argumentsJSON = init.argumentsJSON;
    this.resultText = init.resultText;
    this.state = init.state;
    const configuredRoundIndex: number | undefined = init.roundIndex;
    this.roundIndex = configuredRoundIndex != null ? configuredRoundIndex : 0;
    const configuredImageName: string | null | undefined = init.generatedImageName;
    this.generatedImageName = configuredImageName != null && isGeneratedImageName(configuredImageName) ? configuredImageName : null;
    const configuredImageModelName: string | null | undefined = init.generatedImageModelName;
    this.generatedImageModelName = this.generatedImageName != null && configuredImageModelName != null ? configuredImageModelName : null;
  }

  /** This step with the name of the model that made its image. */
  public withGeneratedImageModelName(generatedImageModelName: string | null): ToolStep {
    return new ToolStep({
      toolCallID: this.toolCallID,
      toolName: this.toolName,
      label: this.label,
      argumentsJSON: this.argumentsJSON,
      resultText: this.resultText,
      state: this.state,
      roundIndex: this.roundIndex,
      generatedImageName: this.generatedImageName,
      generatedImageModelName,
    });
  }

  public toPlain(): Record<string, string | number | null> {
    return {
      generatedImageName: this.generatedImageName,
      generatedImageModelName: this.generatedImageModelName,
      toolCallID: this.toolCallID,
      toolName: this.toolName,
      label: this.label,
      argumentsJSON: this.argumentsJSON,
      resultText: this.resultText,
      state: this.state,
      roundIndex: this.roundIndex,
    };
  }

  /** Rebuilds a saved step, or returns null when it has no call id, so a
   * damaged step is dropped instead of failing the whole conversation. A step
   * saved while still running is restored as failed with the unfinished
   * result: its real result never arrived. */
  public static fromPlain(plain: unknown): ToolStep | null {
    if (typeof plain !== "object" || plain === null) {
      return null;
    }
    const record = plain as Record<string, unknown>;
    const toolCallID = record["toolCallID"];
    if (typeof toolCallID !== "string" || toolCallID.length === 0) {
      return null;
    }
    const toolName = typeof record["toolName"] === "string" ? record["toolName"] : "";
    const savedLabel = typeof record["label"] === "string" ? record["label"] : "";
    const label = savedLabel.length > 0 ? savedLabel : toolName.length > 0 ? toolName : UNKNOWN_TOOL_LABEL;
    const argumentsJSON = typeof record["argumentsJSON"] === "string" ? record["argumentsJSON"] : "{}";
    const wasUnfinished = record["state"] !== ToolStepState.SUCCEEDED && record["state"] !== ToolStepState.FAILED;
    const savedResultText = typeof record["resultText"] === "string" ? record["resultText"] : "";
    const resultText = wasUnfinished && savedResultText.length === 0 ? UNFINISHED_TOOL_CALL_RESULT : savedResultText;
    const state = record["state"] === ToolStepState.SUCCEEDED ? ToolStepState.SUCCEEDED : ToolStepState.FAILED;
    const savedRoundIndex = record["roundIndex"];
    const roundIndex = typeof savedRoundIndex === "number" && Number.isInteger(savedRoundIndex) && savedRoundIndex >= 0 ? savedRoundIndex : 0;
    const savedImageName = record["generatedImageName"];
    const generatedImageName = typeof savedImageName === "string" ? savedImageName : null;
    const savedImageModelName = record["generatedImageModelName"];
    const generatedImageModelName = typeof savedImageModelName === "string" ? savedImageModelName : null;
    return new ToolStep({ toolCallID, toolName, label, argumentsJSON, resultText, state, roundIndex, generatedImageName, generatedImageModelName });
  }
}
