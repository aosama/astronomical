/** A JSON Schema object describing a tool's arguments, as the model sees it. */
export type ToolParametersSchema = {
  type: "object";
  properties: Record<string, unknown>;
  required?: string[];
  additionalProperties?: boolean;
};

/**
 * One capability the model may call while answering. Tools are plain objects
 * in Thin Talk's own terms; the turn runner adapts them to the agent loop and
 * applies the shared limits (timeout, result cap), so a tool only does its job
 * and stays testable without the loop.
 */
export interface ConversationTool {
  /** The name the model calls; lowercase with underscores. */
  readonly name: string;
  /** The short label the transcript shows for a step ("Current time"). */
  readonly label: string;
  /** What the tool does, written for the model deciding whether to call it. */
  readonly description: string;
  readonly parameters: ToolParametersSchema;
  /** Runs the tool with validated arguments and returns the text the model
   * reads. Throwing reports a failed step to the model. The signal fires on a
   * user stop or a timeout. */
  execute(toolArguments: Record<string, unknown>, signal: AbortSignal): Promise<string>;
}
