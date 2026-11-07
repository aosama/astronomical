/** A JSON Schema object describing a tool's arguments, as the model sees it. */
export type ToolParametersSchema = {
  type: "object";
  properties: Record<string, unknown>;
  required?: string[];
  additionalProperties?: boolean;
};

/** What a tool hands back: the text the model reads, and for the image tool
 * the stored name of the image it generated, which the answer shows. */
export interface ToolOutput {
  readonly text: string;
  readonly generatedImageName?: string;
}

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
  /** How long one call may run; the shared default applies when absent. A
   * tool that waits on the daemon (image generation) needs longer. */
  readonly timeoutMilliseconds?: number;
  /** Runs the tool with validated arguments and returns the text the model
   * reads, or a ToolOutput when it also produced an image. Throwing reports a
   * failed step to the model. The signal fires on a user stop or a timeout. */
  execute(toolArguments: Record<string, unknown>, signal: AbortSignal): Promise<string | ToolOutput>;
}
