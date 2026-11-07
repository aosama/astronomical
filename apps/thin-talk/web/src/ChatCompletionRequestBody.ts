/** One tool call an assistant message made, as the daemon receives it back. */
export interface ChatCompletionRequestToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

/** One part of a message that mixes text and images, in the OpenAI format the
 * daemon accepts; an image travels inline as a data URL. */
export type ChatCompletionContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

/** One message as the daemon's chat completion endpoint receives it. */
export type ChatCompletionRequestMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: string | ChatCompletionContentPart[] }
  | { role: "assistant"; content: string; tool_calls?: ChatCompletionRequestToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string };

/** One tool the model may call, in the OpenAI function format. */
export interface ChatCompletionRequestTool {
  type: "function";
  function: { name: string; description: string; parameters: unknown };
}

/**
 * The streaming chat completion request Thin Talk sends. `thinking_budget` is
 * the daemon's canonical spelling of the reasoning cap; the daemon ignores
 * fields it does not know, so a misspelt key would silently leave reasoning
 * uncapped rather than fail. `tools` is present only when the turn offers
 * tools, so a plain turn carries no declarations.
 */
export interface ChatCompletionRequestBody {
  model: string;
  stream: true;
  messages: ChatCompletionRequestMessage[];
  thinking_budget: number;
  tools?: ChatCompletionRequestTool[];
}
