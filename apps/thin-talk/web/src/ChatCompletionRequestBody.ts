/** One message as the daemon's chat completion endpoint receives it. */
export interface ChatCompletionRequestMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * The streaming chat completion request Thin Talk sends. `thinking_budget` is
 * the daemon's canonical spelling of the reasoning cap; the daemon ignores
 * fields it does not know, so a misspelt key would silently leave reasoning
 * uncapped rather than fail.
 */
export interface ChatCompletionRequestBody {
  model: string;
  stream: true;
  messages: ChatCompletionRequestMessage[];
  thinking_budget: number;
}
