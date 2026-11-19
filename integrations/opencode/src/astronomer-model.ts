// A single chat model row from the Astronomical /v1/models wire, after parsing.
//
// Every field the plugin reads is required and already resolved, so downstream
// code never has to re-apply a fallback or invent a default. `context_window`
// and `max_output_tokens` are required: chat models publish them while
// image-generation and embeddings models omit them, so a row missing either is
// rejected rather than registered on a guessed limit.
//
// Fields the daemon publishes but the plugin never reads (created, owned_by,
// output_modalities, reasoning_format, tool_call_format, structured_output
// enforcement) are deliberately absent. Carrying unread wire facts invites
// someone to act on a value nothing validated.

export type AstronomicalModelRow = {
  readonly id: string;
  readonly context_window: number;
  readonly max_output_tokens: number;
  readonly input_modalities: readonly string[];
  readonly supports_reasoning: boolean;
  readonly supports_tool_calls: boolean;
  readonly supports_structured_outputs: boolean;
  readonly supported_endpoints: readonly string[];
};
