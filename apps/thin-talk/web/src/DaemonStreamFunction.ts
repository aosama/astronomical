import type { Api, Message, Model, SimpleStreamOptions, TextContent, TranscriptContext } from "@earendil-works/pi-ai";
import { AssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import { AssistantEventAssembler } from "./AssistantEventAssembler";
import type { ChatCompletionRequestBody, ChatCompletionRequestMessage } from "./ChatCompletionRequestBody";
import { ChatFailure } from "./ChatFailure";
import { ChatFailureKind } from "./ChatFailureKind";
import { DaemonClient } from "./DaemonClient";

/**
 * Pi's stream function for one turn, written over DaemonClient instead of
 * Pi's OpenAI adapter. The adapter bundles the OpenAI SDK (about 275 KB
 * minified), adds request headers the daemon's CORS rule rejects, and refuses
 * to send without an API key; going through DaemonClient keeps the request to
 * the fields the daemon reads and the wire code in one tested place.
 *
 * Pi only sees an event stream that ends in `error`, so the classified
 * ChatFailure behind an error is kept here for the turn runner to rethrow,
 * preserving the failure kinds the conversation already knows how to explain.
 */
export class DaemonStreamFunction {
  private readonly client: DaemonClient;
  private readonly thinkingBudgetTokens: number;
  private readonly turnLifetimeSignal: AbortSignal;
  private failure: ChatFailure | null = null;

  /** `turnLifetimeSignal` fires when the turn that owns this function ends,
   * so a daemon request can never outlive its turn even when Pi stops
   * listening without aborting the request itself. */
  public constructor(client: DaemonClient, thinkingBudgetTokens: number, turnLifetimeSignal: AbortSignal) {
    this.client = client;
    this.thinkingBudgetTokens = thinkingBudgetTokens;
    this.turnLifetimeSignal = turnLifetimeSignal;
  }

  /** The classified failure behind the most recent `error` event, if any. */
  public get lastFailure(): ChatFailure | null {
    return this.failure;
  }

  /** Matches Pi's `StreamFn`: returns the event stream at once and fills it as
   * the daemon answers. */
  public stream(model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions): AssistantMessageEventStream {
    const eventStream = new AssistantMessageEventStream();
    const runSignal: AbortSignal | undefined = options?.signal;
    const requestSignal = runSignal != null ? AbortSignal.any([runSignal, this.turnLifetimeSignal]) : this.turnLifetimeSignal;
    void this.run(eventStream, model, context, requestSignal);
    return eventStream;
  }

  private async run(
    eventStream: AssistantMessageEventStream,
    model: Model<Api>,
    context: TranscriptContext,
    requestSignal: AbortSignal,
  ): Promise<void> {
    const assembler = new AssistantEventAssembler(eventStream, model);
    assembler.start();
    try {
      const requestBody: ChatCompletionRequestBody = {
        model: model.id,
        stream: true,
        messages: DaemonStreamFunction.requestMessages(context.messages),
        thinking_budget: this.thinkingBudgetTokens,
      };
      await this.client.streamChatCompletion(requestBody, requestSignal, (chunk) => assembler.apply(chunk));
      if (requestSignal.aborted) {
        assembler.fail("aborted", "Stopped by the user.");
        return;
      }
      assembler.finish();
    } catch (error) {
      const failure =
        error instanceof ChatFailure
          ? error
          : new ChatFailure(ChatFailureKind.UNKNOWN, error instanceof Error ? error.message : String(error));
      this.failure = failure;
      assembler.fail("error", failure.message);
    }
  }

  /** Pi's transcript in the daemon's message shape. Reasoning is not sent
   * back: the daemon's chat templates drop earlier turns' thinking anyway, and
   * re-sending it would only spend context. */
  public static requestMessages(transcript: readonly Message[]): ChatCompletionRequestMessage[] {
    const requestMessages: ChatCompletionRequestMessage[] = [];
    for (const transcriptMessage of transcript) {
      switch (transcriptMessage.role) {
        case "system": {
          const instructions = DaemonStreamFunction.plainText(transcriptMessage.content);
          if (instructions.length > 0) {
            requestMessages.push({ role: "system", content: instructions });
          }
          break;
        }
        case "user":
          requestMessages.push({ role: "user", content: DaemonStreamFunction.plainText(transcriptMessage.content) });
          break;
        case "assistant":
          requestMessages.push({ role: "assistant", content: DaemonStreamFunction.plainText(transcriptMessage.content) });
          break;
        case "toolResult":
          // No tools are offered yet, so a tool result here means the
          // transcript was built wrongly; failing beats silently dropping it.
          throw new Error("A tool result reached the daemon request before tools are offered.");
      }
    }
    return requestMessages;
  }

  private static plainText(content: string | readonly { type: string }[]): string {
    if (typeof content === "string") {
      return content;
    }
    let joinedText = "";
    for (const contentBlock of content) {
      if (contentBlock.type === "text") {
        joinedText += (contentBlock as TextContent).text;
      }
    }
    return joinedText;
  }
}
