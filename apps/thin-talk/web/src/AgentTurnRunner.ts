import { Agent, type AgentEvent, type AgentMessage } from "@earendil-works/pi-agent-core";
import type { Api, AssistantMessage, Message, Model } from "@earendil-works/pi-ai";
import { ChatFailure } from "./ChatFailure";
import { ChatFailureKind } from "./ChatFailureKind";
import { ChatMessage } from "./ChatMessage";
import { ChatRole } from "./ChatRole";
import type { ChatStreamer, ChatStreamRequest } from "./ChatStreamer";
import { DaemonClient } from "./DaemonClient";
import { daemonModel, emptyAssistantMessage } from "./DaemonModelDescription";
import { DaemonStreamFunction } from "./DaemonStreamFunction";
import { ThinkingEffort } from "./ThinkingEffort";

/** How long a turn may go without any streamed activity before the
 * conversation reports a stall. A turn that is still thinking emits reasoning
 * deltas, so a minute of silence means the stream is stuck. */
const TURN_STALL_TIMEOUT_MILLISECONDS = 60_000;

/**
 * Serves each assistant turn through Pi's agent loop. No tools are offered
 * yet, so a turn is one model response; the loop is in place so tools can be
 * added one file at a time without touching the conversation controller.
 *
 * Pi is confined to this file, DaemonStreamFunction and its helpers
 * (AssistantEventAssembler, DaemonModelDescription): the controller still
 * sees only text and reasoning deltas, and saved conversations keep their own
 * shape, so Pi can be replaced without migrating anyone's history.
 */
export class AgentTurnRunner implements ChatStreamer {
  public readonly stallTimeoutMilliseconds = TURN_STALL_TIMEOUT_MILLISECONDS;
  private readonly client: DaemonClient;

  public constructor(client: DaemonClient) {
    this.client = client;
  }

  public async streamChat(request: ChatStreamRequest): Promise<void> {
    const latestQuestion = request.messages[request.messages.length - 1];
    if (latestQuestion == null || latestQuestion.role !== ChatRole.USER) {
      throw new ChatFailure(ChatFailureKind.INVALID_REQUEST, "the conversation has no question to answer");
    }
    if (request.signal.aborted) {
      return;
    }
    // Pi ends a run without aborting it when a listener throws, which would
    // leave the daemon request streaming in the background and the worker
    // busy. This turn-scoped signal is aborted once the run ends, whatever
    // ended it, so no daemon request outlives its turn.
    const turnLifetime = new AbortController();
    const model = daemonModel(request.modelID);
    const streamFunction = new DaemonStreamFunction(
      this.client,
      ThinkingEffort.thinkingBudgetTokens(request.thinkingEffort),
      turnLifetime.signal,
    );
    const agent = new Agent({
      initialState: {
        model,
        messages: request.messages.slice(0, -1).map((earlierMessage) => AgentTurnRunner.agentMessage(earlierMessage, model)),
      },
      streamFn: streamFunction.stream.bind(streamFunction),
    });
    const stopListening = agent.subscribe((event) => AgentTurnRunner.forwardEvent(event, request));
    const stopTurn = (): void => {
      agent.abort();
      turnLifetime.abort();
    };
    request.signal.addEventListener("abort", stopTurn);
    try {
      await agent.prompt(latestQuestion.content);
    } finally {
      request.signal.removeEventListener("abort", stopTurn);
      stopListening();
      turnLifetime.abort();
    }
    AgentTurnRunner.throwIfTurnFailed(agent.state.messages, streamFunction, request.signal);
  }

  private static agentMessage(chatMessage: ChatMessage, model: Model<Api>): AgentMessage {
    if (chatMessage.role === ChatRole.USER) {
      return { role: "user", content: chatMessage.content, timestamp: Date.now() };
    }
    const assistantMessage: AssistantMessage = emptyAssistantMessage(model);
    assistantMessage.content = [{ type: "text", text: chatMessage.content }];
    return assistantMessage;
  }

  private static forwardEvent(event: AgentEvent, request: ChatStreamRequest): void {
    if (event.type !== "message_update") {
      return;
    }
    const streamEvent = event.assistantMessageEvent;
    if (streamEvent.type === "text_delta") {
      request.onText(streamEvent.delta);
    } else if (streamEvent.type === "thinking_delta") {
      request.onReasoning(streamEvent.delta);
    }
  }

  /** A user stop is not a failure. Otherwise the stream function's classified
   * failure wins, so the conversation keeps its specific explanations; any
   * other error the loop recorded (a listener that threw, for one) becomes an
   * unknown failure. */
  private static throwIfTurnFailed(
    transcript: readonly AgentMessage[],
    streamFunction: DaemonStreamFunction,
    signal: AbortSignal,
  ): void {
    if (signal.aborted) {
      return;
    }
    const streamFailure = streamFunction.lastFailure;
    if (streamFailure != null) {
      throw streamFailure;
    }
    const finalMessage = transcript[transcript.length - 1] as Message | undefined;
    if (finalMessage != null && finalMessage.role === "assistant" && finalMessage.stopReason === "error") {
      const errorMessage: string | undefined = finalMessage.errorMessage;
      throw new ChatFailure(ChatFailureKind.UNKNOWN, errorMessage != null ? errorMessage : null);
    }
  }
}
