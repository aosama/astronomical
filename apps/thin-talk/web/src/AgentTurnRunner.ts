import { Agent, type AgentEvent, type AgentMessage, type AgentTurnContext, type AgentTurnDecision, type BeforeToolCallResult } from "@earendil-works/pi-agent-core";
import type { Message } from "@earendil-works/pi-ai";
import { agentHistory } from "./AgentHistoryReplay";
import { agentTool, generatedImageNameFromToolResult } from "./AgentToolAdapter";
import { ChatFailure } from "./ChatFailure";
import { ChatFailureKind } from "./ChatFailureKind";
import { ChatRole } from "./ChatRole";
import type { ChatStreamer, ChatStreamRequest } from "./ChatStreamer";
import { DaemonClient } from "./DaemonClient";
import { daemonModel } from "./DaemonModelDescription";
import { DaemonStreamFunction } from "./DaemonStreamFunction";
import { ThinkingEffort } from "./ThinkingEffort";
import { ToolCatalog } from "./ToolCatalog";
import { MAXIMUM_TOOL_ROUNDS_PER_TURN, TOOL_ROUND_LIMIT_REASON } from "./ToolLimits";
import { ToolStepTracker } from "./ToolStepTracker";

/** How long a turn may go without any streamed activity before the
 * conversation reports a stall. A turn that is still thinking emits reasoning
 * deltas, and stall detection pauses while a tool runs (each tool has its own
 * timeout), so a minute of silence means the stream is stuck. */
const TURN_STALL_TIMEOUT_MILLISECONDS = 60_000;

/** What one turn tracks while it runs. */
interface TurnProgress {
  readonly toolSteps: ToolStepTracker;
  /** Set when a tool round finishes, so the next answer text starts a new
   * paragraph instead of running on from the text before the tools. */
  paragraphBreakPending: boolean;
  /** Whether any answer text has reached the conversation yet; a break is
   * only needed between two pieces of text. */
  hasForwardedText: boolean;
}

/**
 * Serves each assistant turn through Pi's agent loop: the model answers, may
 * call the tools the catalog offers, reads their results, and keeps going
 * until it answers, the round limit is reached, or the user stops.
 *
 * Pi is confined to this file, DaemonStreamFunction and their helpers
 * (AgentHistoryReplay, AgentToolAdapter, AssistantEventAssembler,
 * DaemonModelDescription): the controller sees only text, reasoning and
 * tool-step updates, and saved conversations keep their own shape, so Pi can
 * be replaced without migrating anyone's history.
 */
export class AgentTurnRunner implements ChatStreamer {
  public readonly stallTimeoutMilliseconds = TURN_STALL_TIMEOUT_MILLISECONDS;
  private readonly client: DaemonClient;
  private readonly toolCatalog: ToolCatalog;

  public constructor(client: DaemonClient, toolCatalog: ToolCatalog) {
    this.client = client;
    this.toolCatalog = toolCatalog;
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
    const streamFunction = new DaemonStreamFunction(
      this.client,
      ThinkingEffort.thinkingBudgetTokens(request.thinkingEffort),
      turnLifetime.signal,
    );
    const turnProgress: TurnProgress = {
      toolSteps: new ToolStepTracker((toolName) => this.toolCatalog.labelFor(toolName)),
      paragraphBreakPending: false,
      hasForwardedText: false,
    };
    const agent = this.turnAgent(request, streamFunction, turnProgress);
    const stopListening = agent.subscribe((event) => AgentTurnRunner.forwardEvent(event, request, turnProgress));
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

  /**
   * The round limit has two parts. Once MAXIMUM_TOOL_ROUNDS_PER_TURN rounds
   * have run, further calls are refused with a reason, so the model reads that
   * it must answer and gets one more request to do so. If it calls tools even
   * then, the turn ends after that refused round.
   */
  private turnAgent(request: ChatStreamRequest, streamFunction: DaemonStreamFunction, turnProgress: TurnProgress): Agent {
    const offeredTools = this.toolCatalog.toolsForTurn({
      modelSupportsToolCalls: request.modelSupportsToolCalls,
      toolsEnabled: request.toolsEnabled,
    });
    const model = daemonModel(request.modelID);
    const refuseCallsPastTheLimit = async (): Promise<BeforeToolCallResult | undefined> =>
      turnProgress.toolSteps.completedRounds >= MAXIMUM_TOOL_ROUNDS_PER_TURN ? { block: true, reason: TOOL_ROUND_LIMIT_REASON } : undefined;
    const countRoundsAndStopAfterARefusedOne = (completedTurn: AgentTurnContext): AgentTurnDecision | undefined => {
      if (completedTurn.toolResults.length === 0) {
        return undefined;
      }
      turnProgress.toolSteps.completeRound();
      turnProgress.paragraphBreakPending = true;
      return turnProgress.toolSteps.completedRounds > MAXIMUM_TOOL_ROUNDS_PER_TURN + 1 ? { action: "end" } : undefined;
    };
    return new Agent({
      initialState: {
        model,
        tools: offeredTools.map((offeredTool) => agentTool(offeredTool, { conversationID: request.conversationID, modelID: request.modelID })),
        messages: agentHistory(request.messages.slice(0, -1), model, offeredTools.length > 0),
      },
      streamFn: streamFunction.stream.bind(streamFunction),
      beforeToolCall: refuseCallsPastTheLimit,
      finishTurn: countRoundsAndStopAfterARefusedOne,
    });
  }

  private static forwardEvent(event: AgentEvent, request: ChatStreamRequest, turnProgress: TurnProgress): void {
    switch (event.type) {
      case "message_update":
        if (event.assistantMessageEvent.type === "text_delta") {
          const paragraphBreak = turnProgress.paragraphBreakPending && turnProgress.hasForwardedText ? "\n\n" : "";
          turnProgress.paragraphBreakPending = false;
          turnProgress.hasForwardedText = true;
          request.onText(paragraphBreak + event.assistantMessageEvent.delta);
        } else if (event.assistantMessageEvent.type === "thinking_delta") {
          request.onReasoning(event.assistantMessageEvent.delta);
        }
        break;
      case "tool_execution_start":
        request.onToolStep(turnProgress.toolSteps.started(event.toolCallId, event.toolName, event.args));
        break;
      case "tool_execution_end":
        request.onToolStep(
          turnProgress.toolSteps.finished(
            event.toolCallId,
            event.toolName,
            event.result,
            event.isError,
            generatedImageNameFromToolResult(event.result),
          ),
        );
        break;
      default:
        break;
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
