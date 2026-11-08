import type { ImageForModel } from "./ConversationTool";
import { Agent, type AgentEvent, type AgentMessage, type AgentTurnContext, type AgentTurnDecision, type BeforeToolCallResult } from "@earendil-works/pi-agent-core";
import type { Message } from "@earendil-works/pi-ai";
import { trimToContextBudget } from "./AgentContextTrimmer";
import { agentHistory } from "./AgentHistoryReplay";
import { agentTool, generatedImageModelNameFromToolResult, generatedImageNameFromToolResult } from "./AgentToolAdapter";
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
  /** Queued messages handed to the loop, keyed by the message object itself:
   * the loop emits that same object when it starts the message, so the id is
   * found by identity, whatever order or timing the loop chose. */
  readonly queuedMessageIDsByMessage: Map<AgentMessage, string>;
}

/** The turn currently running, if any, so queued messages can reach it. */
interface ActiveTurn {
  readonly agent: Agent;
  readonly progress: TurnProgress;
}

/** Reads a generated image back by its stored name, or null when it is gone. */
export type GeneratedImageLoader = (imageName: string) => Promise<ImageForModel | null>;

function noGeneratedImages(): Promise<ImageForModel | null> {
  return Promise.resolve(null);
}

/** How many of a conversation's latest generated images a model that can see
 * is shown again on each turn. */
const RECALLED_IMAGE_LIMIT = 2;

/**
 * Serves each assistant turn through Pi's agent loop: the model answers, may
 * call the tools the catalog offers, reads their results, and keeps going
 * until it answers, the round limit is reached, or the user stops. Messages
 * the user sends meanwhile are answered in the same run, one after another.
 *
 * Pi is confined to this file, DaemonStreamFunction and their helpers
 * (AgentContextTrimmer, AgentHistoryReplay, AgentToolAdapter,
 * AssistantEventAssembler, DaemonModelDescription): the controller sees only text, reasoning and
 * tool-step updates, and saved conversations keep their own shape, so Pi can
 * be replaced without migrating anyone's history.
 */
export class AgentTurnRunner implements ChatStreamer {
  public readonly stallTimeoutMilliseconds = TURN_STALL_TIMEOUT_MILLISECONDS;
  private readonly client: DaemonClient;
  private readonly toolCatalog: ToolCatalog;
  private readonly generatedImageLoader: GeneratedImageLoader;
  private activeTurn: ActiveTurn | null = null;

  public constructor(client: DaemonClient, toolCatalog: ToolCatalog, generatedImageLoader: GeneratedImageLoader = noGeneratedImages) {
    this.client = client;
    this.toolCatalog = toolCatalog;
    this.generatedImageLoader = generatedImageLoader;
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
      queuedMessageIDsByMessage: new Map(),
    };
    const recalledImagesByName = await this.recallGeneratedImages(request);
    // Recalling images waits on the host, before the stop listener below is
    // attached; a stop pressed meanwhile must still end the turn unsent.
    if (request.signal.aborted) {
      return;
    }
    const agent = this.turnAgent(request, streamFunction, turnProgress, recalledImagesByName);
    const stopListening = agent.subscribe((event) => AgentTurnRunner.forwardEvent(event, request, turnProgress));
    this.activeTurn = { agent, progress: turnProgress };
    const stopTurn = (): void => {
      agent.abort();
      turnLifetime.abort();
    };
    request.signal.addEventListener("abort", stopTurn);
    try {
      await agent.prompt(latestQuestion.content);
    } finally {
      // Messages the loop never took stay queued in the conversation, which
      // sends them itself once this turn is over.
      this.activeTurn = null;
      agent.clearAllQueues();
      request.signal.removeEventListener("abort", stopTurn);
      stopListening();
      turnLifetime.abort();
    }
    AgentTurnRunner.throwIfTurnFailed(agent.state.messages, streamFunction, request.signal);
  }

  /** Hands a queued message to the running turn through Pi's steering
   * queue. Steering is taken after the response in progress (and after its
   * tool calls, if it made any), so a message sent mid-answer is read next,
   * and one sent while a tool runs is read before the model continues. One
   * queue keeps queued messages in the order the user sent them. */
  public queueUserMessage(queuedMessageID: string, content: string): boolean {
    const activeTurn = this.activeTurn;
    if (activeTurn == null) {
      return false;
    }
    const queuedMessage: AgentMessage = { role: "user", content, timestamp: Date.now() };
    activeTurn.progress.queuedMessageIDsByMessage.set(queuedMessage, queuedMessageID);
    activeTurn.agent.steer(queuedMessage);
    return true;
  }

  /**
   * The round limit has two parts. Once MAXIMUM_TOOL_ROUNDS_PER_TURN rounds
   * have run, further calls are refused with a reason, so the model reads that
   * it must answer and gets one more request to do so. If it calls tools even
   * then, the turn ends after that refused round.
   */
  private turnAgent(
    request: ChatStreamRequest,
    streamFunction: DaemonStreamFunction,
    turnProgress: TurnProgress,
    recalledImagesByName: ReadonlyMap<string, ImageForModel>,
  ): Agent {
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
        messages: agentHistory(request.messages.slice(0, -1), model, offeredTools.length > 0, recalledImagesByName),
      },
      streamFn: streamFunction.stream.bind(streamFunction),
      transformContext: AgentTurnRunner.contextTrimmer(request.contextWindowTokens),
      beforeToolCall: refuseCallsPastTheLimit,
      finishTurn: countRoundsAndStopAfterARefusedOne,
    });
  }

  /** The images generated earlier in this conversation that a model able to
   * see is shown again, newest first, at most RECALLED_IMAGE_LIMIT of them.
   * Each image costs the model a large block of context on every turn, so
   * only the latest few come back; older ones keep their text result. An
   * image that cannot be read is left out rather than failing the turn. */
  private async recallGeneratedImages(request: ChatStreamRequest): Promise<Map<string, ImageForModel>> {
    const recalledImagesByName = new Map<string, ImageForModel>();
    // Images ride on replayed tool steps, which are replayed only when the
    // turn offers tools; without them the images would be loaded for nothing.
    const offersTools = this.toolCatalog.toolsForTurn({
      modelSupportsToolCalls: request.modelSupportsToolCalls,
      toolsEnabled: request.toolsEnabled,
    }).length > 0;
    if (!request.modelAcceptsImages || !offersTools) {
      return recalledImagesByName;
    }
    const generatedImageNames = request.messages
      .flatMap((earlierMessage) => earlierMessage.toolSteps)
      .flatMap((toolStep) => (toolStep.generatedImageName != null ? [toolStep.generatedImageName] : []));
    for (const imageName of generatedImageNames.slice(-RECALLED_IMAGE_LIMIT)) {
      if (request.signal.aborted) {
        break;
      }
      const recalledImage = await this.generatedImageLoader(imageName).catch(() => null);
      if (recalledImage != null) {
        recalledImagesByName.set(imageName, recalledImage);
      }
    }
    return recalledImagesByName;
  }

  /** Trims each request to the model's context window when the daemon says
   * how large it is; otherwise sends the whole conversation, as before. */
  private static contextTrimmer(contextWindowTokens: number | null): ((transcript: AgentMessage[]) => Promise<AgentMessage[]>) | undefined {
    if (contextWindowTokens == null) {
      return undefined;
    }
    return async (transcript) => trimToContextBudget(transcript, contextWindowTokens);
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
      case "message_start":
        AgentTurnRunner.noteUserMessageStart(event.message, request, turnProgress);
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
            generatedImageModelNameFromToolResult(event.result),
          ),
        );
        break;
      default:
        break;
    }
  }

  /** A user message the loop starts that was handed in by queueUserMessage
   * begins a new answer, with its own paragraphs and round count. The turn's
   * own question is not in the map, so it starts nothing. */
  private static noteUserMessageStart(startedMessage: AgentMessage, request: ChatStreamRequest, turnProgress: TurnProgress): void {
    const queuedMessageID = turnProgress.queuedMessageIDsByMessage.get(startedMessage);
    if (queuedMessageID == null) {
      return;
    }
    turnProgress.queuedMessageIDsByMessage.delete(startedMessage);
    turnProgress.paragraphBreakPending = false;
    turnProgress.hasForwardedText = false;
    turnProgress.toolSteps.beginNextAnswer();
    request.onQueuedMessageAnswered(queuedMessageID);
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
