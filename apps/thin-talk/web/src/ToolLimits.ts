import type { ConversationTool } from "./ConversationTool";

/** Most tool rounds one assistant turn may run. A model that keeps calling
 * tools without answering is stuck, and each round costs a full model request.
 * Past the limit further calls are refused with TOOL_ROUND_LIMIT_REASON, so
 * the model is told to answer with what it has instead of the turn ending
 * silently. */
export const MAXIMUM_TOOL_ROUNDS_PER_TURN = 8;
/** What a refused call tells the model once the round limit is reached. */
export const TOOL_ROUND_LIMIT_REASON =
  "The tool limit for this answer has been reached. Do not call any more tools; answer with the information you already have.";
/** How long one page or host tool may run before it is reported as failed. */
export const TOOL_TIMEOUT_MILLISECONDS = 10_000;
/** The most result text one tool call hands back to the model, so a single
 * large result cannot crowd the conversation out of the context window. */
export const MAXIMUM_TOOL_RESULT_CHARACTERS = 16 * 1024;

/**
 * Runs one tool under the shared limits: it is stopped by the turn's own stop
 * signal or by the timeout, whichever comes first, and its result is capped.
 * Kept apart from the agent loop so the limits are tested on their own.
 */
export async function executeWithinLimits(
  tool: ConversationTool,
  toolArguments: Record<string, unknown>,
  turnSignal: AbortSignal,
  timeoutMilliseconds: number = TOOL_TIMEOUT_MILLISECONDS,
): Promise<string> {
  // A tool is never started after the user stopped the turn, even one that
  // would finish at once.
  if (turnSignal.aborted) {
    throw new Error(`${tool.label} was stopped`);
  }
  const timeoutSignal = AbortSignal.timeout(timeoutMilliseconds);
  const toolSignal = AbortSignal.any([turnSignal, timeoutSignal]);
  // Ends the abort watcher once the tool settles, so a finished tool leaves no
  // listener behind on the turn's signal.
  const watcherCleanup = new AbortController();
  try {
    const resultText = await Promise.race([
      tool.execute(toolArguments, toolSignal),
      rejectWhenAborted(toolSignal, tool.label, timeoutSignal, watcherCleanup.signal),
    ]);
    return capToolResult(resultText);
  } finally {
    watcherCleanup.abort();
  }
}

/** Cuts a result to the cap and says so, so the model knows it saw a part. */
export function capToolResult(resultText: string): string {
  if (resultText.length <= MAXIMUM_TOOL_RESULT_CHARACTERS) {
    return resultText;
  }
  const omittedCharacterCount = resultText.length - MAXIMUM_TOOL_RESULT_CHARACTERS;
  return `${resultText.slice(0, MAXIMUM_TOOL_RESULT_CHARACTERS)}\n[result cut: ${omittedCharacterCount} more characters not shown]`;
}

/** A tool that ignores its signal must still not hold the turn open, so the
 * race rejects as soon as the signal fires. */
function rejectWhenAborted(
  toolSignal: AbortSignal,
  toolLabel: string,
  timeoutSignal: AbortSignal,
  watcherCleanupSignal: AbortSignal,
): Promise<never> {
  return new Promise<never>((_resolve, reject) => {
    const rejectWithReason = (): void => {
      const reason = timeoutSignal.aborted ? `${toolLabel} took too long and was stopped` : `${toolLabel} was stopped`;
      reject(new Error(reason));
    };
    if (toolSignal.aborted) {
      rejectWithReason();
      return;
    }
    toolSignal.addEventListener("abort", rejectWithReason, { once: true, signal: watcherCleanupSignal });
  });
}
