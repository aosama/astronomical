/**
 * The lifecycle state of one message as the transcript renders it. A message is
 * streaming while deltas still arrive, and settles into a terminal state when
 * the turn ends.
 */
export enum ChatMessageState {
  COMPLETE = "complete",
  STREAMING = "streaming",
  STOPPED = "stopped",
  FAILED = "failed",
}
