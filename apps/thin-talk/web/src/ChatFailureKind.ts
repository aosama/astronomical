/**
 * Every failure the conversation can hit, derived from the supervisor's public
 * REST error signals. This is the client-facing view of the backend code, so the
 * client can map each failure to a concrete next action without knowing how the
 * response was decoded.
 */
export enum ChatFailureKind {
  /** No usable chat-capable Library model is discoverable. */
  NO_USABLE_MODEL = "noUsableModel",
  /** The selected model will not load, including the memory-ceiling refusal. */
  MODEL_LOAD_FAILED = "modelLoadFailed",
  /** The supervisor/daemon is unavailable or not ready to serve requests. */
  WORKER_UNAVAILABLE = "workerUnavailable",
  /** Another request already owns the worker's bounded capacity. */
  ENGINE_BUSY = "engineBusy",
  /** The request was malformed, including an over-length ask. */
  INVALID_REQUEST = "invalidRequest",
  /** The model's output could not be parsed into the declared output contract. */
  MALFORMED_MODEL_OUTPUT = "malformedModelOutput",
  /** No stream token arrived within the client watchdog timeout (a stall). */
  STALL = "stall",
  /** Anything the client could not classify into the categories above. */
  UNKNOWN = "unknown",
}
