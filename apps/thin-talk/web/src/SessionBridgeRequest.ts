import { SessionBridgeOp } from "./SessionBridgeOp";

/**
 * One page→host session call. `callId` correlates the asynchronous reply with
 * the pending promise; `payload` is the op's argument (null for ops that take
 * none) and is always plain JSON-serializable data.
 */
export class SessionBridgeRequest {
  public readonly kind: "sessionCall";
  public readonly callId: string;
  public readonly op: SessionBridgeOp;
  public readonly payload: unknown;

  public constructor(init: { callId: string; op: SessionBridgeOp; payload: unknown }) {
    this.kind = "sessionCall";
    this.callId = init.callId;
    this.op = init.op;
    this.payload = init.payload;
  }
}
