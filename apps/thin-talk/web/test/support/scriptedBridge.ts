import { SessionBridge } from "../../src/SessionBridge";
import type { SessionBridgeRequest } from "../../src/SessionBridgeRequest";

/**
 * A session bridge whose host answers each op from a table, recording every
 * request; for tool tests that need the bridge but not the whole fake host.
 * An op missing from the table is rejected, so an unexpected call fails.
 * Answers arrive on the next task, so a test can stop a call before then.
 */
export function scriptedBridge(answersByOp: Record<string, unknown>, sentRequests: SessionBridgeRequest[] = []): SessionBridge {
  return new SessionBridge((request) => {
    sentRequests.push(request);
    setTimeout(() => {
      if (request.op in answersByOp) {
        window.__thintalkBridge?.resolve(request.callId, answersByOp[request.op]);
      } else {
        window.__thintalkBridge?.reject(request.callId, `no scripted answer for ${request.op}`);
      }
    }, 0);
  });
}
