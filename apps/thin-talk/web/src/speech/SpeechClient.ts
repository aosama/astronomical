import { SpeechFetchFailure } from "./SpeechFetchFailure";
import { SpeechFetchFailureKind } from "./SpeechFetchFailureKind";

/** A minute of 24 kHz 16-bit mono WAV is under 3 MB, and a piece is at most
 * about that long (measured: 1,130 characters made 68.7 s of Kokoro audio), so
 * 16 MB is room to spare. The daemon is the local engine and states the size
 * up front, so the declared length is refused before the body is read; a
 * response that does not declare one is refused once read, so nothing over the
 * cap is ever decoded. */
const MAXIMUM_SPEECH_RESPONSE_BYTES = 16 * 1024 * 1024;

/** What one piece asks for. Without a model the daemon picks: the resident
 * voice model, else its built-in default (Kokoro), else the only one. */
export interface SpeechPieceRequest {
  text: string;
  seed: number;
  model: string | null;
  signal: AbortSignal;
}

/**
 * Fetches spoken audio for one piece of text from the daemon's
 * `POST /v1/audio/speech`, as WAV bytes.
 *
 * It sends only `content-type`, like every request from the canvas: the
 * daemon's CORS rule for the canvas origin admits no other header. The same
 * seed goes with every piece of one reply, so a sampling voice model keeps
 * one voice from the first sentence to the last.
 */
export class SpeechClient {
  private readonly daemonBaseURL: string;

  public constructor(daemonBaseURL: string) {
    this.daemonBaseURL = daemonBaseURL;
  }

  public async fetchSpokenPiece(pieceRequest: SpeechPieceRequest): Promise<ArrayBuffer> {
    const requestBody: Record<string, unknown> = { input: pieceRequest.text, seed: pieceRequest.seed, response_format: "wav" };
    if (pieceRequest.model !== null) {
      requestBody["model"] = pieceRequest.model;
    }
    const speechReply = await fetch(`${this.daemonBaseURL}/v1/audio/speech`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(requestBody),
      signal: pieceRequest.signal,
    });
    if (!speechReply.ok) {
      throw await SpeechClient.classifiedFailure(speechReply);
    }
    const declaredByteCount = Number(speechReply.headers.get("content-length"));
    if (declaredByteCount > MAXIMUM_SPEECH_RESPONSE_BYTES) {
      throw new SpeechFetchFailure(SpeechFetchFailureKind.FAILED, `the spoken audio declared ${declaredByteCount} bytes`);
    }
    const audioBytes = await speechReply.arrayBuffer();
    if (audioBytes.byteLength > MAXIMUM_SPEECH_RESPONSE_BYTES) {
      throw new SpeechFetchFailure(SpeechFetchFailureKind.FAILED, `the spoken audio was ${audioBytes.byteLength} bytes`);
    }
    return audioBytes;
  }

  /** The installed voice models' ids, from the daemon's model list. */
  public async listVoiceModels(signal?: AbortSignal): Promise<string[]> {
    const modelListReply = await fetch(`${this.daemonBaseURL}/v1/models`, { signal });
    if (!modelListReply.ok) {
      return [];
    }
    const modelList: unknown = await modelListReply.json();
    const listedModels = typeof modelList === "object" && modelList !== null ? (modelList as Record<string, unknown>)["data"] : null;
    if (!Array.isArray(listedModels)) {
      return [];
    }
    const voiceModelIDs: string[] = [];
    for (const listedModel of listedModels as unknown[]) {
      const modelRecord = typeof listedModel === "object" && listedModel !== null ? (listedModel as Record<string, unknown>) : {};
      const supportedEndpoints = modelRecord["supported_endpoints"];
      if (Array.isArray(supportedEndpoints) && supportedEndpoints.includes("/v1/audio/speech") && typeof modelRecord["id"] === "string") {
        voiceModelIDs.push(modelRecord["id"]);
      }
    }
    return voiceModelIDs;
  }

  private static async classifiedFailure(failedReply: Response): Promise<SpeechFetchFailure> {
    const errorCode = await SpeechClient.errorCodeOf(failedReply);
    if (errorCode === "no_speech_model") {
      return new SpeechFetchFailure(SpeechFetchFailureKind.NO_VOICE, "no voice model is installed");
    }
    if (errorCode === "speech_model_ambiguous") {
      return new SpeechFetchFailure(SpeechFetchFailureKind.AMBIGUOUS_VOICE, "several voice models are installed");
    }
    // 429: the engine is answering another request; 503: it is loading or
    // restarting. Both clear up on their own.
    if (failedReply.status === 429 || failedReply.status === 503) {
      return new SpeechFetchFailure(SpeechFetchFailureKind.BUSY, `the engine answered ${failedReply.status}`);
    }
    return new SpeechFetchFailure(SpeechFetchFailureKind.FAILED, `the engine answered ${failedReply.status} ${errorCode}`);
  }

  private static async errorCodeOf(failedReply: Response): Promise<string> {
    try {
      const errorDocument: unknown = await failedReply.json();
      const errorRecord = typeof errorDocument === "object" && errorDocument !== null ? (errorDocument as Record<string, unknown>)["error"] : null;
      const errorCode = typeof errorRecord === "object" && errorRecord !== null ? (errorRecord as Record<string, unknown>)["code"] : null;
      return typeof errorCode === "string" ? errorCode : "";
    } catch {
      return "";
    }
  }
}
