/** The daemon's single Library download, as `/v1/library/download` reports it. */
export interface LibraryDownloadStatus {
  /** idle, checking_disk, fetching_manifest, downloading, paused, verifying,
   * publishing, failed, or resuming. */
  readonly state: string;
  readonly huggingfaceID: string | null;
  readonly bytesCompleted: number;
  readonly bytesTotal: number;
  readonly errorCode: string | null;
}

export const IDLE_DOWNLOAD_STATUS: LibraryDownloadStatus = {
  state: "idle",
  huggingfaceID: null,
  bytesCompleted: 0,
  bytesTotal: 0,
  errorCode: null,
};

/** The download is held by another model and Astro must wait its turn. */
export const LIBRARY_BUSY_ERROR_CODE = "library_busy";

/** Reads a download document; anything unreadable is treated as idle, the
 * state a poll can always recover from on its next read. */
export function parseLibraryDownloadStatus(downloadBodyText: string): LibraryDownloadStatus {
  let downloadDocument: unknown;
  try {
    downloadDocument = JSON.parse(downloadBodyText);
  } catch {
    return IDLE_DOWNLOAD_STATUS;
  }
  if (typeof downloadDocument !== "object" || downloadDocument === null) {
    return IDLE_DOWNLOAD_STATUS;
  }
  const downloadRecord = downloadDocument as Record<string, unknown>;
  const state = downloadRecord["state"];
  const huggingfaceID = downloadRecord["huggingface_id"];
  const bytesCompleted = downloadRecord["bytes_completed"];
  const bytesTotal = downloadRecord["bytes_total"];
  const errorCode = downloadRecord["error_code"];
  return {
    state: typeof state === "string" ? state : "idle",
    huggingfaceID: typeof huggingfaceID === "string" ? huggingfaceID : null,
    bytesCompleted: typeof bytesCompleted === "number" && bytesCompleted >= 0 ? bytesCompleted : 0,
    bytesTotal: typeof bytesTotal === "number" && bytesTotal >= 0 ? bytesTotal : 0,
    errorCode: typeof errorCode === "string" ? errorCode : null,
  };
}

/** The error code from a refused download request (`{"error": {"code": …}}`),
 * or null when the body carries none. */
export function refusalCodeFromErrorDocument(errorBodyText: string): string | null {
  try {
    const errorDocument = JSON.parse(errorBodyText) as Record<string, unknown>;
    const errorBody = errorDocument["error"] as Record<string, unknown> | undefined;
    const refusalCode = errorBody != null ? errorBody["code"] : null;
    return typeof refusalCode === "string" ? refusalCode : null;
  } catch {
    return null;
  }
}
