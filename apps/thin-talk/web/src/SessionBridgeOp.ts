/**
 * The session-file operations the page may ask the Swift host to perform. The
 * page never touches the filesystem directly (the web view uses a
 * non-persistent data store), so every durable read and write crosses this
 * bridge.
 */
export enum SessionBridgeOp {
  LIST = "list",
  LOAD = "load",
  SAVE = "save",
  DELETE = "delete",
  RENAME = "rename",
  LOAD_PREFS = "loadPrefs",
  SAVE_PREFS = "savePrefs",
  STORE_GENERATED_IMAGE = "storeGeneratedImage",
  DELETE_GENERATED_IMAGES = "deleteGeneratedImages",
  USE_HOST_TOOL = "useHostTool",
  CANCEL_HOST_TOOL_CALL = "cancelHostToolCall",
}
