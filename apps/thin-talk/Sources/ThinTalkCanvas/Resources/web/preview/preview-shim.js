"use strict";
(() => {
  // src/SessionBridgeError.ts
  var SessionBridgeError = class extends Error {
    constructor(message, options) {
      super(message, options);
      this.name = "SessionBridgeError";
    }
  };

  // src/preview/handleSessionCall.ts
  function stringField(payload, key) {
    if (typeof payload !== "object" || payload === null) {
      throw new SessionBridgeError(`payload for "${key}" must be an object`);
    }
    const value = payload[key];
    if (typeof value !== "string") {
      throw new SessionBridgeError(`payload field "${key}" must be a string`);
    }
    return value;
  }
  function normalizeSave(payload) {
    if (typeof payload !== "object" || payload === null) {
      throw new SessionBridgeError("save payload must be an object");
    }
    const record = payload;
    return {
      id: stringField(record, "id"),
      title: typeof record["title"] === "string" ? record["title"] : "",
      updatedAt: typeof record["updatedAt"] === "string" ? record["updatedAt"] : "",
      messages: Array.isArray(record["messages"]) ? record["messages"] : []
    };
  }
  function handleSessionCall(op, payload, adapter) {
    switch (op) {
      case "list" /* LIST */:
        return adapter.listSessions();
      case "load" /* LOAD */:
        return adapter.loadSession(stringField(payload, "id"));
      case "save" /* SAVE */: {
        adapter.saveSession(normalizeSave(payload));
        return null;
      }
      case "delete" /* DELETE */:
        adapter.deleteSession(stringField(payload, "id"));
        return null;
      case "rename" /* RENAME */:
        adapter.renameSession(stringField(payload, "id"), stringField(payload, "title"));
        return null;
      case "loadPrefs" /* LOAD_PREFS */:
        return adapter.loadPreferences();
      case "savePrefs" /* SAVE_PREFS */:
        adapter.savePreferences(payload);
        return null;
      default:
        throw new SessionBridgeError(`host does not know op "${op}"`);
    }
  }

  // src/preview/previewShim.ts
  var SESSIONS_KEY = "thintalk.preview.sessions";
  var PREFERENCES_KEY = "thintalk.preview.preferences";
  function makeStorageAdapter(storage) {
    const readSessions = () => storage.getItem(SESSIONS_KEY) ? JSON.parse(storage.getItem(SESSIONS_KEY)) : {};
    const writeSessions = (sessions) => {
      if (Object.keys(sessions).length === 0) {
        storage.removeItem(SESSIONS_KEY);
      } else {
        storage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
      }
    };
    return {
      listSessions: () => Object.values(readSessions()).map((record) => ({ ...record })).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
      loadSession: (id) => {
        const record = readSessions()[id];
        return record ? { ...record } : null;
      },
      saveSession: (record) => {
        const sessions = readSessions();
        sessions[record.id] = { ...record };
        writeSessions(sessions);
      },
      deleteSession: (id) => {
        const sessions = readSessions();
        delete sessions[id];
        writeSessions(sessions);
      },
      renameSession: (id, title) => {
        const sessions = readSessions();
        if (sessions[id]) {
          sessions[id] = { ...sessions[id], title };
          writeSessions(sessions);
        }
      },
      loadPreferences: () => {
        return storage.getItem(PREFERENCES_KEY) ? JSON.parse(storage.getItem(PREFERENCES_KEY)) : {};
      },
      savePreferences: (value) => {
        storage.setItem(PREFERENCES_KEY, JSON.stringify(value));
      }
    };
  }
  function installPreviewShim(win, storage) {
    const adapter = makeStorageAdapter(storage);
    const postMessage = (payload) => {
      const envelope = payload;
      if (envelope.kind !== "sessionCall" || !envelope.callId || !envelope.op) {
        return;
      }
      try {
        const reply = handleSessionCall(envelope.op, envelope.payload ?? null, adapter);
        win.__thintalkBridge?.resolve?.(envelope.callId, reply);
      } catch (error) {
        win.__thintalkBridge?.reject?.(envelope.callId, error instanceof Error ? error.message : String(error));
      }
    };
    win.webkit = win.webkit || {};
    win.webkit.messageHandlers = win.webkit.messageHandlers || {};
    win.webkit.messageHandlers.thintalk = { postMessage };
  }

  // src/preview/previewShimEntry.ts
  installPreviewShim(window, window.localStorage);
})();
