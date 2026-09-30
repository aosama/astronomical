/**
 * Ambient declarations for the surfaces this page touches that no npm package
 * types: the vendored KaTeX and mermaid globals, the WKWebView message-handler
 * bridge, the host-injected config, and the session-call resolver the Swift
 * host invokes.
 */
export {};

declare global {
  /** The subset of the vendored KaTeX global the maths pipeline uses. */
  interface KatexGlobal {
    renderToString(tex: string, options?: KatexRenderOptions): string;
  }

  interface KatexRenderOptions {
    displayMode: boolean;
    output: "htmlAndMathml" | "html" | "mathml";
    strict: "ignore" | "warn" | "error";
    trust: boolean;
    throwOnError: boolean;
    maxSize: number;
  }

  /** The subset of the vendored mermaid global the diagram pipeline uses. */
  interface MermaidGlobal {
    initialize(config: MermaidInitConfig): void;
    render(id: string, source: string): Promise<MermaidRenderResult>;
  }

  interface MermaidInitConfig {
    startOnLoad: boolean;
    securityLevel: "strict";
    theme: string;
    htmlLabels: boolean;
    flowchart: { htmlLabels: boolean };
    logLevel: number;
  }

  interface MermaidRenderResult {
    svg: string;
  }

  /** The session-call resolver the Swift host invokes to answer a bridge call. */
  interface ThintalkBridgeGlobal {
    resolve(callId: string, payload: unknown): void;
    reject(callId: string, message: string): void;
  }

  /** The config the Swift host injects via a document-start user script. */
  interface ThintalkConfigGlobal {
    supervisorBaseURL: string;
    expectedChannel: string;
    expectedStateDirectory: string;
  }

  /** The test entry point AppShell installs for the harness. */
  interface ThintalkTestEntry {
    messageCount(): number;
    renderedText(messageID: string): string;
    diagnostics(): Record<string, unknown>;
  }

  interface Window {
    katex?: KatexGlobal;
    mermaid?: MermaidGlobal;
    __thintalkBridge?: ThintalkBridgeGlobal;
    __thintalkConfig?: ThintalkConfigGlobal;
    __thintalk?: ThintalkTestEntry;
    webkit?: {
      messageHandlers?: {
        thintalk?: {
          postMessage(payload: unknown): void;
        };
      };
    };
  }
}
