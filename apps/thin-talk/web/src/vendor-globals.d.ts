/**
 * Ambient declarations for the surfaces this page touches that no npm package
 * types: the vendored KaTeX and mermaid globals, the WKWebView message-handler
 * bridge, the host-injected config, and the session-call resolver the Swift
 * host invokes, and the export feature the separately loaded export script
 * registers.
 */
import type { ExportFeature } from "./export/ExportEnvironment";

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
    daemonBaseURL: string;
    expectedChannel: string;
    expectedStateDirectory: string;
    /** Points of native title bar and toolbar drawn over the top of the page;
     * absent for hosts with an ordinary title bar. */
    titlebarInset?: number;
  }

  /** The entry point the native menu bar and toolbar call (HostCommandRouter). */
  interface ThintalkCommandEntry {
    run(payload: unknown): boolean;
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
    __thintalkCommands?: ThintalkCommandEntry;
    /** The export feature, registered by export.js once the page loads it. */
    __thintalkExport?: ExportFeature;
    webkit?: {
      messageHandlers?: {
        thintalk?: {
          postMessage(payload: unknown): void;
        };
      };
    };
  }
}
