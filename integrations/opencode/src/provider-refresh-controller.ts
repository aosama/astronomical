// Keeps OpenCode's provider list in step with what discovery finds. OpenCode
// reads the definitions through one registered transform; when a refresh finds
// different definitions, the controller asks OpenCode to reload, which reruns
// that transform over the new ones.
//
// The refresh carries state across runs (what is registered, what was last
// seen, whether a run is in flight), so it is a class: that state has one
// owner, and each decision (reload or not, what to record) is a named method
// that can be read and tested on its own.

import type { Plugin } from "@opencode/plugin";
import type { ProviderEditor } from "@opencode/plugin/promise/provider";
import { recordAuditEvent, type AstronomicalAuditLogger } from "./audit-log.ts";
import { describeError } from "./error-message.ts";
import type { AstronomicalProviderDefinition } from "./provider-definition.ts";
import {
  describeProviderState,
  type ProviderRefreshState,
} from "./provider-discovery.ts";

// Comparing serialized definitions is how the controller tells a real change
// (a model added, a channel gone) from a refresh that found the same thing,
// so an unchanged list never makes OpenCode reload.
function definitionFingerprint(
  definitions: readonly AstronomicalProviderDefinition[],
): string | undefined {
  return definitions.length > 0 ? JSON.stringify(definitions) : undefined;
}

export class ProviderRefreshController {
  private currentDefinitions: readonly AstronomicalProviderDefinition[] = [];
  private currentFingerprint: string | undefined;
  private registeredFingerprint: string | undefined;
  private currentState: ProviderRefreshState | undefined;
  private isTransformRegistered = false;
  private isRefreshInProgress = false;
  private isDisposed = false;

  // Plain fields, not constructor parameter properties: OpenCode and Node load
  // these sources by stripping types, which cannot rewrite parameter
  // properties into assignments.
  private readonly context: Plugin.Context;
  private readonly discover: () => Promise<ProviderRefreshState>;
  private readonly audit: AstronomicalAuditLogger;

  constructor(
    context: Plugin.Context,
    discover: () => Promise<ProviderRefreshState>,
    audit: AstronomicalAuditLogger,
  ) {
    this.context = context;
    this.discover = discover;
    this.audit = audit;
  }

  /** Runs one discovery pass; overlapping and post-dispose calls are no-ops. */
  async refresh(): Promise<void> {
    if (this.isRefreshInProgress || this.isDisposed) return;
    this.isRefreshInProgress = true;
    try {
      const nextState = await this.discover();
      if (!this.isDisposed) await this.applyState(nextState);
    } catch (cause: unknown) {
      await recordAuditEvent(this.audit, "diagnostic", {
        stage: "provider-refresh",
        reason: describeError(cause),
      });
    } finally {
      this.isRefreshInProgress = false;
    }
  }

  /** Registers the transform OpenCode reads the definitions through. */
  async registerTransform(): Promise<void> {
    await this.context.provider.transform((editor: ProviderEditor): void =>
      this.addCurrentDefinitions(editor),
    );
    this.isTransformRegistered = true;
    this.registeredFingerprint = this.currentFingerprint;
    if (this.currentState?.kind === "ready") {
      await recordAuditEvent(this.audit, "provider-registered", {
        channels: this.currentState.channels,
      });
    }
  }

  dispose(): void {
    this.isDisposed = true;
  }

  // One editor.add per channel: each definition carries its own provider id
  // and baseURL, so Stable and Development coexist in the picker.
  private addCurrentDefinitions(editor: ProviderEditor): void {
    for (const definition of this.currentDefinitions) editor.add(definition);
  }

  private async applyState(nextState: ProviderRefreshState): Promise<void> {
    const nextDefinitions =
      nextState.kind === "ready" ? nextState.definitions : [];
    const previousState = this.currentState;
    this.currentDefinitions = nextDefinitions;
    this.currentFingerprint = definitionFingerprint(nextDefinitions);
    await this.reloadIfDefinitionsChanged(nextState);
    if (nextDefinitions.length === 0) {
      await this.recordStateWithoutProvider(previousState, nextState);
    }
    this.currentState = nextState;
  }

  private async reloadIfDefinitionsChanged(
    nextState: ProviderRefreshState,
  ): Promise<void> {
    if (!this.isTransformRegistered) return;
    if (this.currentFingerprint === this.registeredFingerprint) return;
    const previousRegisteredFingerprint = this.registeredFingerprint;
    await this.context.provider.reload();
    this.registeredFingerprint = this.currentFingerprint;
    if (nextState.kind === "ready") {
      const event =
        previousRegisteredFingerprint === undefined
          ? "provider-registered"
          : "provider-refreshed";
      await recordAuditEvent(this.audit, event, {
        channels: nextState.channels,
      });
    } else if (previousRegisteredFingerprint !== undefined) {
      await recordAuditEvent(this.audit, "provider-removed", {
        reason: describeProviderState(nextState),
      });
    }
  }

  // Recorded once per change of reason, so a channel that stays down does not
  // write a line every refresh.
  private async recordStateWithoutProvider(
    previousState: ProviderRefreshState | undefined,
    nextState: ProviderRefreshState,
  ): Promise<void> {
    if (previousState?.kind === nextState.kind) return;
    const detail = { reason: describeProviderState(nextState) };
    if (previousState === undefined) {
      await recordAuditEvent(this.audit, "setup-skipped", detail);
    } else {
      await recordAuditEvent(this.audit, "diagnostic", detail);
    }
  }
}
