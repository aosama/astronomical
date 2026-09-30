/**
 * How much thinking the model may spend before it must answer.
 *
 * The levels are the user-facing promise and the token counts are what the
 * supervisor enforces, so the number shown next to a level is the budget the
 * server applies to that turn. Quick is the default because a first answer
 * should feel immediate; a reader who wants depth asks for it explicitly.
 */
export enum ThinkingEffort {
  QUICK = "quick",
  BALANCED = "balanced",
  HIGH = "high",
}

export namespace ThinkingEffort {
  /** The effort a fresh install starts on. */
  export const DEFAULT: ThinkingEffort = ThinkingEffort.QUICK;

  /** Every level, in the order the effort menu lists them. */
  export const ALL: readonly ThinkingEffort[] = [
    ThinkingEffort.QUICK,
    ThinkingEffort.BALANCED,
    ThinkingEffort.HIGH,
  ];

  /** Parses a stored value, falling back to the default for anything unknown. */
  export function parse(stored: string | null | undefined): ThinkingEffort {
    if (stored === ThinkingEffort.QUICK || stored === ThinkingEffort.BALANCED || stored === ThinkingEffort.HIGH) {
      return stored;
    }
    return ThinkingEffort.DEFAULT;
  }

  /** The thinking-token budget the supervisor enforces for this level. */
  export function thinkingBudgetTokens(effort: ThinkingEffort): number {
    switch (effort) {
      case ThinkingEffort.QUICK:
        return 256;
      case ThinkingEffort.BALANCED:
        return 512;
      case ThinkingEffort.HIGH:
        return 1024;
    }
  }

  export function displayName(effort: ThinkingEffort): string {
    switch (effort) {
      case ThinkingEffort.QUICK:
        return "Quick";
      case ThinkingEffort.BALANCED:
        return "Balanced";
      case ThinkingEffort.HIGH:
        return "High";
    }
  }

  /** One line for the control itself, so the cost of depth is never hidden. */
  export function budgetSummary(effort: ThinkingEffort): string {
    return `${displayName(effort)} · ${thinkingBudgetTokens(effort)} tokens`;
  }
}
