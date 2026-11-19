// /astronomical-toggle: a two-choice dialog that writes the shared toggle
// file; the server plugin applies it on its next refresh.

import type { Context as TuiPluginContext } from "@opencode/plugin/tui/context";
import { recordAuditEvent, type AstronomicalAuditLogger } from "./audit-log.js";
import { ASTRONOMICAL_PROVIDER_NAME } from "./constants.js";
import { describeError } from "./error-message.js";

type AstronomicalToggleChoice = "enabled" | "disabled";

export interface ToggleCommandInput {
  readonly readEnabled: () => Promise<boolean>;
  readonly writeEnabled: (isEnabled: boolean) => Promise<void>;
  readonly audit: AstronomicalAuditLogger;
}

async function askForToggleChoice(
  context: TuiPluginContext,
  isCurrentlyEnabled: boolean,
): Promise<AstronomicalToggleChoice | undefined> {
  return context.ui.dialog.select<AstronomicalToggleChoice>({
    title: "Astronomical provider",
    current: isCurrentlyEnabled ? "enabled" : "disabled",
    options: [
      {
        title: "Enabled",
        value: "enabled",
        description: "Discover and offer the live chat models.",
      },
      {
        title: "Disabled",
        value: "disabled",
        description: "Remove Astronomical models from the picker.",
      },
    ],
  });
}

/** Lets the user switch the provider on or off; a cancelled dialog changes nothing. */
export async function showAstronomicalToggle(
  context: TuiPluginContext,
  input: ToggleCommandInput,
): Promise<void> {
  try {
    const choice = await askForToggleChoice(context, await input.readEnabled());
    if (choice === undefined) return;
    const isEnabled = choice === "enabled";
    await input.writeEnabled(isEnabled);
    await recordAuditEvent(input.audit, "toggle-command", {
      enabled: isEnabled,
    });
    context.ui.toast.show({
      variant: "success",
      title: ASTRONOMICAL_PROVIDER_NAME,
      message: isEnabled
        ? "Astronomical is enabled. The server will refresh its models shortly."
        : "Astronomical is disabled. The server will remove its models shortly.",
    });
  } catch (cause: unknown) {
    const reason = describeError(cause);
    await recordAuditEvent(input.audit, "diagnostic", {
      stage: "toggle-command",
      reason,
    });
    context.ui.toast.show({
      variant: "error",
      title: ASTRONOMICAL_PROVIDER_NAME,
      message: `Could not update the provider toggle: ${reason}`,
    });
  }
}
