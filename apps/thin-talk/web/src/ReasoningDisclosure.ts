/**
 * Stable identity and open-state rules for one reasoning disclosure.
 *
 * The renderer stamps every reasoning block with this id so the transcript can
 * find it again after each streamed patch, and the transcript records the
 * reader's expand/collapse choice per message id. Both layers share this one
 * naming rule so the accessibility tree and selector-based agents agree, and so
 * a rename cannot silently break the reconciliation that keeps a manual
 * expansion from being clobbered by the next delta.
 */
export function reasoningDisclosureIdentifier(messageID: string): string {
  return `reasoning:${messageID}`;
}
