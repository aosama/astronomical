import Foundation

/**
 * Whether just-read experts may stay in the retained cache.
 *
 * Two caching strategies answer "may this page stay" here: whole-layer
 * caching seats an entire decoder layer's experts or none of them, and
 * hot-expert caching keeps the individual routed experts the router keeps
 * choosing. In hot-expert caching, *hot* means routing frequency, not
 * temperature. Execution families ask this policy; they never invent a
 * second answer.
 */
public enum ExpertPageCommitPolicy {

    /**
     * Seats a complete layer after a mandatory read when the plan asked for it.
     *
     * Prefill does this for multi-token chunks. Decode must do it too after
     * an atomic complete-owner demote, or leftover budget never becomes
     * resident RAM and every generated token streams from SSD.
     *
     * - Parameters:
     *   - routeTokenCount: Token count whose routes demanded the read
     *     (reserved for route-count gating; paging admission is the active gate).
     *   - productionDefaultPaging: Whether production expert paging is enabled.
     *   - residencyTarget: The active plan's target for the layer.
     * - Returns: Whether the complete layer may be committed to retention.
     */
    public static func shouldCommitMandatoryCompleteLayer(
        routeTokenCount: Int32,
        productionDefaultPaging: Bool,
        residencyTarget: ExpertLayerResidencyTarget?
    ) -> Bool {
        guard productionDefaultPaging else {
            return false
        }
        return residencyTarget == .promoteCompleteOnMandatoryRead
    }

    /**
     * Keeps routed experts after a mandatory read so the next forward can
     * hit them (hot-expert caching).
     *
     * Overflow is handled by evicting the least-used retained page, not by
     * refusing to cache what this forward used. The active residency plan is
     * the phase's will for the layer, so the commit honors it: a layer the
     * plan streams operation-local keeps streaming, and a layer the plan is
     * releasing is not refilled behind the plan's back.
     *
     * - Parameters:
     *   - routeTokenCount: Token count whose routes demanded the read
     *     (reserved for route-count gating; paging admission is the active gate).
     *   - productionDefaultPaging: Whether production expert paging is enabled.
     *   - residencyTarget: The active plan's target for the layer.
     *   - layerHasNoRetainedPage: Whether the layer currently retains nothing.
     * - Returns: Whether the routed pages may be committed to retention.
     */
    public static func shouldCommitMandatoryRoutedPage(
        routeTokenCount: Int32,
        productionDefaultPaging: Bool,
        residencyTarget: ExpertLayerResidencyTarget?,
        layerHasNoRetainedPage: Bool
    ) -> Bool {
        guard productionDefaultPaging else {
            return false
        }
        switch residencyTarget {
        case .streamOperationLocal, .releasePartial, .releaseCompleteForExactDeficit:
            return false
        default:
            return true
        }
    }

    /**
     * Structural slot cap for one decode warm table (hot-expert caching).
     *
     * Issue #514: a history-length multiplier (`expertsPerToken * 8`) left
     * leftover decode entitlement unclaimed on every machine, because the
     * cap did not depend on leftover RAM. The structural ceiling is the
     * layer's own expert count — a table that fills becomes a complete
     * layer, which the complete-layer residency machinery already
     * understands. The economic ceiling is leftover decode entitlement,
     * applied by the caller as the budget-affordable minimum of this count.
     * Least-frequently-used eviction still drops one-off routing noise;
     * budget admission still refuses a table the machine cannot hold.
     *
     * - Parameter expertCapacity: The layer's total expert count.
     * - Returns: The structural warm-table slot cap.
     */
    public static func hotExpertWarmSlotCount(expertCapacity: Int) -> Int {
        return expertCapacity
    }
}
