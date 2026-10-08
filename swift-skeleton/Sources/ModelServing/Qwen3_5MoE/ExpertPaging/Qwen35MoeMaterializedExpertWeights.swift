import Foundation

/**
 * The quantized expert weight slices one paged layer forward consumes:
 * one slice set per SwitchGLU projection plus the page-read evidence the
 * decorator accumulates for telemetry and the issue #629 read-once
 * property. A slice set carries only the requested experts — the
 * materializer must never return unrequested expert payloads.
 */
public struct Qwen35MoeMaterializedExpertWeights {

    /// Materialized `gate_proj` slices for the requested experts.
    public var gateProjection: Qwen35MoeMaterializedProjectionSlices

    /// Materialized `up_proj` slices for the requested experts.
    public var upProjection: Qwen35MoeMaterializedProjectionSlices

    /// Materialized `down_proj` slices for the requested experts.
    public var downProjection: Qwen35MoeMaterializedProjectionSlices

    /// How many expert pages the backing store read to serve the request.
    public var expertPageReadCount: Int

    public init(
        gateProjection: Qwen35MoeMaterializedProjectionSlices,
        upProjection: Qwen35MoeMaterializedProjectionSlices,
        downProjection: Qwen35MoeMaterializedProjectionSlices,
        expertPageReadCount: Int
    ) {
        self.gateProjection = gateProjection
        self.upProjection = upProjection
        self.downProjection = downProjection
        self.expertPageReadCount = expertPageReadCount
    }
}
