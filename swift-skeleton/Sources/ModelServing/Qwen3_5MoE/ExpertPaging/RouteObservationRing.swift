import Foundation

/// Bounded first-in-first-out history of true decode-time expert routes
/// captured for the on-device route predictor program, port of the Rust
/// `RouteObservationRing`. Growth stops at capacity and the oldest
/// observation is evicted to admit the newest; stored and evicted totals
/// keep capture cost and turnover measurable.
public final class RouteObservationRing {

    /// One request typically contributes hundreds of decode tokens; 2,048
    /// observations keep the most recent several conversations resident for
    /// training while staying a few megabytes even for large layer counts.
    public static let defaultObservationCapacity = 2_048

    private var retainedObservations: [RouteObservationRecord]

    private let retentionCapacity: Int

    private var storedObservationTotal: UInt64

    private var evictedObservationTotal: UInt64

    /// Creates a ring that retains the most recent `capacity` observations.
    public init(capacity: Int) {
        self.retainedObservations = []
        self.retentionCapacity = max(capacity, 1)
        self.storedObservationTotal = 0
        self.evictedObservationTotal = 0
    }

    /// Stores one observation, evicting the oldest when the ring is full.
    /// Returns `true` when an older observation was evicted to make room.
    @discardableResult
    public func recordObservation(_ observation: RouteObservationRecord) -> Bool {
        let evictedOldest = retainedObservations.count >= retentionCapacity
        if evictedOldest {
            retainedObservations.removeFirst()
            evictedObservationTotal += 1
        }
        retainedObservations.append(observation)
        storedObservationTotal += 1
        return evictedOldest
    }

    /// Observations currently retained, oldest first.
    public func observations() -> [RouteObservationRecord] {
        return retainedObservations
    }

    /// Removes and returns the oldest retained observation, if any. The
    /// predictor trainer consumes history this way so a training slice
    /// always spends its budget on the oldest examples first.
    public func popOldestObservation() -> RouteObservationRecord? {
        guard retainedObservations.isEmpty == false else {
            return nil
        }
        return retainedObservations.removeFirst()
    }

    /// Observations currently retained.
    public var observationCount: Int {
        return retainedObservations.count
    }

    /// Total observations ever stored, including evicted ones.
    public var storedObservationCount: UInt64 {
        return storedObservationTotal
    }

    /// Total observations evicted by the capacity bound.
    public var evictedObservationCount: UInt64 {
        return evictedObservationTotal
    }

    /// The token horizon this ring retains.
    public var capacity: Int {
        return retentionCapacity
    }
}
