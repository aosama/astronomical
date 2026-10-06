import Foundation;

/// A minimal reference box for value-type state that must mutate behind a
/// non-mutating protocol seam; the box itself is never shared across
/// threads — ownership stays with the worker loop.
public final class MutableBox<Value> {

    public var value: Value;

    public init(_ initialValue: Value) {
        self.value = initialValue;
    }
}
