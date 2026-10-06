import Foundation;

import IpcProtocol;

/// Renders one value into the status document's JSON wire form.
public protocol ConfigurationStatusWireEncodable {

    func configurationStatusWireValue() -> JsonWireValue;
}

extension Bool: ConfigurationStatusWireEncodable {

    public func configurationStatusWireValue() -> JsonWireValue {
        return .boolean(self);
    }
}

extension UInt32: ConfigurationStatusWireEncodable {

    public func configurationStatusWireValue() -> JsonWireValue {
        return .unsignedInteger(UInt64(self));
    }
}

extension UInt64: ConfigurationStatusWireEncodable {

    public func configurationStatusWireValue() -> JsonWireValue {
        return .unsignedInteger(self);
    }
}

extension Double: ConfigurationStatusWireEncodable {

    public func configurationStatusWireValue() -> JsonWireValue {
        return .double(self);
    }
}

/// The configured/default/effective triple of one status quantity, porting
/// ConfigurationValue from apps/supervisor/src/configuration_status.rs.
public struct ConfigurationValue<T> {

    /// The operator-authored value, absent when the operator left it unset.
    public let configured: T?;
    /// The repository default, absent where the contract defines none.
    public let defaultValue: T?;
    /// The value the live instance currently enforces.
    public let effective: T?;

    public init(configured: T?, defaultValue: T?, effective: T?) {
        self.configured = configured;
        self.defaultValue = defaultValue;
        self.effective = effective;
    }

    public func wireValue() -> JsonWireValue where T: ConfigurationStatusWireEncodable {
        var wireObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "configured", value: ConfigurationValue.optionalWireValue(self.configured));
        wireObject.appendEntry(key: "default", value: ConfigurationValue.optionalWireValue(self.defaultValue));
        wireObject.appendEntry(key: "effective", value: ConfigurationValue.optionalWireValue(self.effective));
        return .object(wireObject);
    }

    private static func optionalWireValue(_ optionalValue: T?) -> JsonWireValue where T: ConfigurationStatusWireEncodable {
        guard let presentValue: T = optionalValue else {
            return .null;
        }
        return presentValue.configurationStatusWireValue();
    }
}

/// The configured/default/effective triple of a quantity that is legitimately
/// undefined (never configured and carrying no default), adding the explicit
/// is_configured discriminator, porting NullableConfigurationValue.
public struct NullableConfigurationValue<T> {

    /// Whether the operator authored this quantity at all.
    public let isConfigured: Bool;
    public let configured: T?;
    public let defaultValue: T?;
    public let effective: T?;

    public init(isConfigured: Bool, configured: T?, defaultValue: T?, effective: T?) {
        self.isConfigured = isConfigured;
        self.configured = configured;
        self.defaultValue = defaultValue;
        self.effective = effective;
    }

    public func wireValue() -> JsonWireValue where T: ConfigurationStatusWireEncodable {
        var wireObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "is_configured", value: .boolean(self.isConfigured));
        wireObject.appendEntry(key: "configured", value: NullableConfigurationValue.optionalWireValue(self.configured));
        wireObject.appendEntry(key: "default", value: NullableConfigurationValue.optionalWireValue(self.defaultValue));
        wireObject.appendEntry(key: "effective", value: NullableConfigurationValue.optionalWireValue(self.effective));
        return .object(wireObject);
    }

    private static func optionalWireValue(_ optionalValue: T?) -> JsonWireValue where T: ConfigurationStatusWireEncodable {
        guard let presentValue: T = optionalValue else {
            return .null;
        }
        return presentValue.configurationStatusWireValue();
    }
}
