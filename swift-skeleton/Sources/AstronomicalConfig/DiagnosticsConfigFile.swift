import Foundation;

/**
 * `diagnostics` section of the v1 user configuration document.
 *
 * MIGRATION MARKER — deferred from this slice: `log_level` is a Rust enum
 * accepting error/warn/info/debug/trace. This port keeps it a pass-through
 * string; the logging-config slice adds the enum and its validation.
 */
internal struct DiagnosticsConfigFile: Equatable {
    internal let performanceAttributionEnabled: Bool?;
    internal let completionAttributionEnabled: Bool?;
    internal let logLevel: String?;
    internal let retainedLogFiles: Int32?;

    internal init(
        performanceAttributionEnabled: Bool?,
        completionAttributionEnabled: Bool?,
        logLevel: String?,
        retainedLogFiles: Int32?
    ) {
        self.performanceAttributionEnabled = performanceAttributionEnabled;
        self.completionAttributionEnabled = completionAttributionEnabled;
        self.logLevel = logLevel;
        self.retainedLogFiles = retainedLogFiles;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> DiagnosticsConfigFile {
        try StrictJson.requireKnownKeys(
            object: jsonObject,
            knownKeys: [
                "performance_attribution_enabled", "completion_attribution_enabled",
                "log_level", "retained_log_files"
            ],
            fieldName: "diagnostics"
        );
        return DiagnosticsConfigFile(
            performanceAttributionEnabled: try StrictJson.optionalBoolean(
                object: jsonObject,
                fieldName: "performance_attribution_enabled"
            ),
            completionAttributionEnabled: try StrictJson.optionalBoolean(
                object: jsonObject,
                fieldName: "completion_attribution_enabled"
            ),
            logLevel: try StrictJson.optionalString(object: jsonObject, fieldName: "log_level"),
            retainedLogFiles: try StrictJson.optionalSignedInteger(
                object: jsonObject,
                fieldName: "retained_log_files"
            )
        );
    }

    internal func toJsonObject() -> Any {
        var jsonObject: Dictionary<String, Any> = Dictionary<String, Any>();
        if let performanceAttributionEnabled: Bool = self.performanceAttributionEnabled {
            jsonObject["performance_attribution_enabled"] = performanceAttributionEnabled;
        }
        if let completionAttributionEnabled: Bool = self.completionAttributionEnabled {
            jsonObject["completion_attribution_enabled"] = completionAttributionEnabled;
        }
        if let logLevel: String = self.logLevel {
            jsonObject["log_level"] = logLevel;
        }
        if let retainedLogFiles: Int32 = self.retainedLogFiles {
            jsonObject["retained_log_files"] = retainedLogFiles;
        }
        return jsonObject;
    }
}
