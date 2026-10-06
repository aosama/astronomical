import Foundation;

/** Machine-readable diagnostic vocabulary of a discovery report. */
public enum DiscoveryModelDiscoveryDiagnosticCode: String, Equatable, Sendable {
    case ambiguousModelIdentity = "ambiguous_model_identity";
    case unavailableModelDirectory = "unavailable_model_directory";
}
