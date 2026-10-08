import Foundation

/**
 * Supervisor-owned operation names recorded by the common attribution log,
 * mirroring the Rust enum from
 * apps/supervisor/src/supervisor_performance_attribution.rs.
 */
public enum SupervisorPerformanceOperation: Equatable, Sendable {

    case libraryCatalogLoad
    case diskPreflight
    case manifestFetch
    case executablePreflight
    case fileTransfer
    case verification
    case publication
    case discoveryRefresh
    case daemonIpcHandshake
    case daemonIpcStatus
    case daemonIpcChatGenerate
    case daemonIpcEmbedGenerate
    case daemonIpcModelsList
    case daemonIpcCatalog
    case daemonIpcDownloadStart
    case daemonIpcDownloadStatus
    case daemonIpcDefaultModelSet

    public var wireName: String {
        switch (self) {
        case .libraryCatalogLoad: return "library_catalog_load"
        case .diskPreflight: return "disk_preflight"
        case .manifestFetch: return "manifest_fetch"
        case .executablePreflight: return "executable_preflight"
        case .fileTransfer: return "file_transfer"
        case .verification: return "verification"
        case .publication: return "publication"
        case .discoveryRefresh: return "discovery_refresh"
        case .daemonIpcHandshake: return "daemon_ipc_handshake"
        case .daemonIpcStatus: return "daemon_ipc_status"
        case .daemonIpcChatGenerate: return "daemon_ipc_chat_generate"
        case .daemonIpcEmbedGenerate: return "daemon_ipc_embed_generate"
        case .daemonIpcModelsList: return "daemon_ipc_models_list"
        case .daemonIpcCatalog: return "daemon_ipc_catalog"
        case .daemonIpcDownloadStart: return "daemon_ipc_download_start"
        case .daemonIpcDownloadStatus: return "daemon_ipc_download_status"
        case .daemonIpcDefaultModelSet: return "daemon_ipc_default_model_set"
        }
    }
}
