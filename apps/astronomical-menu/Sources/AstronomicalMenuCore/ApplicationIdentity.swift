import Foundation

public enum ApplicationChannel: String, Equatable, Sendable {
  case stable
  case development
  case appStore = "app-store"

  // The App Store build is the Stable product distributed through the store,
  // so it presents the Stable product identity.
  var displayName: String { self == .development ? "Development" : "Stable" }
  var stateDirectoryName: String? {
    switch self {
    case .stable: ".astronomical"
    case .development: ".astronomical-dev"
    case .appStore: nil
    }
  }
  var defaultSupervisorPort: Int { self == .development ? 6733 : 6732 }
}

/// Immutable application identity injected by the app bundle and shared by every local boundary.
struct ApplicationIdentity: Equatable, Sendable {
  let channel: ApplicationChannel
  let supervisorPort: Int
  let stateDirectoryName: String?
  let version: String
  let buildNumber: String
  let commit: String
  let isDirty: Bool

  static func current(bundle: Bundle = .main) -> ApplicationIdentity {
    let rawChannel = bundle.object(forInfoDictionaryKey: "AstronomicalChannel") as? String
    let channel = ApplicationChannel(rawValue: rawChannel ?? "") ?? .development
    let configuredPort = bundle.object(forInfoDictionaryKey: "AstronomicalSupervisorPort") as? Int
    return ApplicationIdentity(
      channel: channel,
      supervisorPort: configuredPort ?? channel.defaultSupervisorPort,
      stateDirectoryName: (bundle.object(forInfoDictionaryKey: "AstronomicalStateDirectoryName") as? String)
        ?? channel.stateDirectoryName,
      version: (bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "unknown",
      buildNumber: (bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String) ?? "0",
      commit: (bundle.object(forInfoDictionaryKey: "AstronomicalBuildCommit") as? String) ?? "unknown",
      isDirty: (bundle.object(forInfoDictionaryKey: "AstronomicalBuildDirty") as? Bool) ?? false
    )
  }

  func endpointURL(path: String) throws -> URL {
    guard let endpointURL = URL(string: "http://127.0.0.1:\(supervisorPort)\(path)") else {
      throw SupervisorClientError.invalidEndpoint
    }
    return endpointURL
  }

  func stateDirectoryURL(homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
    guard let stateDirectoryName else {
      // App Store builds carry no state-directory name: their state root is the
      // platform-standard Application Support directory, which the App Sandbox
      // maps into the app container. The name matches the config crate's
      // APPLICATION_SUPPORT_STABLE_DIRECTORY_NAME so both sides agree.
      return Self.applicationSupportDirectoryURL(homeDirectoryURL: homeDirectoryURL)
        .appendingPathComponent("Astronomical", isDirectory: true)
    }
    return homeDirectoryURL.appendingPathComponent(stateDirectoryName, isDirectory: true)
  }

  static func applicationSupportDirectoryURL(homeDirectoryURL: URL) -> URL {
    homeDirectoryURL.appendingPathComponent("Library/Application Support", isDirectory: true)
  }

  func configFileURL(homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
    stateDirectoryURL(homeDirectoryURL: homeDirectoryURL).appendingPathComponent("config.json")
  }

  func daemonOwnershipURL(homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
    stateDirectoryURL(homeDirectoryURL: homeDirectoryURL).appendingPathComponent("menu-owned-daemon.json")
  }

  var daemonArguments: [String] {
    // The App Store build runs the Stable runtime instance; the compiled
    // app-store-state-root feature redirects its default state location.
    ["--instance", channel == .development ? "development" : "stable"]
  }

  /// Public status uses a privacy-safe home-relative label instead of exposing
  /// the user's absolute home directory. App Store builds live beneath the
  /// platform-standard Application Support directory.
  var expectedServerStateDirectory: String {
    guard let stateDirectoryName else {
      return "~/Library/Application Support/Astronomical"
    }
    return "~/\(stateDirectoryName)"
  }

  var expectedConfigurationFile: String { "\(expectedServerStateDirectory)/config.json" }

  var buildTitle: String {
    let dirtySuffix = isDirty ? "-dirty" : ""
    return "\(version) · \(channel.displayName) · \(commit)\(dirtySuffix)"
  }
}
