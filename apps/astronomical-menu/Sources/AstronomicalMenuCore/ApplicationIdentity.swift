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
  var defaultDaemonPort: Int { self == .development ? 6733 : 6732 }
}

/// Immutable application identity injected by the app bundle and shared by every local boundary.
struct ApplicationIdentity: Equatable, Sendable {
  let channel: ApplicationChannel
  let daemonPort: Int
  let stateDirectoryName: String?
  let version: String
  let buildNumber: String
  let commit: String
  let isDirty: Bool

  static func current(bundle: Bundle = .main) -> ApplicationIdentity {
    // Every bundle value is optional metadata: each falls back to the
    // channel's compiled-in default when the key is missing or has the
    // wrong type.
    let rawChannel = bundle.object(forInfoDictionaryKey: "AstronomicalChannel") as? String
    let channel: ApplicationChannel
    if let rawChannel, let bundledChannel = ApplicationChannel(rawValue: rawChannel) {
      channel = bundledChannel
    } else {
      channel = .development
    }

    let configuredPort = bundle.object(forInfoDictionaryKey: "AstronomicalDaemonPort") as? Int
    let daemonPort: Int
    if let configuredPort {
      daemonPort = configuredPort
    } else {
      daemonPort = channel.defaultDaemonPort
    }

    let configuredStateDirectoryName =
      bundle.object(forInfoDictionaryKey: "AstronomicalStateDirectoryName") as? String
    let stateDirectoryName: String?
    if let configuredStateDirectoryName {
      stateDirectoryName = configuredStateDirectoryName
    } else {
      stateDirectoryName = channel.stateDirectoryName
    }

    let bundledVersion = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
    let version: String
    if let bundledVersion {
      version = bundledVersion
    } else {
      version = "unknown"
    }

    let bundledBuildNumber = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String
    let buildNumber: String
    if let bundledBuildNumber {
      buildNumber = bundledBuildNumber
    } else {
      buildNumber = "0"
    }

    let bundledCommit = bundle.object(forInfoDictionaryKey: "AstronomicalBuildCommit") as? String
    let commit: String
    if let bundledCommit {
      commit = bundledCommit
    } else {
      commit = "unknown"
    }

    let bundledDirtyFlag = bundle.object(forInfoDictionaryKey: "AstronomicalBuildDirty") as? Bool
    let isDirty: Bool
    if let bundledDirtyFlag {
      isDirty = bundledDirtyFlag
    } else {
      isDirty = false
    }

    return ApplicationIdentity(
      channel: channel,
      daemonPort: daemonPort,
      stateDirectoryName: stateDirectoryName,
      version: version,
      buildNumber: buildNumber,
      commit: commit,
      isDirty: isDirty
    )
  }

  func endpointURL(path: String) throws -> URL {
    guard let endpointURL = URL(string: "http://127.0.0.1:\(daemonPort)\(path)") else {
      throw DaemonClientError.invalidEndpoint
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
