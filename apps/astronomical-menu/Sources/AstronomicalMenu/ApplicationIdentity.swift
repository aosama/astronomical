import Foundation

enum ApplicationChannel: String, Equatable, Sendable {
  case stable
  case development

  var displayName: String { self == .stable ? "Stable" : "Development" }
  var stateDirectoryName: String { self == .stable ? ".astronomical" : ".astronomical-dev" }
  var defaultSupervisorPort: Int { self == .stable ? 6732 : 6733 }
}

/// Immutable application identity injected by the app bundle and shared by every local boundary.
struct ApplicationIdentity: Equatable, Sendable {
  let channel: ApplicationChannel
  let supervisorPort: Int
  let stateDirectoryName: String
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
    homeDirectoryURL.appendingPathComponent(stateDirectoryName, isDirectory: true)
  }

  func configFileURL(homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
    stateDirectoryURL(homeDirectoryURL: homeDirectoryURL).appendingPathComponent("config.json")
  }

  func daemonOwnershipURL(homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
    stateDirectoryURL(homeDirectoryURL: homeDirectoryURL).appendingPathComponent("menu-owned-daemon.json")
  }

  var daemonArguments: [String] { ["--instance", channel.rawValue] }

  /// Public status uses a privacy-safe home-relative label instead of exposing
  /// the user's absolute home directory.
  var expectedServerStateDirectory: String { "~/\(stateDirectoryName)" }

  var expectedConfigurationFile: String { "\(expectedServerStateDirectory)/config.json" }

  var buildTitle: String {
    let dirtySuffix = isDirty ? "-dirty" : ""
    return "\(version) · \(channel.displayName) · \(commit)\(dirtySuffix)"
  }
}
