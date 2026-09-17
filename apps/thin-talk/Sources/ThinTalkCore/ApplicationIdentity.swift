import Foundation

/// The runtime channel a thin chat app is built for. Each channel talks to an
/// isolated supervisor state directory and a distinct local port.
public enum ThinTalkChannel: String, Equatable, Sendable {
  case stable
  case development

  /// The Stable build connects to the production channel; the Development build
  /// to the isolated test channel.
  public var displayName: String { self == .development ? "Development" : "Stable" }

  /// The state directory this channel owns under the home directory.
  public var stateDirectoryName: String { self == .development ? ".astronomical-dev" : ".astronomical" }

  /// The default supervisor port for this channel.
  public var defaultSupervisorPort: Int { self == .development ? 6733 : 6732 }
}

/// Immutable application identity injected by the app bundle and shared by every
/// local boundary. It is built from the Info.plist, never from the host machine,
/// so the same binary can drive either isolated supervisor instance.
public struct ThinTalkApplicationIdentity: Equatable, Sendable {
  public let channel: ThinTalkChannel
  public let supervisorPort: Int
  public let stateDirectoryName: String
  public let version: String
  public let buildNumber: String
  public let commit: String
  public let isDirty: Bool

  public static func current(bundle: Bundle = .main) -> ThinTalkApplicationIdentity {
    let rawChannel = bundle.object(forInfoDictionaryKey: "AstronomicalChannel") as? String
    let channel = ThinTalkChannel(rawValue: rawChannel ?? "") ?? .development
    let configuredPort = bundle.object(forInfoDictionaryKey: "AstronomicalSupervisorPort") as? Int
    return ThinTalkApplicationIdentity(
      channel: channel,
      supervisorPort: configuredPort ?? channel.defaultSupervisorPort,
      stateDirectoryName:
        (bundle.object(forInfoDictionaryKey: "AstronomicalStateDirectoryName") as? String)
          ?? channel.stateDirectoryName,
      version: (bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String)
        ?? "unknown",
      buildNumber: (bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String) ?? "0",
      commit: (bundle.object(forInfoDictionaryKey: "AstronomicalBuildCommit") as? String)
        ?? "unknown",
      isDirty: (bundle.object(forInfoDictionaryKey: "AstronomicalBuildDirty") as? Bool) ?? false
    )
  }

  /// Designated initializer covering every identity field.
  public init(
    channel: ThinTalkChannel,
    supervisorPort: Int,
    stateDirectoryName: String,
    version: String,
    buildNumber: String,
    commit: String,
    isDirty: Bool
  ) {
    self.channel = channel
    self.supervisorPort = supervisorPort
    self.stateDirectoryName = stateDirectoryName
    self.version = version
    self.buildNumber = buildNumber
    self.commit = commit
    self.isDirty = isDirty
  }

  /// Builds an identity from a bare port, primarily for tests.
  public init(channel: ThinTalkChannel, supervisorPort: Int) {
    self.init(
      channel: channel,
      supervisorPort: supervisorPort,
      stateDirectoryName: channel.stateDirectoryName,
      version: "0.1.0",
      buildNumber: "1",
      commit: "test",
      isDirty: false
    )
  }

  /// Resolves the public endpoint URL for one REST path on this channel's
  /// supervisor. Public endpoints need no authentication.
  public func endpointURL(path: String) throws -> URL {
    guard path.hasPrefix("/") else {
      throw ThinTalkClientError.invalidEndpoint
    }
    guard let endpointURL = URL(string: "http://127.0.0.1:\(supervisorPort)\(path)") else {
      throw ThinTalkClientError.invalidEndpoint
    }
    return endpointURL
  }

  /// A privacy-safe, home-relative label for the connected channel's state
  /// directory, never the caller's absolute home directory.
  public var expectedServerStateDirectory: String { "~/\(stateDirectoryName)" }
}
