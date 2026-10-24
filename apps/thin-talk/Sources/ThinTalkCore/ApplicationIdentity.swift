import Foundation

/// The runtime channel a thin chat app is built for. Each channel talks to an
/// isolated daemon state directory and a distinct local port.
public enum ThinTalkChannel: String, Equatable, Sendable {
  case stable
  case development

  /// The Stable build connects to the production channel; the Development build
  /// to the isolated test channel.
  public var displayName: String { self == .development ? "Development" : "Stable" }

  /// The state directory this channel owns under the home directory.
  public var stateDirectoryName: String { self == .development ? ".astronomical-dev" : ".astronomical" }

  /// The default daemon port for this channel. Mirrors the daemon's master
  /// table (STABLE_LOOPBACK_PORT / DEVELOPMENT_LOOPBACK_PORT in
  /// crates/config/src/astronomical_runtime_instance.rs); Swift cannot import
  /// the Rust constants, so keep the values in sync by hand.
  public var defaultDaemonPort: Int { self == .development ? 6733 : 6732 }
}

/// Immutable application identity injected by the app bundle and shared by every
/// local boundary. It is built from the Info.plist, never from the host machine,
/// so channel, daemon port, and state directory share one bundled identity.
public struct ThinTalkApplicationIdentity: Equatable, Sendable {
  public let channel: ThinTalkChannel
  public let daemonPort: Int
  public let stateDirectoryName: String
  public let version: String
  public let buildNumber: String
  public let commit: String
  public let isDirty: Bool

  public static func current(bundle: Bundle = .main) -> ThinTalkApplicationIdentity {
    let rawChannel = bundle.object(forInfoDictionaryKey: "AstronomicalChannel") as? String
    let channel: ThinTalkChannel
    if let rawChannel, let bundledChannel = ThinTalkChannel(rawValue: rawChannel) {
      channel = bundledChannel
    } else {
      channel = .development
    }

    let daemonPort: Int
    if let bundledPort = bundle.object(forInfoDictionaryKey: "AstronomicalDaemonPort") as? Int {
      daemonPort = bundledPort
    } else {
      daemonPort = channel.defaultDaemonPort
    }

    let configuredStateDirectoryName =
      bundle.object(forInfoDictionaryKey: "AstronomicalStateDirectoryName") as? String
    let stateDirectoryName: String
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

    return ThinTalkApplicationIdentity(
      channel: channel,
      daemonPort: daemonPort,
      stateDirectoryName: stateDirectoryName,
      version: version,
      buildNumber: buildNumber,
      commit: commit,
      isDirty: isDirty
    )
  }

  /// Designated initializer covering every identity field.
  public init(
    channel: ThinTalkChannel,
    daemonPort: Int,
    stateDirectoryName: String,
    version: String,
    buildNumber: String,
    commit: String,
    isDirty: Bool
  ) {
    self.channel = channel
    self.daemonPort = daemonPort
    self.stateDirectoryName = stateDirectoryName
    self.version = version
    self.buildNumber = buildNumber
    self.commit = commit
    self.isDirty = isDirty
  }

  /// Builds an identity from a bare port, primarily for tests.
  public init(channel: ThinTalkChannel, daemonPort: Int) {
    self.init(
      channel: channel,
      daemonPort: daemonPort,
      stateDirectoryName: channel.stateDirectoryName,
      version: "0.1.0",
      buildNumber: "1",
      commit: "test",
      isDirty: false
    )
  }

  /// Resolves the public endpoint URL for one REST path on this channel's
  /// daemon. Public endpoints need no authentication.
  public func endpointURL(path: String) throws -> URL {
    guard path.hasPrefix("/") else {
      throw ThinTalkEndpointError.invalidEndpoint
    }
    guard let endpointURL = URL(string: "http://127.0.0.1:\(daemonPort)\(path)") else {
      throw ThinTalkEndpointError.invalidEndpoint
    }
    return endpointURL
  }

  /// A privacy-safe, home-relative label for the connected channel's state
  /// directory, never the caller's absolute home directory.
  public var expectedServerStateDirectory: String { "~/\(stateDirectoryName)" }
}

/// An endpoint path the identity cannot turn into a daemon URL.
public enum ThinTalkEndpointError: Error, Equatable, Sendable {
  case invalidEndpoint
}
