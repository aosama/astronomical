import Foundation

public enum ApplicationChannel: String, Equatable, Sendable {
  case stable
  case development
  case appStore = "app-store"

  // The App Store build is the Stable product distributed through the store,
  // so it presents the Stable product identity.
  public var displayName: String { self == .development ? "Development" : "Stable" }
  // The engine knows two runtime instances and names the one it runs in its
  // status. The store build is a way of distributing Stable, not a third
  // instance, so it starts and expects the Stable one. Comparing the engine's
  // answer with the channel name instead made a store build's agent refuse
  // its own engine and restart it every few seconds (issue #2188).
  public var runtimeInstanceName: String { self == .development ? "development" : "stable" }
  public var stateDirectoryName: String? {
    switch self {
    case .stable: ".astronomical"
    case .development: ".astronomical-dev"
    case .appStore: nil
    }
  }
  // Mirrors the daemon's master table (STABLE_LOOPBACK_PORT /
  // DEVELOPMENT_LOOPBACK_PORT in crates/config/src/astronomical_runtime_instance.rs);
  // Swift cannot import the Rust constants, so keep the values in sync by hand.
  public var defaultDaemonPort: Int { self == .development ? 6733 : 6732 }
  // macOS gives a shared scheme to whichever installed app it prefers, so a shared
  // one let Development's Observatory open Stable's chat window. The App Store
  // build is the Stable product on Stable's port, so sharing Stable's scheme can
  // never reach a different instance.
  // Mirrors `chat_url_scheme` in xtask's channel profiles and THIN_TALK_CHAT_LINKS
  // in the Observatory console; neither can import these values.
  public var chatURLScheme: String { self == .development ? "astronomical-dev" : "astronomical" }
  // Mirrors `bundle_identifier` in xtask's channel profiles. The main app and
  // its agent both derive their command names from it, so each must reach the
  // same value without reading the other's bundle.
  public var applicationBundleIdentifier: String {
    switch self {
    case .stable: "dev.astronomical.app"
    case .development: "dev.astronomical.app.development"
    case .appStore: "app.astronomical.appstore"
    }
  }
  // Mirrors `agent_bundle_identifier` in xtask's channel profiles. A helper's
  // identifier must extend its main app's for the App Store, and the main app
  // registers the agent as its login item under this exact value.
  public var agentBundleIdentifier: String { "\(applicationBundleIdentifier).agent" }
}

/// Immutable application identity injected by the app bundle and shared by every local boundary.
public struct ApplicationIdentity: Equatable, Sendable {
  public let channel: ApplicationChannel
  public let daemonPort: Int
  public let stateDirectoryName: String?
  public let version: String
  public let buildNumber: String
  public let commit: String
  public let isDirty: Bool
  /// The app group both sandboxed applications of an App Store build belong
  /// to, or nil for a build that is not sandboxed. The packager derives it
  /// from the signing team and writes it into both bundles' plists.
  public let applicationGroupIdentifier: String?

  /// Resolves an app group's container. Only the platform can: the container
  /// is named after the signing team and is created on first request.
  public typealias GroupContainerResolver = @Sendable (_ applicationGroupIdentifier: String) -> URL?

  public static let systemGroupContainerURL: GroupContainerResolver = { applicationGroupIdentifier in
    FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: applicationGroupIdentifier)
  }

  public init(
    channel: ApplicationChannel,
    daemonPort: Int,
    stateDirectoryName: String?,
    version: String,
    buildNumber: String,
    commit: String,
    isDirty: Bool,
    applicationGroupIdentifier: String? = nil
  ) {
    self.channel = channel
    self.daemonPort = daemonPort
    self.stateDirectoryName = stateDirectoryName
    self.version = version
    self.buildNumber = buildNumber
    self.commit = commit
    self.isDirty = isDirty
    self.applicationGroupIdentifier = applicationGroupIdentifier
  }

  public static func current(bundle: Bundle = .main) -> ApplicationIdentity {
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
      isDirty: isDirty,
      applicationGroupIdentifier: bundle.object(forInfoDictionaryKey: "AstronomicalApplicationGroupIdentifier")
        as? String
    )
  }

  public func endpointURL(path: String) throws -> URL {
    guard let endpointURL = URL(string: "http://127.0.0.1:\(daemonPort)\(path)") else {
      throw ApplicationIdentityError.invalidEndpoint
    }
    return endpointURL
  }

  /// Where this instance keeps its state.
  ///
  /// Direct channels use a dot-folder in the home directory. An App Store
  /// build runs as two sandboxed applications, each with a container of its
  /// own, so its state lives in their shared group container: in either
  /// application's own container the other could not reach it, and the chat
  /// app and the engine would each see a different instance. A store build
  /// without a group (one that predates the two applications) keeps the
  /// container's Application Support directory.
  public func stateDirectoryURL(
    homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser,
    groupContainerURL: GroupContainerResolver = ApplicationIdentity.systemGroupContainerURL
  ) -> URL {
    if let stateDirectoryName {
      return homeDirectoryURL.appendingPathComponent(stateDirectoryName, isDirectory: true)
    }
    if let applicationGroupContainerURL = applicationGroupContainerURL(groupContainerURL: groupContainerURL) {
      return applicationGroupContainerURL.appendingPathComponent(
        Self.containerStateDirectoryName, isDirectory: true)
    }
    return legacyContainerStateDirectoryURL(homeDirectoryURL: homeDirectoryURL)
  }

  /// The name of the state root inside a sandbox location. It matches the
  /// config crate's APPLICATION_SUPPORT_STABLE_DIRECTORY_NAME, which the
  /// engine uses beneath both Application Support and a group container.
  static let containerStateDirectoryName = "Astronomical"

  /// The group container of a store build, or nil for every other build.
  public func applicationGroupContainerURL(
    groupContainerURL: GroupContainerResolver = ApplicationIdentity.systemGroupContainerURL
  ) -> URL? {
    guard stateDirectoryName == nil, let applicationGroupIdentifier else { return nil }
    return groupContainerURL(applicationGroupIdentifier)
  }

  /// Where a store build kept its state before the engine moved to the
  /// menu-bar agent: Application Support, which the App Sandbox maps into
  /// the calling application's own container.
  public func legacyContainerStateDirectoryURL(
    homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser
  ) -> URL {
    Self.applicationSupportDirectoryURL(homeDirectoryURL: homeDirectoryURL)
      .appendingPathComponent(Self.containerStateDirectoryName, isDirectory: true)
  }

  public static func applicationSupportDirectoryURL(homeDirectoryURL: URL) -> URL {
    homeDirectoryURL.appendingPathComponent("Library/Application Support", isDirectory: true)
  }

  /// The file the menu-bar agent locks so that only one agent runs. The agent
  /// creates it, and with it the state directory, before the engine has
  /// written anything, so its presence alone does not mean state exists.
  public static let agentInstanceLockFileName = "menu-agent.lock"

  public func configFileURL(homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
    stateDirectoryURL(homeDirectoryURL: homeDirectoryURL).appendingPathComponent("config.json")
  }

  public func daemonOwnershipURL(homeDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
    stateDirectoryURL(homeDirectoryURL: homeDirectoryURL).appendingPathComponent("menu-owned-daemon.json")
  }

  public var daemonArguments: [String] { daemonArguments() }

  /// The App Store build runs the Stable runtime instance. With an app group
  /// the engine is told the group container, which only this side can
  /// resolve; without one its compiled app-store-state-root feature falls
  /// back to the container's Application Support.
  public func daemonArguments(
    groupContainerURL: GroupContainerResolver = ApplicationIdentity.systemGroupContainerURL
  ) -> [String] {
    let instanceArguments = ["--instance", channel.runtimeInstanceName]
    guard let applicationGroupContainerURL = applicationGroupContainerURL(groupContainerURL: groupContainerURL)
    else { return instanceArguments }
    return instanceArguments + ["--application-group-container", applicationGroupContainerURL.path]
  }

  /// Public status uses a privacy-safe home-relative label instead of exposing
  /// the user's absolute home directory. App Store builds live in their app
  /// group container, or beneath Application Support when they have no group.
  public var expectedServerStateDirectory: String {
    if let stateDirectoryName { return "~/\(stateDirectoryName)" }
    if let applicationGroupIdentifier {
      return "~/Library/Group Containers/\(applicationGroupIdentifier)/\(Self.containerStateDirectoryName)"
    }
    return "~/Library/Application Support/\(Self.containerStateDirectoryName)"
  }

  public var expectedConfigurationFile: String { "\(expectedServerStateDirectory)/config.json" }

  public var buildTitle: String {
    let dirtySuffix = isDirty ? "-dirty" : ""
    return "\(version) · \(channel.displayName) · \(commit)\(dirtySuffix)"
  }
  /// Identifies one build of the bundle. The main app sends it to the agent at
  /// launch, and an agent started from an older build relaunches itself, so
  /// the engine never lags the app after an update installed while the agent
  /// kept running.
  public var buildFingerprint: String {
    "\(version)|\(buildNumber)|\(commit)|\(isDirty)"
  }
}

public enum ApplicationIdentityError: LocalizedError {
  case invalidEndpoint

  public var errorDescription: String? { "The local server address is invalid" }
}

