// Where things live inside the application bundle, as both processes see it.
//
//   Astronomical.app/Contents/MacOS/                          main app, engine, worker, terminal command
//   Astronomical.app/Contents/Library/LoginItems/<Agent>.app  the menu-bar agent
//
// The agent sits in LoginItems because that is the one place macOS lets the
// main app register a helper to start at login (SMAppService.loginItem).

import Foundation

public enum ApplicationBundleLayout {
  private static let loginItemsPathComponents = ["Contents", "Library", "LoginItems"]

  /// The file this process runs from, or nil when macOS cannot say.
  ///
  /// The first launch argument is not that file: launchd starts a login item
  /// with its service label there (`dev.astronomical.app.agent`), which read
  /// as a path names a file in the working directory `/`. The agent then
  /// looked for the engine beside that file and never started it after a
  /// login. The bundle reports the path the kernel loaded, whichever way the
  /// process was started, so it is asked first; the launch argument is used
  /// only when it is an absolute path, which a label never is.
  public static func runningExecutableURL(
    bundleExecutableURL: URL? = Bundle.main.executableURL,
    launchArguments: [String] = CommandLine.arguments
  ) -> URL? {
    if let bundleExecutableURL {
      return bundleExecutableURL.standardizedFileURL.resolvingSymlinksInPath()
    }
    guard let launchedPath = launchArguments.first, launchedPath.hasPrefix("/") else { return nil }
    return URL(fileURLWithPath: launchedPath).standardizedFileURL.resolvingSymlinksInPath()
  }

  /// The directory holding the engine and the terminal command, for an
  /// executable at `executableURL`. The agent is nested four levels below the
  /// main bundle's Contents, so its own directory is not that directory. An
  /// executable outside the nested layout (a bare `swift run`, a test) finds
  /// its neighbours beside itself.
  public static func bundledExecutablesDirectory(forExecutableAt executableURL: URL) -> URL {
    let executablePathComponents = executableURL.standardizedFileURL.pathComponents
    let loginItemsLength = loginItemsPathComponents.count
    var contentsIndex: Int?
    for componentIndex in executablePathComponents.indices
    where componentIndex + loginItemsLength <= executablePathComponents.count {
      let candidateComponents = executablePathComponents[componentIndex..<(componentIndex + loginItemsLength)]
      if Array(candidateComponents) == loginItemsPathComponents {
        contentsIndex = componentIndex
      }
    }
    guard let contentsIndex else { return executableURL.deletingLastPathComponent() }
    let mainContentsPath = NSString.path(withComponents: Array(executablePathComponents[...contentsIndex]))
    return URL(fileURLWithPath: mainContentsPath, isDirectory: true)
      .appendingPathComponent("MacOS", isDirectory: true)
  }

  /// The main application bundle that contains the agent bundle at
  /// `agentBundleURL`, or nil when the agent is not nested in one.
  public static func mainApplicationBundleURL(forAgentBundleAt agentBundleURL: URL) -> URL? {
    let agentPathComponents = agentBundleURL.standardizedFileURL.pathComponents
    let enclosingLength = loginItemsPathComponents.count + 1
    guard agentPathComponents.count > enclosingLength,
      Array(agentPathComponents.dropLast().suffix(loginItemsPathComponents.count)) == loginItemsPathComponents
    else { return nil }
    let mainBundlePath = NSString.path(withComponents: Array(agentPathComponents.dropLast(enclosingLength)))
    return URL(fileURLWithPath: mainBundlePath, isDirectory: true)
  }

  /// The agent's bundle inside the main application bundle, or nil when the
  /// bundle carries none (a bare executable run has no bundle at all).
  public static func agentBundleURL(
    inApplicationBundleAt applicationBundleURL: URL, fileManager: FileManager = .default
  ) -> URL? {
    var loginItemsDirectoryURL = applicationBundleURL
    for pathComponent in loginItemsPathComponents {
      loginItemsDirectoryURL.appendPathComponent(pathComponent, isDirectory: true)
    }
    let loginItemURLs =
      (try? fileManager.contentsOfDirectory(at: loginItemsDirectoryURL, includingPropertiesForKeys: nil)) ?? []
    return loginItemURLs.first { $0.pathExtension == "app" }
  }
}
