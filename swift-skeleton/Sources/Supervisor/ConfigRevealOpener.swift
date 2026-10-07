import Foundation;

import AstronomicalConfig;

/// Reveals the active configuration file in Finder, migrating the
/// `/usr/bin/open -R` invocation from
/// apps/supervisor/src/config_reveal_endpoint.rs: detached standard streams
/// and a bounded wait, because a wedged Finder must never wedge the endpoint.
public enum ConfigRevealOpener {

    public static let revealTimeoutSeconds: Double = 5;

    public static func revealInFinder(configFilePath: FilePath) -> Bool {
        let openProcess: Process = Process();
        openProcess.executableURL = URL(fileURLWithPath: "/usr/bin/open");
        openProcess.arguments = ["-R", configFilePath.string];
        openProcess.standardInput = FileHandle.nullDevice;
        openProcess.standardOutput = FileHandle.nullDevice;
        openProcess.standardError = FileHandle.nullDevice;
        do {
            try openProcess.run();
        } catch {
            return false;
        }
        let waitDeadline: Date = Date().addingTimeInterval(ConfigRevealOpener.revealTimeoutSeconds);
        while openProcess.isRunning && Date() < waitDeadline {
            Thread.sleep(forTimeInterval: 0.02);
        }
        if openProcess.isRunning {
            openProcess.terminate();
            return false;
        }
        return openProcess.terminationStatus == 0;
    }
}
