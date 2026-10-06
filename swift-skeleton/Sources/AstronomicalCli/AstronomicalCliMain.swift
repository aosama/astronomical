import Foundation;

import AstronomicalConfig;

import AstronomicalCli;

/// The astronomical CLI (Command Line Interface) entry point.
///
/// Carried contracts from the migration skeleton:
/// - Commands emit a live progress indicator instead of silent waits.
/// - The CLI resolves default models through the shared AstronomicalConfig
///   default-model surface so CLI and daemon can never drift.
@main
final class AstronomicalCliMain {

    static func main() {
        let processArguments: Array<String> = Array(CommandLine.arguments.dropFirst());
        guard let verb: String = processArguments.first else {
            failWithNotice();
        }
        if verb == "status" {
            runStatusVerb();
        }
        failWithNotice();
    }

    private static func runStatusVerb() -> Never {
        let instancePaths: AstronomicalInstancePaths;
        do {
            instancePaths = try AstronomicalInstancePaths.forCurrentUser(
                runtimeInstance: .development);
        } catch {
            FileHandle.standardError.write(Data(
                "astronomical: could not resolve instance paths: \(error)\n".utf8));
            exit(2);
        }
        let report: String;
        do {
            report = try StatusCommand.run(instancePaths: instancePaths);
        } catch {
            FileHandle.standardError.write(Data(
                "astronomical: status failed: \(error)\n".utf8));
            exit(1);
        }
        print(report, terminator: "");
        exit(0);
    }

    private static func failWithNotice() -> Never {
        let notPortedNotice: String = "astronomical: this verb is not ported to the Swift CLI yet\n";
        FileHandle.standardError.write(Data(notPortedNotice.utf8));
        exit(1);
    }
}
