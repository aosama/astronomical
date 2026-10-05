// AstronomicalCliMain.swift — AstronomicalCli
//
// MIGRATION MARKER — placeholder entry point; the command registry is
// not ported yet (wave 2), so main only reports that and exits.
//
// Migrates from (wave 2): apps/astronomical CLI (Command Line Interface)
// entry point and command registry.
//
// Carried contracts:
// - Commands emit a live progress indicator instead of silent waits.
// - The CLI resolves default models through the shared AstronomicalConfig
//   default-model surface so CLI and daemon can never drift.

import Foundation;

@main
final class AstronomicalCliMain {
    static func main() {
        let notPortedNotice: String = "astronomical: the Swift CLI (Command Line Interface) is not ported yet\n";
        guard let noticeBytes: Data = notPortedNotice.data(using: .utf8) else {
            exit(1);
        }
        FileHandle.standardError.write(noticeBytes);
        exit(1);
    }
}
