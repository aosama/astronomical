import Foundation

import Testing

import AstronomicalConfig

@testable import Supervisor

/**
 * Unit journeys for the supervisor operator log: diagnostics persist into
 * the instance logs directory under the supervisor file family, retention
 * prunes the oldest rotated files, and the disabled log records nothing.
 */
@Suite(.tags(.hermeticJourney))
final class SupervisorOperatorLogTests {

    @Test
    func should_persist_diagnostic_lines_into_the_supervisor_log_family() throws {
        let logDirectoryPath: String = SupervisorOperatorLogTests.makeLogDirectory();
        let operatorLog: SupervisorOperatorLog = SupervisorOperatorLog.open(
            logDirectory: FilePath(string: logDirectoryPath));

        operatorLog.recordDiagnosticLine("worker containment: first diagnostic");
        operatorLog.recordDiagnosticLine("worker containment: second diagnostic");

        let persistedText: String = SupervisorOperatorLogTests.readSupervisorLogs(
            logDirectoryPath: logDirectoryPath);
        #expect(persistedText.contains("first diagnostic"));
        #expect(persistedText.contains("second diagnostic"));
        #expect(SupervisorOperatorLogTests.supervisorLogFileNames(
            logDirectoryPath: logDirectoryPath).count == 1);
    }

    @Test
    func should_prune_rotated_files_beyond_the_retention_bound() throws {
        let logDirectoryPath: String = SupervisorOperatorLogTests.makeLogDirectory();
        for staleHourOffset: Int in [0, 1, 2] {
            let staleStamp: String = SupervisorOperatorLogTests.hourStamp(
                hoursBefore: staleHourOffset + 1);
            let staleLogText: String = "stale supervisor." + staleStamp + ".log\n";
            try staleLogText.write(
                toFile: logDirectoryPath + "/supervisor." + staleStamp + ".log",
                atomically: true,
                encoding: String.Encoding.utf8);
        }
        let operatorLog: SupervisorOperatorLog = SupervisorOperatorLog.open(
            logDirectory: FilePath(string: logDirectoryPath),
            retainedLogFileCount: 2);

        operatorLog.recordDiagnosticLine("fresh diagnostic after rotation");

        let survivingLogFileNames: Array<String> = SupervisorOperatorLogTests.supervisorLogFileNames(
            logDirectoryPath: logDirectoryPath);
        #expect(survivingLogFileNames.count == 2, "retention must keep exactly two files, got \(survivingLogFileNames)");
        #expect(!survivingLogFileNames.contains("supervisor."
            + SupervisorOperatorLogTests.hourStamp(hoursBefore: 3) + ".log"),
            "the oldest rotated file must be pruned");
        #expect(SupervisorOperatorLogTests.readSupervisorLogs(
            logDirectoryPath: logDirectoryPath).contains("fresh diagnostic after rotation"));
    }

    @Test
    func should_record_nothing_when_disabled() throws {
        let logDirectoryPath: String = SupervisorOperatorLogTests.makeLogDirectory();
        let disabledLog: SupervisorOperatorLog = SupervisorOperatorLog.disabled();

        disabledLog.recordDiagnosticLine("must never be recorded");

        #expect(SupervisorOperatorLogTests.supervisorLogFileNames(
            logDirectoryPath: logDirectoryPath).isEmpty);
    }

    private static func makeLogDirectory() -> String {
        let logDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("aa-operator-log-\(UUID().uuidString)", isDirectory: true);
        try? FileManager.default.createDirectory(at: logDirectoryUrl, withIntermediateDirectories: true);
        return logDirectoryUrl.path;
    }

    private static func supervisorLogFileNames(logDirectoryPath: String) -> Array<String> {
        let directoryEntryNames: Array<String> =
            (try? FileManager.default.contentsOfDirectory(atPath: logDirectoryPath)) ?? [];
        return directoryEntryNames
            .filter({ (entryName: String) -> Bool in
                return entryName.hasPrefix("supervisor.") && entryName.hasSuffix(".log");
            })
            .sorted();
    }

    private static func readSupervisorLogs(logDirectoryPath: String) -> String {
        var combinedLogText: String = "";
        for logFileName: String in SupervisorOperatorLogTests.supervisorLogFileNames(
            logDirectoryPath: logDirectoryPath) {
            if let logText: String = try? String(
                contentsOfFile: logDirectoryPath + "/" + logFileName,
                encoding: String.Encoding.utf8) {
                combinedLogText += logText;
            }
        }
        return combinedLogText;
    }

    private static func hourStamp(hoursBefore: Int) -> String {
        let hourStampFormatter: DateFormatter = DateFormatter();
        hourStampFormatter.dateFormat = "yyyyMMdd-HH";
        hourStampFormatter.locale = Locale(identifier: "en_US_POSIX");
        hourStampFormatter.timeZone = TimeZone(identifier: "UTC");
        return hourStampFormatter.string(from: Date().addingTimeInterval(-Double(hoursBefore) * 3_600));
    }
}
