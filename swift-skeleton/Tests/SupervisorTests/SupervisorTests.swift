import XCTest;

@testable import Supervisor;
@testable import AstronomicalConfig;

/// Hermetic coverage for the daemon argument parser and the single-instance
/// lock. Every test uses temporary directories or fictional placeholder paths;
/// the Development instance is the only instance these tests may name.
final class SupervisorTests: XCTestCase {

    func testEmptyArgumentsDefaultToTheDevelopmentInstance() throws {
        let command: DaemonCommand = try DaemonArguments.parse(processArguments: ["astronomicald"]);
        guard case let .run(daemonArguments) = command else {
            return XCTFail("empty arguments should produce a run command");
        }
        XCTAssertEqual(daemonArguments.runtimeInstance, AstronomicalRuntimeInstance.development);
        XCTAssertNil(daemonArguments.stateDirectoryOverride);
    }

    func testExplicitInstanceArgumentIsAcceptedOnce() throws {
        let command: DaemonCommand = try DaemonArguments.parse(processArguments: [
            "astronomicald", "--instance", "development",
        ]);
        guard case let .run(daemonArguments) = command else {
            return XCTFail("instance arguments should produce a run command");
        }
        XCTAssertEqual(daemonArguments.runtimeInstance, AstronomicalRuntimeInstance.development);
    }

    func testUnknownInstanceValueIsRejected() {
        XCTAssertThrowsError(try DaemonArguments.parse(processArguments: [
            "astronomicald", "--instance", "canary",
        ])) { (thrownError: any Error) in
            XCTAssertEqual(
                thrownError as? DaemonArgumentError,
                DaemonArgumentError.invalidInstance(rawValue: "canary"));
        }
    }

    func testRepeatedInstanceArgumentIsRejected() {
        XCTAssertThrowsError(try DaemonArguments.parse(processArguments: [
            "astronomicald", "--instance", "development", "--instance", "stable",
        ])) { (thrownError: any Error) in
            XCTAssertEqual(
                thrownError as? DaemonArgumentError,
                DaemonArgumentError.repeatedArgument(argumentName: "--instance"));
        }
    }

    func testMissingInstanceValueIsRejected() {
        XCTAssertThrowsError(try DaemonArguments.parse(processArguments: [
            "astronomicald", "--instance",
        ])) { (thrownError: any Error) in
            XCTAssertEqual(
                thrownError as? DaemonArgumentError,
                DaemonArgumentError.missingValue(argumentName: "--instance"));
        }
    }

    func testRelativeStateDirectoryIsRejected() {
        XCTAssertThrowsError(try DaemonArguments.parse(processArguments: [
            "astronomicald", "--state-directory", "relative/state",
        ])) { (thrownError: any Error) in
            XCTAssertEqual(
                thrownError as? DaemonArgumentError,
                DaemonArgumentError.invalidStateDirectory(path: "relative/state"));
        }
    }

    func testRootStateDirectoryIsRejected() {
        XCTAssertThrowsError(try DaemonArguments.parse(processArguments: [
            "astronomicald", "--state-directory", "/",
        ])) { (thrownError: any Error) in
            XCTAssertEqual(
                thrownError as? DaemonArgumentError,
                DaemonArgumentError.invalidStateDirectory(path: "/"));
        }
    }

    func testUnknownArgumentIsRejected() {
        XCTAssertThrowsError(try DaemonArguments.parse(processArguments: [
            "astronomicald", "--daemonize",
        ])) { (thrownError: any Error) in
            XCTAssertEqual(
                thrownError as? DaemonArgumentError,
                DaemonArgumentError.unknownArgument(argument: "--daemonize"));
        }
    }

    func testHelpAndVersionShortCircuitBeforeValidation() throws {
        XCTAssertEqual(try DaemonArguments.parse(processArguments: ["astronomicald", "--help"]), DaemonCommand.help);
        XCTAssertEqual(try DaemonArguments.parse(processArguments: ["astronomicald", "-h"]), DaemonCommand.help);
        XCTAssertEqual(try DaemonArguments.parse(processArguments: ["astronomicald", "--version"]), DaemonCommand.version);
        XCTAssertTrue(DaemonArguments.helpText().contains("--instance"));
    }

    func testStateDirectoryOverrideResolvesThroughTheOverride() throws {
        let overrideDirectory: String = "/astronomical-test/fictional-state-root";
        let command: DaemonCommand = try DaemonArguments.parse(processArguments: [
            "astronomicald", "--state-directory", overrideDirectory,
        ]);
        guard case let .run(daemonArguments) = command else {
            return XCTFail("state-directory arguments should produce a run command");
        }
        let instancePaths: AstronomicalInstancePaths = try daemonArguments.resolveInstancePaths();
        XCTAssertTrue(instancePaths.stateDirectory.string.hasPrefix(overrideDirectory));
    }

    func testInstanceLockAcquiresInsideATemporaryStateDirectoryAndBlocksASecondHolder() throws {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "astronomical-supervisor-lock-\(UUID().uuidString)";
        let lockFilePath: String = temporaryStateDirectory + "/daemon.lock";
        let firstLock: AstronomicalInstanceLock = try AstronomicalInstanceLock.acquire(lockFilePath: lockFilePath);
        defer {
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }
        XCTAssertThrowsError(try AstronomicalInstanceLock.acquire(lockFilePath: lockFilePath)) { (thrownError: any Error) in
            XCTAssertEqual(
                thrownError as? AstronomicalInstanceLockError,
                AstronomicalInstanceLockError.alreadyRunning);
        }
        // Dropping the first holder releases the advisory lock with its file
        // descriptor, so the next acquisition for the same path succeeds.
        _ = firstLock;
    }
}
