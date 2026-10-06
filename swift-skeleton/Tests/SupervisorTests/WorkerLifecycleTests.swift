import XCTest;

import Foundation;

import AstronomicalConfig;
import IpcProtocol;

@testable import Supervisor;

/// Hermetic coverage of supervisor-owned worker lifecycle: graceful close
/// with outcome reporting, failure containment, and the relaunch recovery
/// acknowledgement, driven by fake `/bin/bash` workers over real pipes.
final class WorkerLifecycleTests: XCTestCase {

    func testShutdownClosesAnEofResponsiveWorkerGracefully() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/cat");
        let healthState: WorkerHealthState = WorkerHealthState();

        let terminationOutcome: WorkerTerminationOutcome = try WorkerLifecycle.shutdown(
            workerProcess: workerProcess,
            healthState: healthState);

        XCTAssertEqual(terminationOutcome, .graceful(processExitSuccessful: true));
        XCTAssertFalse(workerProcess.hasLivingProcess());
        XCTAssertEqual(
            healthState.daemonStatusReport(),
            DaemonStatusReport(workerStatus: .unavailable, readyModelId: nil));
    }

    func testShutdownEscalatesWhenTheWorkerIgnoresEof() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", "exec sleep 30\n"]);
        let healthState: WorkerHealthState = WorkerHealthState();

        let terminationOutcome: WorkerTerminationOutcome = try WorkerLifecycle.shutdown(
            workerProcess: workerProcess,
            healthState: healthState);

        XCTAssertEqual(terminationOutcome, .forced(processExitSuccessful: false));
        XCTAssertFalse(workerProcess.hasLivingProcess());
        XCTAssertEqual(
            healthState.daemonStatusReport(),
            DaemonStatusReport(workerStatus: .unavailable, readyModelId: nil));
    }

    func testShutdownSkipsTheCloseForAnAlreadyExitedProcess() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", "exec sleep 30\n"]);
        _ = try workerProcess.forceTerminate();
        XCTAssertFalse(workerProcess.hasLivingProcess());
        let healthState: WorkerHealthState = WorkerHealthState();

        let terminationOutcome: WorkerTerminationOutcome = try WorkerLifecycle.shutdown(
            workerProcess: workerProcess,
            healthState: healthState);

        XCTAssertEqual(terminationOutcome, .graceful(processExitSuccessful: true));
    }

    func testContainmentForceTerminatesAndMarksHealthUnavailable() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", "exec sleep 30\n"]);
        let healthState: WorkerHealthState = WorkerHealthState();

        WorkerLifecycle.containFailure(
            workerProcess: workerProcess,
            healthState: healthState,
            operationFailure: WorkerControlError.workerEventStreamClosed);

        XCTAssertFalse(workerProcess.hasLivingProcess());
        XCTAssertEqual(
            healthState.daemonStatusReport(),
            DaemonStatusReport(workerStatus: .unavailable, readyModelId: nil));
    }

    func testRelaunchAfterFailureWaitsForTheReplacementAcknowledgement() throws {
        let markerFilePath: String = NSTemporaryDirectory()
            + "asup-relaunch-\(UUID().uuidString.prefix(8))";
        defer {
            try? FileManager.default.removeItem(atPath: markerFilePath);
        }
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + "if [ -f '\(markerFilePath)' ]; then\n"
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.modelLessRuntimePolicyPayload())
            + "  exec sleep 30\n"
            + "else\n"
            + "  touch '\(markerFilePath)'\n"
            + "fi\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript],
            workerStartupConfiguration: WorkerStartupHandshakeTests.startupConfiguration());
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);
        let healthState: WorkerHealthState = WorkerHealthState();
        // The first run only lays the marker and exits; recovery relaunches
        // the same launch inputs into the acknowledging branch.
        let exitDeadline: Date = Date().addingTimeInterval(5);
        while workerProcess.hasLivingProcess() && Date() < exitDeadline {
            Thread.sleep(forTimeInterval: 0.02);
        }
        _ = try workerProcess.forceTerminate();

        try WorkerLifecycle.relaunchAfterFailure(
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: healthState,
            recoveryAcknowledgementTimeout: 10);

        XCTAssertEqual(
            healthState.daemonStatusReport(),
            DaemonStatusReport(workerStatus: .ready, readyModelId: nil));
        XCTAssertEqual(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration?.configurationGeneration,
            "gen-1");
        XCTAssertTrue(healthState.hasAcknowledgedLifecycle());
        _ = try workerProcess.close();
    }

    func testRelaunchRecoveryTimesOutAgainstASilentReplacement() throws {
        let markerFilePath: String = NSTemporaryDirectory()
            + "asup-silent-\(UUID().uuidString.prefix(8))";
        defer {
            try? FileManager.default.removeItem(atPath: markerFilePath);
        }
        let fakeWorkerScript: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + "if [ -f '\(markerFilePath)' ]; then\n"
            + "  exec sleep 30\n"
            + "else\n"
            + "  touch '\(markerFilePath)'\n"
            + "fi\n";
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", fakeWorkerScript],
            workerStartupConfiguration: WorkerStartupHandshakeTests.startupConfiguration());
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: workerProcess);
        let healthState: WorkerHealthState = WorkerHealthState();
        let exitDeadline: Date = Date().addingTimeInterval(5);
        while workerProcess.hasLivingProcess() && Date() < exitDeadline {
            Thread.sleep(forTimeInterval: 0.02);
        }
        _ = try workerProcess.forceTerminate();

        XCTAssertThrowsError(try WorkerLifecycle.relaunchAfterFailure(
            workerProcess: workerProcess,
            eventPump: eventPump,
            healthState: healthState,
            recoveryAcknowledgementTimeout: 1)) { (thrownError: any Error) in
            guard case WorkerControlError.candidateAcknowledgementTimeout = thrownError else {
                return XCTFail("expected a candidate acknowledgement timeout, got \(thrownError)");
            }
        }
        _ = try workerProcess.close();
    }
}

/// The worker executable is located beside the running daemon binary; the
/// mechanism contract is the platform-stable file name.
final class FallbackWorkerExecutablePathTests: XCTestCase {

    func testDerivedWorkerExecutableUsesThePlatformStableName() throws {
        let workerExecutablePath: FilePath = try FallbackWorkerExecutablePath.derive();
        XCTAssertEqual(
            workerExecutablePath.string.hasSuffix("/" + FallbackWorkerExecutablePath.workerExecutableName),
            true,
            "worker path should end with the worker binary name: \(workerExecutablePath)");
    }
}
