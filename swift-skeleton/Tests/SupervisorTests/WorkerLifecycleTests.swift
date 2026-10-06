import Testing;

import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/// Hermetic coverage of supervisor-owned worker lifecycle: graceful close
/// with outcome reporting, failure containment, and the relaunch recovery
/// acknowledgement, driven by fake `/bin/bash` workers over real pipes.
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerLifecycleTests {

    @Test
    func should_close_an_eof_responsive_worker_gracefully() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/cat");
        let healthState: WorkerHealthState = WorkerHealthState();

        let terminationOutcome: WorkerTerminationOutcome = try WorkerLifecycle.shutdown(
            workerProcess: workerProcess,
            healthState: healthState);

        #expect(terminationOutcome == .graceful(processExitSuccessful: true));
        #expect(!workerProcess.hasLivingProcess());
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .unavailable, readyModelId: nil));
    }

    @Test
    func should_escalate_when_the_worker_ignores_eof() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", "exec sleep 30\n"]);
        let healthState: WorkerHealthState = WorkerHealthState();

        let terminationOutcome: WorkerTerminationOutcome = try WorkerLifecycle.shutdown(
            workerProcess: workerProcess,
            healthState: healthState);

        #expect(terminationOutcome == .forced(processExitSuccessful: false));
        #expect(!workerProcess.hasLivingProcess());
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .unavailable, readyModelId: nil));
    }

    @Test
    func should_skip_the_close_for_an_already_exited_process() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", "exec sleep 30\n"]);
        _ = try workerProcess.forceTerminate();
        #expect(!workerProcess.hasLivingProcess());
        let healthState: WorkerHealthState = WorkerHealthState();

        let terminationOutcome: WorkerTerminationOutcome = try WorkerLifecycle.shutdown(
            workerProcess: workerProcess,
            healthState: healthState);

        #expect(terminationOutcome == .graceful(processExitSuccessful: true));
    }

    @Test
    func should_force_terminate_on_containment_and_mark_health_unavailable() throws {
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/bash",
            arguments: ["-c", "exec sleep 30\n"]);
        let healthState: WorkerHealthState = WorkerHealthState();

        WorkerLifecycle.containFailure(
            workerProcess: workerProcess,
            healthState: healthState,
            operationFailure: WorkerControlError.workerEventStreamClosed);

        #expect(!workerProcess.hasLivingProcess());
        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .unavailable, readyModelId: nil));
    }

    @Test
    func should_reset_an_ignored_termination_signal_in_the_spawned_worker() throws {
        // An ignored disposition is the one signal state that survives exec,
        // so a supervisor that ignores SIGTERM would otherwise spawn workers
        // immune to the graceful escalation rung; the spawn must reset it.
        // Without the reset this journey takes the full SIGKILL ladder
        // (~10 seconds) instead of ending at SIGTERM (~5 seconds).
        signal(SIGTERM, SIG_IGN);
        defer {
            signal(SIGTERM, SIG_DFL);
        }
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/sleep",
            arguments: ["30"]);

        let closeStartedAt: Date = Date();
        let terminationOutcome: WorkerTerminationOutcome = try workerProcess.close();
        let closeElapsedSeconds: TimeInterval = Date().timeIntervalSince(closeStartedAt);

        #expect(terminationOutcome == .forced(processExitSuccessful: false));
        #expect(
            closeElapsedSeconds < 8,
            "SIGTERM must end the worker in the first escalation rung, took \(closeElapsedSeconds)s");
    }

    @Test
    func should_relaunch_after_failure_and_wait_for_the_replacement_acknowledgement() throws {
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

        #expect(
            healthState.daemonStatusReport()
                == DaemonStatusReport(workerStatus: .ready, readyModelId: nil));
        #expect(
            healthState.currentSnapshot().workerRuntimeFeatureConfiguration?.configurationGeneration
                == "gen-1");
        #expect(healthState.hasAcknowledgedLifecycle());
        _ = try workerProcess.close();
    }

    @Test
    func should_time_out_relaunch_recovery_against_a_silent_replacement() throws {
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

        do {
            try WorkerLifecycle.relaunchAfterFailure(
                workerProcess: workerProcess,
                eventPump: eventPump,
                healthState: healthState,
                recoveryAcknowledgementTimeout: 1);
            Issue.record("expected a candidate acknowledgement timeout");
        } catch {
            guard case WorkerControlError.candidateAcknowledgementTimeout = error else {
                Issue.record(Comment(stringLiteral: "expected a candidate acknowledgement timeout, got \(error)"));
                return;
            }
        }
        _ = try workerProcess.close();
    }
}

/// The worker executable is located beside the running daemon binary; the
/// mechanism contract is the platform-stable file name.
@Suite(.tags(.hermeticJourney))
final class FallbackWorkerExecutablePathTests {

    @Test
    func should_derive_the_worker_executable_with_the_platform_stable_name() throws {
        let workerExecutablePath: FilePath = try FallbackWorkerExecutablePath.derive();
        #expect(
            workerExecutablePath.string.hasSuffix("/" + FallbackWorkerExecutablePath.workerExecutableName),
            "worker path should end with the worker binary name: \(workerExecutablePath)");
    }
}
