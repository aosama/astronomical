import Testing;

import Foundation;

import IpcProtocol;
import AstronomicalConfig;
import JourneyCategories;

@testable import Supervisor;
@testable import AstronomicalConfig;

/// Hermetic coverage for the daemon argument parser and the single-instance
/// lock. Every journey uses temporary directories or fictional placeholder
/// paths; the Development instance is the only instance these journeys may
/// name.
@Suite(.serialized, .tags(.hermeticJourney))
final class SupervisorTests {

    @Test
    func should_default_empty_arguments_to_the_development_instance() throws {
        let command: DaemonCommand = try DaemonArguments.parse(processArguments: ["astronomicald"]);
        guard case let .run(daemonArguments) = command else {
            Issue.record("empty arguments should produce a run command");
            return;
        }
        #expect(daemonArguments.runtimeInstance == AstronomicalRuntimeInstance.development);
        #expect(daemonArguments.stateDirectoryOverride == nil);
    }

    @Test
    func should_accept_an_explicit_instance_argument_once() throws {
        let command: DaemonCommand = try DaemonArguments.parse(processArguments: [
            "astronomicald", "--instance", "development",
        ]);
        guard case let .run(daemonArguments) = command else {
            Issue.record("instance arguments should produce a run command");
            return;
        }
        #expect(daemonArguments.runtimeInstance == AstronomicalRuntimeInstance.development);
    }

    @Test
    func should_reject_an_unknown_instance_value() throws {
        do {
            _ = try DaemonArguments.parse(processArguments: [
                "astronomicald", "--instance", "canary",
            ]);
            Issue.record("an unknown instance value must be rejected");
        } catch {
            #expect(
                error as? DaemonArgumentError
                    == DaemonArgumentError.invalidInstance(rawValue: "canary"));
        }
    }

    @Test
    func should_reject_a_repeated_instance_argument() throws {
        do {
            _ = try DaemonArguments.parse(processArguments: [
                "astronomicald", "--instance", "development", "--instance", "stable",
            ]);
            Issue.record("a repeated instance argument must be rejected");
        } catch {
            #expect(
                error as? DaemonArgumentError
                    == DaemonArgumentError.repeatedArgument(argumentName: "--instance"));
        }
    }

    @Test
    func should_reject_a_missing_instance_value() throws {
        do {
            _ = try DaemonArguments.parse(processArguments: [
                "astronomicald", "--instance",
            ]);
            Issue.record("a missing instance value must be rejected");
        } catch {
            #expect(
                error as? DaemonArgumentError
                    == DaemonArgumentError.missingValue(argumentName: "--instance"));
        }
    }

    @Test
    func should_reject_a_relative_state_directory() throws {
        do {
            _ = try DaemonArguments.parse(processArguments: [
                "astronomicald", "--state-directory", "relative/state",
            ]);
            Issue.record("a relative state directory must be rejected");
        } catch {
            #expect(
                error as? DaemonArgumentError
                    == DaemonArgumentError.invalidStateDirectory(path: "relative/state"));
        }
    }

    @Test
    func should_reject_the_root_state_directory() throws {
        do {
            _ = try DaemonArguments.parse(processArguments: [
                "astronomicald", "--state-directory", "/",
            ]);
            Issue.record("the root state directory must be rejected");
        } catch {
            #expect(
                error as? DaemonArgumentError
                    == DaemonArgumentError.invalidStateDirectory(path: "/"));
        }
    }

    @Test
    func should_reject_an_unknown_argument() throws {
        do {
            _ = try DaemonArguments.parse(processArguments: [
                "astronomicald", "--daemonize",
            ]);
            Issue.record("an unknown argument must be rejected");
        } catch {
            #expect(
                error as? DaemonArgumentError
                    == DaemonArgumentError.unknownArgument(argument: "--daemonize"));
        }
    }

    @Test
    func should_short_circuit_help_and_version_before_validation() throws {
        #expect(try DaemonArguments.parse(processArguments: ["astronomicald", "--help"]) == DaemonCommand.help);
        #expect(try DaemonArguments.parse(processArguments: ["astronomicald", "-h"]) == DaemonCommand.help);
        #expect(try DaemonArguments.parse(processArguments: ["astronomicald", "--version"]) == DaemonCommand.version);
        #expect(DaemonArguments.helpText().contains("--instance"));
    }

    @Test
    func should_resolve_a_state_directory_override_through_the_override() throws {
        let overrideDirectory: String = "/astronomical-test/fictional-state-root";
        let command: DaemonCommand = try DaemonArguments.parse(processArguments: [
            "astronomicald", "--state-directory", overrideDirectory,
        ]);
        guard case let .run(daemonArguments) = command else {
            Issue.record("state-directory arguments should produce a run command");
            return;
        }
        let instancePaths: AstronomicalInstancePaths = try daemonArguments.resolveInstancePaths();
        #expect(instancePaths.stateDirectory.string.hasPrefix(overrideDirectory));
    }

    @Test
    func should_acquire_the_instance_lock_inside_a_temporary_state_directory_and_block_a_second_holder() throws {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "astronomical-supervisor-lock-\(UUID().uuidString)";
        let lockFilePath: String = temporaryStateDirectory + "/daemon.lock";
        let firstLock: AstronomicalInstanceLock = try AstronomicalInstanceLock.acquire(lockFilePath: lockFilePath);
        defer {
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }
        do {
            _ = try AstronomicalInstanceLock.acquire(lockFilePath: lockFilePath);
            Issue.record("a second acquisition of the same lock must be rejected");
        } catch {
            #expect(
                error as? AstronomicalInstanceLockError
                    == AstronomicalInstanceLockError.alreadyRunning);
        }
        // Dropping the first holder releases the advisory lock with its file
        // descriptor, so the next acquisition for the same path succeeds.
        _ = firstLock;
    }

    @Test
    func should_track_the_launched_worker_pid_and_terminate_it_on_a_command_side_close() throws {
        // /bin/cat mirrors the worker's transport shape: it holds its stdin
        // open and exits when the supervisor half-closes the command side.
        let workerProcess: WorkerProcess = try WorkerProcess.launch(workerExecutablePath: "/bin/cat");
        #expect(workerProcess.processId != nil);
        _ = try workerProcess.close();
        #expect(workerProcess.processId == nil);
    }

    @Test
    func should_escalate_to_a_signal_for_a_worker_that_ignores_eof() throws {
        // sleep never reads stdin, so half-closing the command side cannot
        // end it; the escalation path must SIGTERM it out of existence inside
        // the shutdown timeout.
        let workerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: "/bin/sleep",
            arguments: ["30"]);
        #expect(workerProcess.processId != nil);
        _ = try workerProcess.close();
        #expect(workerProcess.processId == nil);
    }

    @Test
    func should_answer_handshake_and_status_over_the_daemon_ipc_service_and_clean_its_socket_on_shutdown() throws {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "asup-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let service: DaemonIpcService = try DaemonIpcService.start(
            instancePaths: instancePaths,
            healthProvider: { return DaemonStatusReport.unavailable(); });
        defer {
            service.shutdown();
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }
        #expect(FileManager.default.fileExists(atPath: service.socketPath));

        let handshakeClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try handshakeClient.sendRequest(DaemonRequest.handshake);
        #expect(
            try handshakeClient.nextResponse()
                == DaemonResponse.handshakeAccepted(
                    protocolVersion: DaemonProtocol.protocolVersion,
                    applicationName: DaemonProtocol.applicationName));

        let statusClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try statusClient.sendRequest(DaemonRequest.status);
        #expect(
            try statusClient.nextResponse()
                == DaemonResponse.status(
                    workerStatus: DaemonWorkerStatus.unavailable,
                    readyModelId: nil,
                    defaultModelId: DefaultModel.builtinDefaultModelId));

        service.shutdown();
        #expect(!FileManager.default.fileExists(atPath: service.socketPath));
    }
}
