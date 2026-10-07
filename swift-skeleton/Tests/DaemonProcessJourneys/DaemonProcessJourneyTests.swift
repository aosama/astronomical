import Foundation;

import Testing;

import JourneyCategories;

/**
 * Acceptance journeys for the astronomicald daemon as a process, migrating
 * apps/supervisor/tests/rest_api/daemon_process.rs and
 * daemon_instance_isolation.rs: a second daemon for one instance state is
 * rejected before serving, ambiguous or relative arguments are rejected
 * before startup with the usage text, health survives a missing worker,
 * malformed and retired-configuration files fail startup with a diagnostic,
 * and the stable and development instances start and stop independently.
 *
 * The journeys that script a live worker (unexpected-model readiness,
 * malformed-output progress, responses JSON and SSE through the daemon)
 * stay in the Rust suite until the Swift test-worker family exists — they
 * are recorded as blocked on that family, not as ported.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class DaemonProcessJourneyTests {

    @Test
    func should_reject_a_second_daemon_for_the_same_instance_state() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(journeyName: "second-daemon");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: stateDirectoryPath);
        let runningDaemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: daemonExecutablePath,
            runtimeInstance: "development",
            stateDirectoryPath: stateDirectoryPath);

        let secondDaemon: (exitStatus: Int32, standardError: String) =
            try DaemonProcessJourneySupport.runDaemonExpectingStartupFailure(
                daemonExecutablePath: daemonExecutablePath,
                arguments: ["--instance", "development", "--state-directory", stateDirectoryPath],
                deadlineSeconds: 10);

        #expect(secondDaemon.exitStatus == 1, "the second daemon must refuse the locked instance, got \(secondDaemon.exitStatus)");
        #expect(
            secondDaemon.standardError.contains("already running"),
            "the refusal must name the running instance: \(secondDaemon.standardError)");
        let healthResponse: String? = DaemonProcessJourneySupport.getEndpoint(
            port: runningDaemon.restPort,
            endpointPath: "/health");
        #expect(healthResponse?.hasPrefix("HTTP/1.1 200 OK") == true, "the first daemon keeps serving");
        let firstExitStatus: Int32? = DaemonProcessJourneySupport.terminateAndWait(
            daemonProcess: runningDaemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(firstExitStatus == 0, "SIGTERM must end the daemon cleanly, got \(String(describing: firstExitStatus))");
    }

    @Test
    func should_reject_ambiguous_or_relative_instance_arguments_before_startup() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        for invalidArguments: Array<String> in [
            ["--state-directory", "relative-state"],
            ["--instance", "stable", "--instance", "development"]
        ] {
            let daemonOutcome: (exitStatus: Int32, standardError: String) =
                try DaemonProcessJourneySupport.runDaemonExpectingStartupFailure(
                    daemonExecutablePath: daemonExecutablePath,
                    arguments: invalidArguments,
                    deadlineSeconds: 10);
            #expect(
                daemonOutcome.exitStatus == 2,
                "argument failures exit 2 before startup, got \(daemonOutcome.exitStatus)");
            #expect(
                daemonOutcome.standardError.contains("Usage: astronomicald"),
                "the rejection teaches the fix inline: \(daemonOutcome.standardError)");
        }
    }

    @Test
    func should_keep_health_available_when_worker_startup_fails() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(journeyName: "health-without-worker");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: stateDirectoryPath);
        // No inference worker binary sits beside the test-run daemon, so
        // worker launch fails and the daemon serves from its unavailable
        // supervisor — the exact state this journey proves stays answerable.
        let daemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: daemonExecutablePath,
            runtimeInstance: "development",
            stateDirectoryPath: stateDirectoryPath);

        let healthResponse: String? = DaemonProcessJourneySupport.getEndpoint(
            port: daemon.restPort,
            endpointPath: "/health");
        #expect(healthResponse?.hasPrefix("HTTP/1.1 200 OK") == true, "health must answer while the worker is down");
        let readyResponse: String? = DaemonProcessJourneySupport.getEndpoint(
            port: daemon.restPort,
            endpointPath: "/ready");
        #expect(readyResponse?.hasPrefix("HTTP/1.1 503 ") == true, "readiness must stay unavailable without a worker");
        let exitStatus: Int32? = DaemonProcessJourneySupport.terminateAndWait(
            daemonProcess: daemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(exitStatus == 0, "SIGTERM must end the daemon cleanly, got \(String(describing: exitStatus))");
    }

    @Test
    func should_fail_startup_when_user_config_is_malformed() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(journeyName: "malformed-config");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeRawConfig(
            stateDirectoryPath: stateDirectoryPath,
            configJson: "{\"supervisor\":{\"bind_address\":\"127.0.0.1:0\"}");

        let daemonOutcome: (exitStatus: Int32, standardError: String) =
            try DaemonProcessJourneySupport.runDaemonExpectingStartupFailure(
                daemonExecutablePath: daemonExecutablePath,
                arguments: ["--instance", "development", "--state-directory", stateDirectoryPath],
                deadlineSeconds: 10);

        #expect(daemonOutcome.exitStatus != 0, "a malformed config must fail startup");
        #expect(
            daemonOutcome.standardError.contains("could not resolve the runtime configuration"),
            "the failure must point at the configuration: \(daemonOutcome.standardError)");
    }

    @Test
    func should_reject_retired_supervisor_configuration_for_a_standard_instance() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(journeyName: "retired-field");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeRawConfig(
            stateDirectoryPath: stateDirectoryPath,
            configJson: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[]},"
                + "\"supervisor\":{\"bind_address\":\"127.0.0.1:6732\"}}");

        let daemonOutcome: (exitStatus: Int32, standardError: String) =
            try DaemonProcessJourneySupport.runDaemonExpectingStartupFailure(
                daemonExecutablePath: daemonExecutablePath,
                arguments: ["--instance", "development", "--state-directory", stateDirectoryPath],
                deadlineSeconds: 10);

        #expect(daemonOutcome.exitStatus != 0, "the retired supervisor field must fail startup");
        #expect(
            daemonOutcome.standardError.contains("unknown field"),
            "the failure must name the retired field: \(daemonOutcome.standardError)");
    }

    @Test
    func should_keep_stable_running_while_development_starts_and_stops_independently() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let stableStateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(journeyName: "isolation-stable");
        let developmentStateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(journeyName: "isolation-development");
        defer {
            DaemonProcessJourneySupport.removeStateDirectory(stableStateDirectoryPath);
            DaemonProcessJourneySupport.removeStateDirectory(developmentStateDirectoryPath);
        };
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: stableStateDirectoryPath);
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: developmentStateDirectoryPath);
        let stableDaemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: daemonExecutablePath,
            runtimeInstance: "stable",
            stateDirectoryPath: stableStateDirectoryPath);
        let stableProcessIdentifier: Int32 = stableDaemon.daemonProcess.processIdentifier;
        let developmentDaemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: daemonExecutablePath,
            runtimeInstance: "development",
            stateDirectoryPath: developmentStateDirectoryPath);

        let stableStatus: String? = DaemonProcessJourneySupport.getEndpoint(
            port: stableDaemon.restPort,
            endpointPath: "/v1/status");
        let developmentStatus: String? = DaemonProcessJourneySupport.getEndpoint(
            port: developmentDaemon.restPort,
            endpointPath: "/v1/status");
        #expect(stableStatus?.contains("\"channel\":\"stable\"") == true, "the stable daemon reports its channel");
        #expect(developmentStatus?.contains("\"channel\":\"development\"") == true, "the development daemon reports its channel");
        #expect(stableStatus?.contains("\"state_directory\":\"custom\"") == true, "the stable status names its custom state");
        #expect(developmentStatus?.contains("\"state_directory\":\"custom\"") == true, "the development status names its custom state");
        #expect(stableStatus?.contains("\"version\"") == true, "the status reports the build version");

        let developmentShutdown: String? = DaemonProcessJourneySupport.postEmptyEndpoint(
            port: developmentDaemon.restPort,
            endpointPath: "/v1/control/shutdown");
        #expect(developmentShutdown?.hasPrefix("HTTP/1.1 202 Accepted") == true, "the graceful shutdown answers 202: \(developmentShutdown ?? "")");
        let developmentExitStatus: Int32? = DaemonProcessJourneySupport.waitTerminationWithoutSignal(
            daemonProcess: developmentDaemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(developmentExitStatus == 0, "the development daemon must stop on request, got \(String(describing: developmentExitStatus))");

        #expect(stableDaemon.daemonProcess.isRunning, "the stable daemon must outlive the development shutdown");
        #expect(stableDaemon.daemonProcess.processIdentifier == stableProcessIdentifier, "the stable daemon must be the same process");
        let stableHealth: String? = DaemonProcessJourneySupport.getEndpoint(
            port: stableDaemon.restPort,
            endpointPath: "/health");
        #expect(stableHealth?.hasPrefix("HTTP/1.1 200 OK") == true, "the stable daemon keeps answering health");
        let stableExitStatus: Int32? = DaemonProcessJourneySupport.terminateAndWait(
            daemonProcess: stableDaemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(stableExitStatus == 0, "SIGTERM must end the stable daemon cleanly, got \(String(describing: stableExitStatus))");
    }
}
