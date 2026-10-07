import Foundation;

import Testing;

import Supervisor;
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
    func should_serve_the_library_catalog_console_and_auto_discovery_in_the_real_daemon() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(journeyName: "library-daemon");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: stateDirectoryPath);
        // A pre-published library model: the exact fixture shape discovery
        // recognizes, so /v1/models and the catalog readiness join answer
        // from disk before any download ran.
        let publishedModelDirectory: String = stateDirectoryPath + "/models/astronomical-test/example-qwen";
        try FileManager.default.createDirectory(atPath: publishedModelDirectory, withIntermediateDirectories: true);
        let modelConfigJson: String = "{\"model_type\":\"qwen3_5_moe\",\"text_config\":{\"max_position_embeddings\":262144}}";
        try modelConfigJson.write(toFile: publishedModelDirectory + "/config.json", atomically: true, encoding: .utf8);
        try "{\"version\":1,\"model\":{\"type\":\"BPE\"}}".write(
            toFile: publishedModelDirectory + "/tokenizer.json", atomically: true, encoding: .utf8);
        try "fictional-shard".write(toFile: publishedModelDirectory + "/model-00001.safetensors", atomically: true, encoding: .utf8);
        try "{\"metadata\":{\"total_size\":15},\"weight_map\":{\"model.embed_tokens.weight\":\"model-00001.safetensors\"}}".write(
            toFile: publishedModelDirectory + "/model.safetensors.index.json", atomically: true, encoding: .utf8);

        let runningDaemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: daemonExecutablePath,
            runtimeInstance: "development",
            stateDirectoryPath: stateDirectoryPath);

        let catalogResponse: String? = DaemonProcessJourneySupport.waitUntilEndpointContains(
            port: runningDaemon.restPort,
            endpointPath: "/v1/library/catalog",
            expectedFragment: "\"schema_version\":2", deadlineSeconds: 10);
        #expect(catalogResponse?.hasPrefix("HTTP/1.1 200 OK") == true, "the bundled catalog serves from the real daemon");

        let consoleResponse: String? = DaemonProcessJourneySupport.getEndpoint(
            port: runningDaemon.restPort,
            endpointPath: "/library");
        #expect(
            consoleResponse?.contains("data-observatory-view=\"library\"") == true,
            "the console library view is served by the daemon");

        let modelsResponse: String? = DaemonProcessJourneySupport.waitUntilEndpointContains(
            port: runningDaemon.restPort,
            endpointPath: "/v1/models",
            expectedFragment: "example-qwen", deadlineSeconds: 10);
        #expect(modelsResponse != nil, "auto-discovery lists the published library model");
        // Attribution stays off by default: a serving daemon must not leave a
        // supervisor attribution file behind in the instance state.
        #expect(
            FileManager.default.fileExists(
                atPath: stateDirectoryPath + "/logs/supervisor-performance-attribution.jsonl") == false,
            "a disabled attribution flag must not create the attribution file");

        let exitStatus: Int32? = DaemonProcessJourneySupport.terminateAndWait(
            daemonProcess: runningDaemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(exitStatus == 0);
    }

    @Test
    func should_record_the_startup_catalog_load_when_attribution_is_enabled() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(
            journeyName: "library-attribution");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeRawConfig(
            stateDirectoryPath: stateDirectoryPath,
            configJson: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[]},"
                + "\"chunking\":{\"fixed_prompt_processing_chunk_size_tokens\":2048},"
                + "\"diagnostics\":{\"performance_attribution_enabled\":true}}");

        let runningDaemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: daemonExecutablePath,
            runtimeInstance: "development",
            stateDirectoryPath: stateDirectoryPath);

        // spawnDaemon returned only after the startup line, so the catalog
        // attribution row must already be flushed to disk before serving.
        let attributionText: String = try String(
            contentsOfFile: stateDirectoryPath + "/logs/supervisor-performance-attribution.jsonl",
            encoding: .utf8);
        let parsedAttributionDocument: Any = try JSONSerialization.jsonObject(
            with: Data(attributionText.trimmingCharacters(in: .whitespacesAndNewlines).utf8),
            options: []);
        guard let attributionRecord: [String: Any] = parsedAttributionDocument as? [String: Any] else {
            Issue.record("the startup attribution record should contain a JSON object: \(attributionText)");
            return;
        }
        #expect(attributionRecord["operation"] as? String == "library_catalog_load");
        #expect(attributionRecord["outcome"] as? String == "success");
        let bundledCatalog: DownloadCatalog = try DownloadCatalog.loadBundled();
        #expect(
            attributionRecord["catalog_entry_count"] as? Int == bundledCatalog.entryCount,
            "the attribution row must carry the bundled catalog's entry count");

        let exitStatus: Int32? = DaemonProcessJourneySupport.terminateAndWait(
            daemonProcess: runningDaemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(exitStatus == 0);
    }

    @Test
    func should_fail_before_binding_when_the_startup_attribution_file_cannot_open() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(
            journeyName: "library-attribution-refused");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeRawConfig(
            stateDirectoryPath: stateDirectoryPath,
            configJson: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[]},"
                + "\"chunking\":{\"fixed_prompt_processing_chunk_size_tokens\":2048},"
                + "\"diagnostics\":{\"performance_attribution_enabled\":true}}");
        // A directory occupies the attribution file path, so the open must
        // fail and the daemon must refuse to serve anything.
        try FileManager.default.createDirectory(
            atPath: stateDirectoryPath + "/logs/supervisor-performance-attribution.jsonl",
            withIntermediateDirectories: true);

        let refusedDaemon: (exitStatus: Int32, standardOutput: String, standardError: String) =
            try DaemonProcessJourneySupport.runDaemonExpectingStartupFailure(
                daemonExecutablePath: daemonExecutablePath,
                arguments: ["--instance", "development", "--state-directory", stateDirectoryPath],
                deadlineSeconds: 10);

        #expect(refusedDaemon.exitStatus != 0, "an unopenable attribution file must fail startup");
        #expect(
            refusedDaemon.standardOutput.contains("serving REST on http://") == false,
            "the daemon must never announce a bound endpoint");
        #expect(
            refusedDaemon.standardError.contains("failed to create the supervisor performance-attribution log"),
            "the refusal must name the attribution log failure: \(refusedDaemon.standardError)");
    }

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

        let secondDaemon: (exitStatus: Int32, standardOutput: String, standardError: String) =
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
            let daemonOutcome: (exitStatus: Int32, standardOutput: String, standardError: String) =
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
    func should_persist_the_exact_worker_stderr_when_the_worker_becomes_unavailable() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let probeWorkerExecutablePath: String = try DaemonProcessJourneySupport.locateBuiltExecutable(
            executableName: "SupervisorStderrProbeWorker");
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(
            journeyName: "stderr-diagnostic");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: stateDirectoryPath);
        // A synthetic bundle: the daemon and the probe worker copied beside
        // each other, so the daemon's worker-path derivation finds the probe
        // — the worker that writes its stderr diagnostic, idles, and exits.
        // The SwiftPM resource bundles ride along beside the binaries: Rust
        // embeds its assets, Swift loads them from the bundle directories
        // next to the executable.
        let bundleBinDirectoryPath: String = stateDirectoryPath + "/bin";
        try FileManager.default.createDirectory(
            atPath: bundleBinDirectoryPath,
            withIntermediateDirectories: true);
        try FileManager.default.copyItem(
            atPath: daemonExecutablePath,
            toPath: bundleBinDirectoryPath + "/" + DaemonProcessJourneySupport.daemonExecutableName);
        try FileManager.default.copyItem(
            atPath: probeWorkerExecutablePath,
            toPath: bundleBinDirectoryPath + "/astronomical-inference-worker");
        let daemonProductsDirectoryPath: String =
            (daemonExecutablePath as NSString).deletingLastPathComponent;
        let productsEntryNames: Array<String> = try FileManager.default.contentsOfDirectory(
            atPath: daemonProductsDirectoryPath);
        for productsEntryName: String in productsEntryNames {
            if (productsEntryName as NSString).pathExtension == "bundle" {
                try FileManager.default.copyItem(
                    atPath: daemonProductsDirectoryPath + "/" + productsEntryName,
                    toPath: bundleBinDirectoryPath + "/" + productsEntryName);
            }
        }
        let daemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: bundleBinDirectoryPath + "/" + DaemonProcessJourneySupport.daemonExecutableName,
            runtimeInstance: "development",
            stateDirectoryPath: stateDirectoryPath);

        // The bound covers the exit-diagnostics settle plus test-machine
        // spawn jitter; a healthy containment lands in well under half of it.
        let diagnosticDeadline: Date = Date().addingTimeInterval(10);
        var supervisorLogText: String = "";
        while (Date() < diagnosticDeadline) {
            supervisorLogText = DaemonProcessJourneySupport.readSupervisorLogs(
                stateDirectoryPath: stateDirectoryPath);
            if supervisorLogText.contains("worker process exited after closing its event stream") {
                break;
            }
            Thread.sleep(forTimeInterval: 0.025);
        }
        #expect(
            supervisorLogText.contains("worker process exited after closing its event stream"),
            "the supervisor log must retain the worker exit diagnostic");
        #expect(
            supervisorLogText.contains("exit code 0"),
            "the supervisor log must retain the worker exit status");
        #expect(
            supervisorLogText.contains("stderr-probe worker observed visible stderr"),
            "the supervisor log must retain the exact worker stderr tail");
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

        let daemonOutcome: (exitStatus: Int32, standardOutput: String, standardError: String) =
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

        let daemonOutcome: (exitStatus: Int32, standardOutput: String, standardError: String) =
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
