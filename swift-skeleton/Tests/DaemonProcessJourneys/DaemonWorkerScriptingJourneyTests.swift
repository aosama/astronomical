import Foundation;

import Testing;

import JourneyCategories;

/**
 * Acceptance journeys for the daemon serving scripted worker fixtures,
 * migrating the test-worker arms of
 * apps/supervisor/tests/rest_api/daemon_process.rs: the daemon stays ready
 * with whichever model a request loads and advertises it, generation
 * progress stays observable on the status document while a
 * malformed-output stream is in flight before its error frame, and the
 * Responses surface answers JSON and SSE through the daemon process. The
 * worker is the built supervisor test worker beside the daemon in a
 * synthetic bundle; the scripted model identities are discoverable
 * directory-leaf fixtures, exactly the Rust fixture's model-id steering.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class DaemonWorkerScriptingJourneyTests {


    /// The worker becomes ready with whichever model a client request loads,
    /// and the daemon advertises that model — no hardcoded expected identity
    /// rejects the loaded model anymore.
    @Test
    func should_stay_ready_with_whichever_model_the_worker_loads() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let workerExecutablePath: String = try DaemonProcessJourneySupport.locateBuiltExecutable(
            executableName: "SupervisorIdleWorker");
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(
            journeyName: "ready-model");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: stateDirectoryPath);
        try DaemonProcessJourneySupport.writeDiscoveredChatModelFixture(
            stateDirectoryPath: stateDirectoryPath,
            modelDirectoryName: "test-worker-model");
        let bundleBinDirectoryPath: String = try DaemonProcessJourneySupport.makeWorkerBearingBundleDirectory(
            stateDirectoryPath: stateDirectoryPath,
            daemonExecutablePath: daemonExecutablePath,
            workerExecutablePath: workerExecutablePath);
        let runningDaemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: bundleBinDirectoryPath + "/" + DaemonProcessJourneySupport.daemonExecutableName,
            runtimeInstance: "development",
            stateDirectoryPath: stateDirectoryPath);

        // One request loads the discovered model onto the worker; its plain
        // completion also proves the daemon serves the loaded model.
        let chatResponse: String? = DaemonProcessJourneySupport.postJsonEndpoint(
            port: runningDaemon.restPort,
            endpointPath: "/v1/chat/completions",
            bodyJson: "{\"model\":\"test-worker-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":4}");
        #expect(chatResponse?.hasPrefix("HTTP/1.1 200 OK") == true, "the loaded model must serve, got: \(chatResponse ?? "no response")");

        let readyResponse: String? = DaemonProcessJourneySupport.waitUntilEndpointContains(
            port: runningDaemon.restPort,
            endpointPath: "/ready",
            expectedFragment: "HTTP/1.1 200 OK",
            deadlineSeconds: 5);
        #expect(readyResponse?.hasPrefix("HTTP/1.1 200 OK") == true, "the daemon must be ready with the loaded model");

        let modelsResponse: String? = DaemonProcessJourneySupport.getEndpoint(
            port: runningDaemon.restPort,
            endpointPath: "/v1/models");
        #expect(
            modelsResponse?.contains("test-worker-model") == true,
            "the loaded model must be advertised, got: \(modelsResponse ?? "no response")");

        let exitStatus: Int32? = DaemonProcessJourneySupport.terminateAndWait(
            daemonProcess: runningDaemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(exitStatus == 0);
    }

    /// While a malformed-output stream is in flight, the status document
    /// reports the generating phase with its token progress; the stream then
    /// fails with the malformed-output error frame and no terminator, and
    /// the status settles back to idle without a progress section.
    @Test
    func should_show_generation_progress_for_malformed_model_output_before_stream_failure() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let workerExecutablePath: String = try DaemonProcessJourneySupport.locateBuiltExecutable(
            executableName: "SupervisorIdleWorker");
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(
            journeyName: "malformed-progress");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: stateDirectoryPath);
        try DaemonProcessJourneySupport.writeDiscoveredChatModelFixture(
            stateDirectoryPath: stateDirectoryPath,
            modelDirectoryName: "delayed-malformed-output-fixture");
        let bundleBinDirectoryPath: String = try DaemonProcessJourneySupport.makeWorkerBearingBundleDirectory(
            stateDirectoryPath: stateDirectoryPath,
            daemonExecutablePath: daemonExecutablePath,
            workerExecutablePath: workerExecutablePath);
        let runningDaemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: bundleBinDirectoryPath + "/" + DaemonProcessJourneySupport.daemonExecutableName,
            runtimeInstance: "development",
            stateDirectoryPath: stateDirectoryPath);
        let chatPort: UInt16 = runningDaemon.restPort;

        let chatResponseBox: ChatResponseBox = ChatResponseBox();
        let chatRequestThread: Thread = Thread {
            chatResponseBox.replace(with: DaemonProcessJourneySupport.postJsonEndpoint(
                port: chatPort,
                endpointPath: "/v1/chat/completions",
                bodyJson: "{\"model\":\"delayed-malformed-output-fixture\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true,\"max_tokens\":16}") ?? "");
        };
        chatRequestThread.name = "astronomical-journey-chat-request";
        chatRequestThread.start();

        let activeStatusResponse: String? = DaemonProcessJourneySupport.waitUntilEndpointContains(
            port: runningDaemon.restPort,
            endpointPath: "/v1/status",
            expectedFragment: "\"phase\":\"generation\"",
            deadlineSeconds: 5);
        #expect(
            activeStatusResponse?.contains("\"status\":\"ready\"") == true,
            "an in-flight generation keeps the daemon ready, got: \(activeStatusResponse ?? "no response")");
        #expect(
            activeStatusResponse?.contains("\"activity\":\"generating\"") == true,
            "the activity must report the generating phase");
        #expect(
            activeStatusResponse?.contains("\"processed_tokens\":3") == true,
            "the status must carry the scripted token progress");
        #expect(
            activeStatusResponse?.contains("\"total_tokens\":16") == true,
            "the status must carry the requested output budget");

        let chatResponse: String = chatResponseBox.waitForText(deadlineSeconds: 10);
        #expect(chatResponse.hasPrefix("HTTP/1.1 200 OK"), "the stream must open, got: \(chatResponse)");
        #expect(
            chatResponse.contains("\"code\":\"chat_malformed_model_output\"") == true,
            "the malformed output must surface as its error code");
        #expect(chatResponse.contains("[DONE]") == false, "a failed stream carries no terminator");

        let finalStatusResponse: String? = DaemonProcessJourneySupport.waitUntilEndpointContains(
            port: runningDaemon.restPort,
            endpointPath: "/v1/status",
            expectedFragment: "\"activity\":\"idle\"",
            deadlineSeconds: 5);
        #expect(
            finalStatusResponse?.contains("\"status\":\"ready\"") == true,
            "the daemon must stay ready after the failure");
        #expect(
            finalStatusResponse?.contains("\"progress\"") == false,
            "the settled status must not claim active progress");

        let exitStatus: Int32? = DaemonProcessJourneySupport.terminateAndWait(
            daemonProcess: runningDaemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(exitStatus == 0);
    }

    /// The Responses surface answers through the daemon process itself: the
    /// JSON object with the scripted output text, and the SSE event sequence
    /// from creation to completion without an OpenAI chat terminator.
    @Test
    func should_serve_responses_json_and_sse_through_the_daemon_process() throws {
        let daemonExecutablePath: String = try DaemonProcessJourneySupport.locateDaemonExecutable();
        let workerExecutablePath: String = try DaemonProcessJourneySupport.locateBuiltExecutable(
            executableName: "SupervisorIdleWorker");
        let stateDirectoryPath: String = try DaemonProcessJourneySupport.makeStateDirectory(
            journeyName: "responses-daemon");
        defer { DaemonProcessJourneySupport.removeStateDirectory(stateDirectoryPath); };
        try DaemonProcessJourneySupport.writeInstanceConfig(stateDirectoryPath: stateDirectoryPath);
        try DaemonProcessJourneySupport.writeDiscoveredChatModelFixture(
            stateDirectoryPath: stateDirectoryPath,
            modelDirectoryName: "accepted-chat-fixture");
        let bundleBinDirectoryPath: String = try DaemonProcessJourneySupport.makeWorkerBearingBundleDirectory(
            stateDirectoryPath: stateDirectoryPath,
            daemonExecutablePath: daemonExecutablePath,
            workerExecutablePath: workerExecutablePath);
        let runningDaemon: (daemonProcess: Process, restPort: UInt16) = try DaemonProcessJourneySupport.spawnDaemon(
            daemonExecutablePath: bundleBinDirectoryPath + "/" + DaemonProcessJourneySupport.daemonExecutableName,
            runtimeInstance: "development",
            stateDirectoryPath: stateDirectoryPath);

        let jsonResponseBody: String = "{\"model\":\"accepted-chat-fixture\",\"input\":\"hello\",\"stream\":false}";
        let jsonResponse: String? = DaemonProcessJourneySupport.postJsonEndpoint(
            port: runningDaemon.restPort,
            endpointPath: "/v1/responses",
            bodyJson: jsonResponseBody);
        #expect(
            jsonResponse?.hasPrefix("HTTP/1.1 200 OK") == true,
            "the JSON response must answer 200, got: \(jsonResponse ?? "no response")");
        #expect(
            jsonResponse?.contains("\"object\":\"response\"") == true,
            "the JSON response must be a response object");
        #expect(
            jsonResponse?.contains("\"output_text\":\"accepted chat text\"") == true,
            "the scripted chat text must surface as the output text");

        let streamingResponseBody: String = "{\"model\":\"accepted-chat-fixture\",\"input\":\"hello\",\"stream\":true}";
        let streamingResponse: String? = DaemonProcessJourneySupport.postJsonEndpoint(
            port: runningDaemon.restPort,
            endpointPath: "/v1/responses",
            bodyJson: streamingResponseBody);
        #expect(
            streamingResponse?.hasPrefix("HTTP/1.1 200 OK") == true,
            "the SSE response must answer 200, got: \(streamingResponse ?? "no response")");
        #expect(
            streamingResponse?.contains("event: response.created") == true,
            "the SSE stream must open with the created event");
        #expect(
            streamingResponse?.contains("event: response.output_text.delta") == true,
            "the SSE stream must carry the text deltas");
        #expect(
            streamingResponse?.contains("event: response.completed") == true,
            "the SSE stream must complete");
        #expect(
            streamingResponse?.contains("[DONE]") == false,
            "the responses SSE stream carries no OpenAI chat terminator");

        let exitStatus: Int32? = DaemonProcessJourneySupport.terminateAndWait(
            daemonProcess: runningDaemon.daemonProcess,
            deadlineSeconds: 5);
        #expect(exitStatus == 0);
    }
}


/// Thread-safe one-shot carrier for a chat response a helper thread reads
/// while the journey polls status from the test thread.
final class ChatResponseBox: @unchecked Sendable {

    private let stateLock: NSLock = NSLock();
    private var responseText: String = "";

    func replace(with responseText: String) -> Void {
        self.stateLock.lock();
        self.responseText = responseText;
        self.stateLock.unlock();
    }

    func waitForText(deadlineSeconds: Double) -> String {
        let deadline: Date = Date().addingTimeInterval(deadlineSeconds);
        while Date() < deadline {
            self.stateLock.lock();
            let currentText: String = self.responseText;
            self.stateLock.unlock();
            if !currentText.isEmpty {
                return currentText;
            }
            Thread.sleep(forTimeInterval: 0.05);
        }
        return "";
    }
}
