import Darwin;
import Foundation;

import Testing;

import JourneyCategories;

@testable import IpcProtocol;

/**
 * Creates and owns one unique temporary directory for one transport journey,
 * removed on suite teardown. Socket paths live inside it so no journey ever
 * touches a developer path or a fixed endpoint.
 */
private final class TransportTemporaryDirectoryFixture {
    private let rootDirectoryPathValue: String;

    init() throws {
        // A 12-hex-char UUID prefix keeps the full socket path safely inside
        // sockaddr_un's 104-byte sun_path budget on every machine.
        let fixtureDirectoryName: String = "ast-ipc-\(String(UUID().uuidString.prefix(12)))";
        let candidateRootPath: String = FileManager.default.temporaryDirectory
            .appendingPathComponent(fixtureDirectoryName).path;
        try FileManager.default.createDirectory(atPath: candidateRootPath, withIntermediateDirectories: true);
        self.rootDirectoryPathValue = candidateRootPath;
    }

    var rootDirectoryPath: String {
        return self.rootDirectoryPathValue;
    }

    func pathAppendingComponent(_ component: String) -> String {
        return URL(fileURLWithPath: self.rootDirectoryPathValue).appendingPathComponent(component).path;
    }

    func destroy() throws -> Void {
        try FileManager.default.removeItem(atPath: self.rootDirectoryPathValue);
    }
}

/**
 * One connected AF_UNIX socketpair wrapped as two owned streams, standing in
 * for a client-server connection so the frame codec can be driven directly.
 */
private final class ConnectedSocketPairFixture {
    let clientEnd: UnixSocketStream;
    let serverEnd: UnixSocketStream;

    init() throws {
        var pairedFileDescriptors: Array<Int32> = [0, 0];
        let socketpairResult: Int32 = socketpair(AF_UNIX, SOCK_STREAM, 0, &pairedFileDescriptors);
        if socketpairResult < 0 {
            let capturedErrno: Int32 = errno;
            throw IpcIoError(
                underlyingErrorDescription: "socketpair(AF_UNIX, SOCK_STREAM) failed with errno \(capturedErrno)");
        }
        self.clientEnd = UnixSocketStream(ownedFileDescriptor: pairedFileDescriptors[0]);
        self.serverEnd = UnixSocketStream(ownedFileDescriptor: pairedFileDescriptors[1]);
    }
}

/**
 * Sends one daemon response frame from a background thread so an oversized
 * frame can be written while the journey thread blocks inside ProtocolReader.
 * A socketpair's kernel buffer is far smaller than the frame, so the writer
 * must overlap the reader instead of running ahead of it.
 */
private final class SocketResponseWriterThread: Thread {
    private let protocolWriter: ProtocolWriter;
    private let daemonResponseToSend: DaemonResponse;
    private let mainFinishedSemaphore: DispatchSemaphore;
    private(set) var writeFailure: Error?;

    init(protocolWriter: ProtocolWriter, daemonResponseToSend: DaemonResponse) {
        self.protocolWriter = protocolWriter;
        self.daemonResponseToSend = daemonResponseToSend;
        self.writeFailure = nil;
        self.mainFinishedSemaphore = DispatchSemaphore(value: 0);
        super.init();
    }

    override func main() {
        defer {
            self.mainFinishedSemaphore.signal();
        }
        do {
            try self.protocolWriter.sendDaemonResponse(self.daemonResponseToSend);
        } catch {
            self.writeFailure = error;
        }
    }

    func waitUntilFinished() -> Void {
        self.mainFinishedSemaphore.wait();
    }
}

/**
 * Transport acceptance journeys for the synchronous unix-socket IPC layer,
 * ported from crates/ipc-protocol/tests/hermetic/daemon_transport.rs. The
 * Swift transport blocks on the caller's thread, so every listener journey
 * connects the client first — the kernel queues the pending connection — and
 * then serves inline; the only background thread is the one writer that must
 * overlap a blocking frame read in the oversized-frame journey.
 */
@Suite(.tags(.hermeticJourney))
final class TransportTests {
    private static let SOCKET_FILE_NAME: String = "ipc.sock";
    private static let LARGE_FRAME_TEXT_BYTE_COUNT: Int = 200_000;

    private var temporaryDirectoryFixture: TransportTemporaryDirectoryFixture?;

    deinit {
        if let fixture: TransportTemporaryDirectoryFixture = self.temporaryDirectoryFixture {
            try? fixture.destroy();
        }
    }

    @Test
    func should_round_trip_a_handshake_request_and_accepted_response_over_a_unix_socket() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let daemonListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
        try daemonClient.sendRequest(DaemonRequest.handshake);

        var receivedRequest: DaemonRequest?;
        let cannedResponse: DaemonResponse = TransportTests.handshakeAcceptedResponse();
        try daemonListener.serveNextRequest(handleRequest: { (incomingRequest: DaemonRequest) -> DaemonResponse in
            receivedRequest = incomingRequest;
            return cannedResponse;
        });

        #expect(receivedRequest == DaemonRequest.handshake);
        let daemonResponse: DaemonResponse? = try daemonClient.nextResponse();
        guard let unwrappedResponse: DaemonResponse = daemonResponse else {
            Issue.record("the daemon should answer the handshake before closing");
            return;
        }
        #expect(unwrappedResponse == cannedResponse);
        #expect(daemonListener.socketPath == socketPath);
    }

    @Test
    func should_replace_a_stale_socket_file_when_binding() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let staleBytes: Data = Data("stale bytes from a dead daemon".utf8);
        try staleBytes.write(to: URL(fileURLWithPath: socketPath));

        let boundListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        #expect(boundListener.socketPath == socketPath, "binding over a stale socket file should replace it");
    }

    @Test
    func should_rebind_after_the_previous_listener_releases_the_socket_file() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        do {
            let firstListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);
            #expect(firstListener.socketPath == socketPath, "the first listener should hold the socket path");
        }
        #expect(
            FileManager.default.fileExists(atPath: socketPath),
            "a closed listener leaves the socket file behind as stale residue");

        let reboundListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        #expect(reboundListener.socketPath == socketPath, "binding over a stale socket file should rebind");
    }

    @Test
    func should_create_the_daemon_socket_with_owner_only_permissions() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let daemonListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        let socketAttributes: Dictionary<FileAttributeKey, Any> = try FileManager.default.attributesOfItem(
            atPath: daemonListener.socketPath);
        guard let permissionValue: NSNumber = socketAttributes[.posixPermissions] as? NSNumber else {
            Issue.record("the bound socket file should report posix permissions");
            return;
        }
        #expect(
            permissionValue.uint16Value == 0o600,
            "the daemon socket must be readable and writable by the owning user only");
    }

    @Test
    func should_report_daemon_not_running_when_connecting_to_a_missing_socket() throws -> Void {
        let socketPath: String = try self.makeSocketPath();

        do {
            _ = try DaemonIpcClient.connect(socketPath: socketPath);
            Issue.record("connecting to a missing daemon socket should report daemonNotRunning");
        } catch let transportError as DaemonTransportError {
            guard case DaemonTransportError.daemonNotRunning(let reportedPath) = transportError else {
                Issue.record(Comment(stringLiteral: "expected daemonNotRunning, got \(transportError)"));
                return;
            }
            #expect(reportedPath == socketPath);
        }
    }

    @Test
    func should_refuse_to_bind_while_a_live_socket_is_answering() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let liveListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        do {
            _ = try DaemonIpcListener.bind(socketPath: socketPath);
            Issue.record("binding over a live socket should report bindFailed");
        } catch let transportError as DaemonTransportError {
            guard case DaemonTransportError.bindFailed(let reportedPath, let bindSource) = transportError else {
                Issue.record(Comment(stringLiteral: "expected bindFailed, got \(transportError)"));
                return;
            }
            #expect(reportedPath == socketPath);
            #expect(
                bindSource.underlyingErrorDescription.contains("address in use"),
                "the refusal should carry the address-in-use source, got: \(bindSource)");
        }

        // The final read keeps the live listener alive through the refusal so
        // its socket cannot be reclassified as stale residue mid-journey.
        #expect(liveListener.socketPath == socketPath);
    }

    @Test
    func should_report_a_missing_parent_directory_when_binding() throws -> Void {
        let fixture: TransportTemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let socketPath: String = URL(fileURLWithPath: fixture.rootDirectoryPath)
            .appendingPathComponent("absent-subdirectory")
            .appendingPathComponent(TransportTests.SOCKET_FILE_NAME).path;

        do {
            _ = try DaemonIpcListener.bind(socketPath: socketPath);
            Issue.record("binding without a parent directory should report socketParentDirectoryMissing");
        } catch let transportError as DaemonTransportError {
            guard case DaemonTransportError.socketParentDirectoryMissing(let reportedPath) = transportError else {
                Issue.record(Comment(stringLiteral: "expected socketParentDirectoryMissing, got \(transportError)"));
                return;
            }
            #expect(reportedPath == socketPath);
        }
    }

    @Test
    func should_keep_serving_after_a_client_disconnects_without_a_request() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let daemonListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        do {
            let silentClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
            _ = silentClient;
        }
        try daemonListener.serveNextRequest(handleRequest: { (_ incomingRequest: DaemonRequest) -> DaemonResponse in
            Issue.record("a vanished client must not reach the request handler");
            return TransportTests.handshakeAcceptedResponse();
        });

        let followingClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
        try followingClient.sendRequest(DaemonRequest.handshake);
        try daemonListener.serveNextRequest(handleRequest: { (incomingRequest: DaemonRequest) -> DaemonResponse in
            #expect(incomingRequest == DaemonRequest.handshake);
            return TransportTests.handshakeAcceptedResponse();
        });
        let followingResponse: DaemonResponse? = try followingClient.nextResponse();
        #expect(
            followingResponse == TransportTests.handshakeAcceptedResponse(),
            "the listener should still answer after the silent disconnect");
    }

    @Test
    func should_stream_multiple_responses_for_one_request() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let daemonListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        let chatGenerateRequest: DaemonRequest = TransportTests.chatGenerateRequest();
        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
        try daemonClient.sendRequest(chatGenerateRequest);

        var receivedRequest: DaemonRequest?;
        try daemonListener.serveStreamingRequest(
            handleRequest: { (incomingRequest: DaemonRequest, responseWriter: StreamingResponseWriter) throws -> Void in
                receivedRequest = incomingRequest;
                try responseWriter.sendResponse(DaemonResponse.chatGenerationText(text: "Hello"));
                try responseWriter.sendResponse(DaemonResponse.chatGenerationText(text: " world"));
                try responseWriter.sendResponse(DaemonResponse.chatGenerationCompleted(
                    promptTokenCount: 5,
                    generatedTokenCount: 2,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: ChatGenerationCompletionReason.endOfSequence));
                try responseWriter.close();
            });

        #expect(receivedRequest == chatGenerateRequest);
        let firstFragment: DaemonResponse? = try daemonClient.nextResponse();
        #expect(firstFragment == DaemonResponse.chatGenerationText(text: "Hello"));
        let secondFragment: DaemonResponse? = try daemonClient.nextResponse();
        #expect(secondFragment == DaemonResponse.chatGenerationText(text: " world"));
        let completionFrame: DaemonResponse? = try daemonClient.nextResponse();
        #expect(
            completionFrame
                == DaemonResponse.chatGenerationCompleted(
                    promptTokenCount: 5,
                    generatedTokenCount: 2,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: ChatGenerationCompletionReason.endOfSequence));

        let connectionClosed: DaemonResponse? = try daemonClient.nextResponse();
        #expect(connectionClosed == nil, "the daemon should close the connection after the terminal frame");
    }

    @Test
    func should_survive_a_client_disconnect_mid_stream() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let daemonListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        do {
            let abandoningClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
            try abandoningClient.sendRequest(TransportTests.chatGenerateRequest());
        }
        // The client vanishes mid-stream; the kernel buffers a few frames before
        // the sends start failing, which is exactly what the daemon must survive.
        let oversizedFragmentText: String = String(repeating: "x", count: 64 * 1024);
        var streamFailure: Error?;
        do {
            try daemonListener.serveStreamingRequest(
                handleRequest: { (_ incomingRequest: DaemonRequest, responseWriter: StreamingResponseWriter) throws -> Void in
                    for _ in 0..<64 {
                        try responseWriter.sendResponse(DaemonResponse.chatGenerationText(text: oversizedFragmentText));
                    }
                    try responseWriter.close();
                });
        } catch {
            streamFailure = error;
        }
        if let streamFailure: Error = streamFailure {
            if case DaemonTransportError.acceptFailed = streamFailure {
                Issue.record(Comment(stringLiteral:
                    "a vanished mid-stream client is a per-request failure, not a listener failure: \(streamFailure)"));
            }
        }

        let followingClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
        try followingClient.sendRequest(DaemonRequest.handshake);
        try daemonListener.serveNextRequest(handleRequest: { (_ incomingRequest: DaemonRequest) -> DaemonResponse in
            return TransportTests.handshakeAcceptedResponse();
        });
        let followingResponse: DaemonResponse? = try followingClient.nextResponse();
        #expect(
            followingResponse == TransportTests.handshakeAcceptedResponse(),
            "the listener should survive the mid-stream disconnect");
    }

    @Test
    func should_report_a_clean_end_of_file_at_a_frame_boundary() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        let protocolWriter: ProtocolWriter = ProtocolWriter(socket: socketPair.clientEnd);
        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);

        try protocolWriter.sendDaemonResponse(DaemonResponse.chatGenerationText(text: "only frame"));
        try socketPair.clientEnd.shutdownWrite();

        let firstResponse: DaemonResponse? = try protocolReader.nextDaemonResponse();
        #expect(firstResponse == DaemonResponse.chatGenerationText(text: "only frame"));
        let terminalResponse: DaemonResponse? = try protocolReader.nextDaemonResponse();
        #expect(terminalResponse == nil, "clean EOF at a frame boundary should read as nil, not an error");
    }

    @Test
    func should_report_bytes_remaining_when_a_frame_is_truncated() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        try socketPair.clientEnd.writeAll(Data([0x00, 0x00, 0x00, 0x0A]));
        try socketPair.clientEnd.writeAll(Data([0x7B, 0x22, 0x6B]));
        socketPair.clientEnd.close();

        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);
        do {
            _ = try protocolReader.nextDaemonResponse();
            Issue.record("a truncated frame should raise readFrame");
        } catch ProtocolError.readFrame(let frameIoError) {
            #expect(frameIoError.underlyingErrorDescription == "bytes remaining on stream");
        }
    }

    @Test
    func should_reject_an_oversized_frame_length_prefix() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        try socketPair.clientEnd.writeAll(Data([0x02, 0x00, 0x00, 0x01]));
        socketPair.clientEnd.close();

        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);
        do {
            _ = try protocolReader.nextDaemonResponse();
            Issue.record("an oversized length prefix should raise readFrame");
        } catch ProtocolError.readFrame(let frameIoError) {
            #expect(frameIoError.underlyingErrorDescription == "frame size too big");
        }
    }

    @Test
    func should_decode_a_frame_delivered_one_byte_at_a_time() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        let expectedResponse: DaemonResponse = DaemonResponse.chatGenerationText(text: "chunked delivery");
        let serializedResponse: Data = try MessageCodec.encodeDaemonResponse(expectedResponse);
        let framedBytes: Data = TransportTests.framedBytes(serializedResponse);

        for framedByte in Array<UInt8>(framedBytes) {
            try socketPair.clientEnd.writeAll(Data([framedByte]));
        }

        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);
        let decodedResponse: DaemonResponse? = try protocolReader.nextDaemonResponse();
        #expect(decodedResponse == expectedResponse);
    }

    @Test
    func should_decode_a_frame_larger_than_one_receive_buffer() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        let oversizedFragmentText: String = String(repeating: "x", count: TransportTests.LARGE_FRAME_TEXT_BYTE_COUNT);
        let expectedResponse: DaemonResponse = DaemonResponse.chatGenerationText(text: oversizedFragmentText);

        let responseWriter: ProtocolWriter = ProtocolWriter(socket: socketPair.clientEnd);
        let writerThread: SocketResponseWriterThread = SocketResponseWriterThread(
            protocolWriter: responseWriter, daemonResponseToSend: expectedResponse);
        writerThread.start();

        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);
        let decodedResponse: DaemonResponse? = try protocolReader.nextDaemonResponse();
        writerThread.waitUntilFinished();
        if let writeFailure: Error = writerThread.writeFailure {
            throw writeFailure;
        }

        #expect(decodedResponse == expectedResponse);
    }

    private func makeTemporaryDirectoryFixture() throws -> TransportTemporaryDirectoryFixture {
        let fixture: TransportTemporaryDirectoryFixture = try TransportTemporaryDirectoryFixture();
        self.temporaryDirectoryFixture = fixture;
        return fixture;
    }

    private func makeSocketPath() throws -> String {
        let fixture: TransportTemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        return fixture.pathAppendingComponent(TransportTests.SOCKET_FILE_NAME);
    }

    private static func handshakeAcceptedResponse() -> DaemonResponse {
        return DaemonResponse.handshakeAccepted(
            protocolVersion: DaemonProtocol.protocolVersion,
            applicationName: DaemonProtocol.applicationName);
    }

    private static func chatGenerateRequest() -> DaemonRequest {
        let requestMessages: Array<ChatMessage> = [
            ChatMessage.user(content: "Say hello", images: Array<ChatImageInput>()),
        ];
        return DaemonRequest.chatGenerate(
            model: "example/local-model",
            messages: requestMessages,
            settings: ChatGenerationSettings(
                maxOutputTokens: 128,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            schemaJson: nil);
    }

    private static func framedBytes(_ serializedMessage: Data) -> Data {
        var framedByteArray: Array<UInt8> = Array<UInt8>();
        let frameLength: UInt32 = UInt32(serializedMessage.count);
        framedByteArray.append(UInt8((frameLength >> 24) & 0xFF));
        framedByteArray.append(UInt8((frameLength >> 16) & 0xFF));
        framedByteArray.append(UInt8((frameLength >> 8) & 0xFF));
        framedByteArray.append(UInt8(frameLength & 0xFF));
        framedByteArray.append(contentsOf: Array<UInt8>(serializedMessage));
        return Data(framedByteArray);
    }
}
