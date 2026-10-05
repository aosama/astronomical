import Darwin;
import Foundation;
import XCTest;
@testable import IpcProtocol;

/**
 * Creates and owns one unique temporary directory for one transport test,
 * removed in tearDown. Socket paths live inside it so no test ever touches a
 * developer path or a fixed endpoint.
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
 * frame can be written while the test thread blocks inside ProtocolReader.
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
 * Swift transport blocks on the caller's thread, so every listener test
 * connects the client first — the kernel queues the pending connection — and
 * then serves inline; the only background thread is the one writer that must
 * overlap a blocking frame read in the oversized-frame journey.
 */
final class TransportTests: XCTestCase {
    private static let SOCKET_FILE_NAME: String = "ipc.sock";
    private static let LARGE_FRAME_TEXT_BYTE_COUNT: Int = 200_000;

    private var temporaryDirectoryFixture: TransportTemporaryDirectoryFixture?;

    override func setUp() {
        self.temporaryDirectoryFixture = nil;
        super.setUp();
    }

    override func tearDown() {
        guard let fixture: TransportTemporaryDirectoryFixture = self.temporaryDirectoryFixture else {
            super.tearDown();
            return;
        }
        do {
            try fixture.destroy();
        } catch {
            XCTFail("temporary directory should be removed: \(error)");
        }
        self.temporaryDirectoryFixture = nil;
        super.tearDown();
    }

    func testShouldRoundTripHandshakeRequestAndAcceptedResponseOverUnixSocket() throws -> Void {
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

        XCTAssertEqual(receivedRequest, DaemonRequest.handshake);
        let daemonResponse: DaemonResponse? = try daemonClient.nextResponse();
        guard let unwrappedResponse: DaemonResponse = daemonResponse else {
            return XCTFail("the daemon should answer the handshake before closing");
        }
        XCTAssertEqual(unwrappedResponse, cannedResponse);
        XCTAssertEqual(daemonListener.socketPath, socketPath);
    }

    func testShouldReplaceAStaleSocketFileWhenBinding() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let staleBytes: Data = Data("stale bytes from a dead daemon".utf8);
        try staleBytes.write(to: URL(fileURLWithPath: socketPath));

        let boundListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        XCTAssertEqual(boundListener.socketPath, socketPath, "binding over a stale socket file should replace it");
    }

    func testShouldRebindAfterThePreviousListenerReleasesTheSocketFile() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        do {
            let firstListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);
            XCTAssertEqual(firstListener.socketPath, socketPath, "the first listener should hold the socket path");
        }
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: socketPath),
            "a closed listener leaves the socket file behind as stale residue");

        let reboundListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        XCTAssertEqual(reboundListener.socketPath, socketPath, "binding over a stale socket file should rebind");
    }

    func testShouldCreateDaemonSocketWithOwnerOnlyPermissions() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let daemonListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        let socketAttributes: Dictionary<FileAttributeKey, Any> = try FileManager.default.attributesOfItem(
            atPath: daemonListener.socketPath);
        guard let permissionValue: NSNumber = socketAttributes[.posixPermissions] as? NSNumber else {
            return XCTFail("the bound socket file should report posix permissions");
        }
        XCTAssertEqual(
            permissionValue.uint16Value, 0o600,
            "the daemon socket must be readable and writable by the owning user only");
    }

    func testShouldReportDaemonNotRunningWhenConnectingToMissingSocket() throws -> Void {
        let socketPath: String = try self.makeSocketPath();

        XCTAssertThrowsError(try DaemonIpcClient.connect(socketPath: socketPath)) { (thrownError: Error) in
            guard case DaemonTransportError.daemonNotRunning(let reportedPath) = thrownError else {
                XCTFail("connecting to a missing daemon socket should report daemonNotRunning, got: \(thrownError)");
                return;
            }
            XCTAssertEqual(reportedPath, socketPath);
        };
    }

    func testShouldRefuseToBindWhileALiveSocketIsAnswering() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let liveListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        XCTAssertThrowsError(try DaemonIpcListener.bind(socketPath: socketPath)) { (thrownError: Error) in
            guard case DaemonTransportError.bindFailed(let reportedPath, let bindSource) = thrownError else {
                XCTFail("binding over a live socket should report bindFailed, got: \(thrownError)");
                return;
            }
            XCTAssertEqual(reportedPath, socketPath);
            XCTAssertTrue(
                bindSource.underlyingErrorDescription.contains("address in use"),
                "the refusal should carry the address-in-use source, got: \(bindSource)");
        };

        // The final read keeps the live listener alive through the refusal so
        // its socket cannot be reclassified as stale residue mid-test.
        XCTAssertEqual(liveListener.socketPath, socketPath);
    }

    func testShouldReportMissingParentDirectoryWhenBinding() throws -> Void {
        let fixture: TransportTemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let socketPath: String = URL(fileURLWithPath: fixture.rootDirectoryPath)
            .appendingPathComponent("absent-subdirectory")
            .appendingPathComponent(TransportTests.SOCKET_FILE_NAME).path;

        XCTAssertThrowsError(try DaemonIpcListener.bind(socketPath: socketPath)) { (thrownError: Error) in
            guard case DaemonTransportError.socketParentDirectoryMissing(let reportedPath) = thrownError else {
                XCTFail(
                    "binding without a parent directory should report socketParentDirectoryMissing, got: \(thrownError)");
                return;
            }
            XCTAssertEqual(reportedPath, socketPath);
        };
    }

    func testShouldKeepServingAfterClientDisconnectsWithoutARequest() throws -> Void {
        let socketPath: String = try self.makeSocketPath();
        let daemonListener: DaemonIpcListener = try DaemonIpcListener.bind(socketPath: socketPath);

        do {
            let silentClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
            XCTAssertNotNil(silentClient, "connecting a client that will immediately disconnect should succeed");
        }
        try daemonListener.serveNextRequest(handleRequest: { (_ incomingRequest: DaemonRequest) -> DaemonResponse in
            XCTFail("a vanished client must not reach the request handler");
            return TransportTests.handshakeAcceptedResponse();
        });

        let followingClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
        try followingClient.sendRequest(DaemonRequest.handshake);
        try daemonListener.serveNextRequest(handleRequest: { (incomingRequest: DaemonRequest) -> DaemonResponse in
            XCTAssertEqual(incomingRequest, DaemonRequest.handshake);
            return TransportTests.handshakeAcceptedResponse();
        });
        let followingResponse: DaemonResponse? = try followingClient.nextResponse();
        XCTAssertEqual(
            followingResponse, TransportTests.handshakeAcceptedResponse(),
            "the listener should still answer after the silent disconnect");
    }

    func testShouldStreamMultipleResponsesForOneRequest() throws -> Void {
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

        XCTAssertEqual(receivedRequest, chatGenerateRequest);
        let firstFragment: DaemonResponse? = try daemonClient.nextResponse();
        XCTAssertEqual(firstFragment, DaemonResponse.chatGenerationText(text: "Hello"));
        let secondFragment: DaemonResponse? = try daemonClient.nextResponse();
        XCTAssertEqual(secondFragment, DaemonResponse.chatGenerationText(text: " world"));
        let completionFrame: DaemonResponse? = try daemonClient.nextResponse();
        XCTAssertEqual(
            completionFrame,
            DaemonResponse.chatGenerationCompleted(
                promptTokenCount: 5,
                generatedTokenCount: 2,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: ChatGenerationCompletionReason.endOfSequence));

        let connectionClosed: DaemonResponse? = try daemonClient.nextResponse();
        XCTAssertNil(connectionClosed, "the daemon should close the connection after the terminal frame");
    }

    func testShouldSurviveAClientDisconnectMidStream() throws -> Void {
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
                XCTFail(
                    "a vanished mid-stream client is a per-request failure, not a listener failure: \(streamFailure)");
            }
        }

        let followingClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: socketPath);
        try followingClient.sendRequest(DaemonRequest.handshake);
        try daemonListener.serveNextRequest(handleRequest: { (_ incomingRequest: DaemonRequest) -> DaemonResponse in
            return TransportTests.handshakeAcceptedResponse();
        });
        let followingResponse: DaemonResponse? = try followingClient.nextResponse();
        XCTAssertEqual(
            followingResponse, TransportTests.handshakeAcceptedResponse(),
            "the listener should survive the mid-stream disconnect");
    }

    func testShouldReportCleanEndOfFileAtFrameBoundary() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        let protocolWriter: ProtocolWriter = ProtocolWriter(socket: socketPair.clientEnd);
        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);

        try protocolWriter.sendDaemonResponse(DaemonResponse.chatGenerationText(text: "only frame"));
        try socketPair.clientEnd.shutdownWrite();

        let firstResponse: DaemonResponse? = try protocolReader.nextDaemonResponse();
        XCTAssertEqual(firstResponse, DaemonResponse.chatGenerationText(text: "only frame"));
        let terminalResponse: DaemonResponse? = try protocolReader.nextDaemonResponse();
        XCTAssertNil(terminalResponse, "clean EOF at a frame boundary should read as nil, not an error");
    }

    func testShouldReportBytesRemainingWhenAFrameIsTruncated() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        try socketPair.clientEnd.writeAll(Data([0x00, 0x00, 0x00, 0x0A]));
        try socketPair.clientEnd.writeAll(Data([0x7B, 0x22, 0x6B]));
        socketPair.clientEnd.close();

        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);
        XCTAssertThrowsError(try protocolReader.nextDaemonResponse()) { (thrownError: Error) in
            guard case ProtocolError.readFrame(let frameIoError) = thrownError else {
                XCTFail("a truncated frame should raise readFrame, got: \(thrownError)");
                return;
            }
            XCTAssertEqual(frameIoError.underlyingErrorDescription, "bytes remaining on stream");
        };
    }

    func testShouldRejectAnOversizedFrameLengthPrefix() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        try socketPair.clientEnd.writeAll(Data([0x02, 0x00, 0x00, 0x01]));
        socketPair.clientEnd.close();

        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);
        XCTAssertThrowsError(try protocolReader.nextDaemonResponse()) { (thrownError: Error) in
            guard case ProtocolError.readFrame(let frameIoError) = thrownError else {
                XCTFail("an oversized length prefix should raise readFrame, got: \(thrownError)");
                return;
            }
            XCTAssertEqual(frameIoError.underlyingErrorDescription, "frame size too big");
        };
    }

    func testShouldDecodeAFrameDeliveredOneByteAtATime() throws -> Void {
        let socketPair: ConnectedSocketPairFixture = try ConnectedSocketPairFixture();
        let expectedResponse: DaemonResponse = DaemonResponse.chatGenerationText(text: "chunked delivery");
        let serializedResponse: Data = try MessageCodec.encodeDaemonResponse(expectedResponse);
        let framedBytes: Data = TransportTests.framedBytes(serializedResponse);

        for framedByte in Array<UInt8>(framedBytes) {
            try socketPair.clientEnd.writeAll(Data([framedByte]));
        }

        let protocolReader: ProtocolReader = ProtocolReader(socket: socketPair.serverEnd);
        let decodedResponse: DaemonResponse? = try protocolReader.nextDaemonResponse();
        XCTAssertEqual(decodedResponse, expectedResponse);
    }

    func testShouldDecodeAFrameLargerThanOneReceiveBuffer() throws -> Void {
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

        XCTAssertEqual(decodedResponse, expectedResponse);
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
