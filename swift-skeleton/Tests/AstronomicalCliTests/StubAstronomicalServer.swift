import Foundation

import AstronomicalCli;
import IpcProtocol;

@testable import AstronomicalCli;

/// Loopback Astronomical stand-in for launch journeys, porting stub_server.rs:
/// serves `/v1/status` and `/v1/models` on an ephemeral port so tests never
/// touch a real instance.
final class StubAstronomicalServer {

    let port: UInt16;
    private let listenerFileDescriptor: Int32;
    private var isStopped: Bool = false;
    private let stateLock: NSLock = NSLock();

    init?(statusBody: String, modelsBody: String, modelsStatusLine: String = "200 OK") {
        let listenerFileDescriptor: Int32 = socket(AF_INET, Int32(SOCK_STREAM), 0);
        guard listenerFileDescriptor >= 0 else {
            return nil;
        }
        var anyAddress: sockaddr_in = sockaddr_in();
        anyAddress.sin_family = sa_family_t(AF_INET);
        anyAddress.sin_port = 0;
        anyAddress.sin_addr = in_addr(s_addr: INADDR_ANY);
        let bindOutcome: Int32 = withUnsafePointer(to: &anyAddress) { addressPointer in
            return addressPointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketPointer in
                return Darwin.bind(listenerFileDescriptor, socketPointer, socklen_t(MemoryLayout<sockaddr_in>.size));
            }
        };
        guard bindOutcome == 0, listen(listenerFileDescriptor, 8) == 0 else {
            close(listenerFileDescriptor);
            return nil;
        }
        var boundAddress: sockaddr_in = sockaddr_in();
        var boundAddressLength: socklen_t = socklen_t(MemoryLayout<sockaddr_in>.size);
        let getsocknameOutcome: Int32 = withUnsafeMutablePointer(to: &boundAddress) { boundAddressPointer in
            return boundAddressPointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketPointer in
                return Darwin.getsockname(listenerFileDescriptor, socketPointer, &boundAddressLength);
            }
        };
        guard getsocknameOutcome == 0 else {
            close(listenerFileDescriptor);
            return nil;
        }
        self.port = UInt16(bigEndian: boundAddress.sin_port);
        self.listenerFileDescriptor = listenerFileDescriptor;
        let capturedStatusBody: String = statusBody;
        let capturedModelsBody: String = modelsBody;
        let capturedModelsStatusLine: String = modelsStatusLine;
        let serveThread: Thread = Thread { [weak self] in
            self?.serveLoop(
                statusBody: capturedStatusBody,
                modelsBody: capturedModelsBody,
                modelsStatusLine: capturedModelsStatusLine
            );
        };
        serveThread.name = "stub-astronomical-http";
        serveThread.start();
    }

    deinit {
        self.stop();
    }

    func stop() {
        self.stateLock.lock();
        let wasStopped: Bool = self.isStopped;
        self.isStopped = true;
        self.stateLock.unlock();
        if !wasStopped {
            close(self.listenerFileDescriptor);
        }
    }

    private func serveLoop(
        statusBody: String,
        modelsBody: String,
        modelsStatusLine: String
    ) {
        while (true) {
            self.stateLock.lock();
            let isStopped: Bool = self.isStopped;
            self.stateLock.unlock();
            if (isStopped) {
                return;
            }
            var peerAddress: sockaddr = sockaddr();
            var peerAddressLength: socklen_t = socklen_t(MemoryLayout<sockaddr>.size);
            let acceptedFileDescriptor: Int32 = accept(self.listenerFileDescriptor, &peerAddress, &peerAddressLength);
            if (acceptedFileDescriptor < 0) {
                return;
            }
            StubAstronomicalServer.respond(
                acceptedFileDescriptor,
                statusBody: statusBody,
                modelsBody: modelsBody,
                modelsStatusLine: modelsStatusLine
            );
            close(acceptedFileDescriptor);
        }
    }

    private static func respond(
        _ connectionFileDescriptor: Int32,
        statusBody: String,
        modelsBody: String,
        modelsStatusLine: String
    ) {
        var readBuffer: [UInt8] = [UInt8](repeating: 0, count: 4096);
        _ = read(connectionFileDescriptor, &readBuffer, readBuffer.count);
        let requestText: String = String(decoding: readBuffer, as: UTF8.self);
        let requestFields: Array<Substring> = requestText.split(separator: " ");
        let requestPath: String = requestFields.count > 1 ? String(requestFields[1]) : "/";
        let responseBody: String;
        var statusLine: String = "200 OK";
        if requestPath.hasPrefix("/v1/status") {
            responseBody = statusBody;
        } else if requestPath.hasPrefix("/v1/models") {
            responseBody = modelsBody;
            statusLine = modelsStatusLine;
        } else {
            responseBody = "{\"error\":\"unknown\"}";
        }
        let responseText: String = "HTTP/1.1 \(statusLine)\r\nContent-Type: application/json\r\n"
            + "Content-Length: \(responseBody.utf8.count)\r\nConnection: close\r\n\r\n\(responseBody)";
        let responseBytes: [UInt8] = Array(responseText.utf8);
        _ = responseBytes.withUnsafeBufferPointer { (bufferPointer: UnsafeBufferPointer<UInt8>) -> Void in
            guard let baseAddress: UnsafePointer<UInt8> = bufferPointer.baseAddress else {
                return;
            }
            _ = write(connectionFileDescriptor, baseAddress, responseBytes.count);
        };
    }
}

/// PATH fixture factory for launch journeys: one directory holding a
/// runnable `opencode` stub.
enum LaunchPathFixture {

    static func makeOpencodePath() throws -> (directory: String, pathValue: String) {
        let toolDirectory: String = CliJourneySupport.freshTestDirectory("launch-bin");
        let opencodePath: String = toolDirectory + "/opencode";
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: URL(fileURLWithPath: opencodePath));
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: opencodePath
        );
        return (toolDirectory, "/usr/bin:\(toolDirectory)");
    }
}
