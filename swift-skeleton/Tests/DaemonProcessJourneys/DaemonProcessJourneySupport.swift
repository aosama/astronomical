import Foundation;

/// Spawns the built astronomicald daemon binary the way the Rust
/// daemon-process journeys spawned theirs: every daemon gets its own
/// temporary state directory (so the ephemeral loopback bind never touches
/// the installed Stable or Development instances), reports its REST address
/// on stdout, and is always terminated by the journey that started it.
///
/// This target is deliberately its own SwiftPM test target — a separate
/// process under `swift test`, mirroring the Rust tree's separate
/// integration-test binary — so spawning real daemons can never starve the
/// parallel hermetic suites' dispatch servicing inside one shared runner
/// process.
enum DaemonProcessJourneySupport {

    static let startupLineRestPrefix: String = "serving REST on http://";
    static let daemonExecutableName: String = "AstronomicalDaemon";

    /// The daemon binary built by this package; SwiftPM builds every
    /// executable target before tests run. SwiftPM gives tests no
    /// CARGO_BIN_EXE-style variable, so the executable is located from this
    /// source file's package root, checking both build layouts.
    static func locateDaemonExecutable() throws -> String {
        let packageRootUrl: URL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent();
        let buildLayouts: Array<String> = [
            ".build/out/Products/Debug",
            ".build/debug"
        ];
        for buildLayout: String in buildLayouts {
            let daemonExecutableUrl: URL = packageRootUrl
                .appendingPathComponent(buildLayout)
                .appendingPathComponent(DaemonProcessJourneySupport.daemonExecutableName);
            if FileManager.default.isExecutableFile(atPath: daemonExecutableUrl.path) {
                return daemonExecutableUrl.path;
            }
        }
        throw DaemonProcessJourneySupportFailure.missingDaemonExecutable(
            path: packageRootUrl.appendingPathComponent(".build/...").path);
    }

    static func makeStateDirectory(journeyName: String) throws -> String {
        // Unix-domain socket paths cap around 104 bytes and the IPC socket
        // lives at state-dir/ipc.sock, so the directory stays SHORT — a short
        // journey tag plus a slice of UUID, never the full journey name.
        let shortJourneyTag: String = String(journeyName.prefix(8));
        let stateDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("aa-\(shortJourneyTag)-\(UUID().uuidString.prefix(8))");
        try FileManager.default.createDirectory(at: stateDirectoryUrl, withIntermediateDirectories: true);
        return stateDirectoryUrl.path;
    }

    static func removeStateDirectory(_ stateDirectoryPath: String) -> Void {
        try? FileManager.default.removeItem(atPath: stateDirectoryPath);
    }

    static func writeInstanceConfig(stateDirectoryPath: String) throws -> Void {
        try DaemonProcessJourneySupport.writeRawConfig(
            stateDirectoryPath: stateDirectoryPath,
            configJson: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[]},"
                + "\"chunking\":{\"fixed_prompt_processing_chunk_size_tokens\":2048}}");
    }

    static func writeRawConfig(stateDirectoryPath: String, configJson: String) throws -> Void {
        try configJson.write(
            toFile: stateDirectoryPath + "/config.json",
            atomically: true,
            encoding: String.Encoding.utf8);
    }

    /// Starts one daemon and returns it with the REST address it published.
    /// The first stdout line is the live-progress startup line; the wait for
    /// it is bounded so a wedged daemon fails the journey, not the machine.
    static func spawnDaemon(
        daemonExecutablePath: String,
        runtimeInstance: String,
        stateDirectoryPath: String
    ) throws -> (daemonProcess: Process, restPort: UInt16) {
        let daemonProcess: Process = Process();
        daemonProcess.executableURL = URL(fileURLWithPath: daemonExecutablePath);
        daemonProcess.arguments = ["--instance", runtimeInstance, "--state-directory", stateDirectoryPath];
        let stdoutPipe: Pipe = Pipe();
        let stderrPipe: Pipe = Pipe();
        daemonProcess.standardOutput = stdoutPipe;
        daemonProcess.standardError = stderrPipe;
        daemonProcess.standardInput = FileHandle.nullDevice;
        try daemonProcess.run();
        let startupLine: String = try DaemonProcessJourneySupport.readLineWithDeadline(
            pipe: stdoutPipe,
            deadlineSeconds: 15,
            onTimeout: {
                let daemonState: String;
                if daemonProcess.isRunning {
                    daemonState = "still running";
                } else {
                    daemonState = "exited status \(daemonProcess.terminationStatus) reason \(daemonProcess.terminationReason.rawValue)";
                }
                return "daemon \(daemonState); stderr: "
                    + DaemonProcessJourneySupport.drainPipeWithGrace(pipe: stderrPipe, graceSeconds: 1);
            });
        guard let restPort: UInt16 = DaemonProcessJourneySupport.restPort(fromStartupLine: startupLine) else {
            _ = DaemonProcessJourneySupport.terminateAndWait(daemonProcess: daemonProcess, deadlineSeconds: 3);
            throw DaemonProcessJourneySupportFailure.startupLineMissingRestAddress(startupLine: startupLine);
        }
        return (daemonProcess, restPort);
    }

    /// Runs the daemon expecting a startup refusal and returns its exit
    /// status, stdout, and stderr; the wait is bounded so an unexpectedly
    /// serving daemon fails the journey instead of hanging it.
    static func runDaemonExpectingStartupFailure(
        daemonExecutablePath: String,
        arguments: Array<String>,
        deadlineSeconds: Double
    ) throws -> (exitStatus: Int32, standardOutput: String, standardError: String) {
        let daemonProcess: Process = Process();
        daemonProcess.executableURL = URL(fileURLWithPath: daemonExecutablePath);
        daemonProcess.arguments = arguments;
        let stdoutPipe: Pipe = Pipe();
        let stderrPipe: Pipe = Pipe();
        daemonProcess.standardOutput = stdoutPipe;
        daemonProcess.standardError = stderrPipe;
        daemonProcess.standardInput = FileHandle.nullDevice;
        try daemonProcess.run();
        let exitStatus: Int32? = DaemonProcessJourneySupport.waitWithDeadline(
            daemonProcess: daemonProcess,
            deadlineSeconds: deadlineSeconds);
        guard let finishedExitStatus: Int32 = exitStatus else {
            daemonProcess.terminate();
            throw DaemonProcessJourneySupportFailure.daemonDidNotExitBeforeDeadline;
        }
        let stderrData: Data = stderrPipe.fileHandleForReading.readDataToEndOfFile();
        let stdoutData: Data = stdoutPipe.fileHandleForReading.readDataToEndOfFile();
        return (
            finishedExitStatus,
            String(decoding: stdoutData, as: UTF8.self),
            String(decoding: stderrData, as: UTF8.self));
    }

    /// Sends SIGTERM — the signal the real LaunchAgents own — and returns
    /// the exit status, or nil when the daemon outlives the deadline.
    static func terminateAndWait(daemonProcess: Process, deadlineSeconds: Double) -> Int32? {
        if daemonProcess.isRunning {
            daemonProcess.terminate();
        }
        return DaemonProcessJourneySupport.waitWithDeadline(
            daemonProcess: daemonProcess,
            deadlineSeconds: deadlineSeconds);
    }

    /// Waits for a daemon the journey already asked to shut down over its
    /// own control endpoint; no signal is sent, the graceful path owns it.
    static func waitTerminationWithoutSignal(daemonProcess: Process, deadlineSeconds: Double) -> Int32? {
        return DaemonProcessJourneySupport.waitWithDeadline(
            daemonProcess: daemonProcess,
            deadlineSeconds: deadlineSeconds);
    }

    static func getEndpoint(port: UInt16, endpointPath: String) -> String? {
        return DaemonProcessJourneySupport.exchange(
            port: port,
            requestText: "GET \(endpointPath) HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    }

    static func postEmptyEndpoint(port: UInt16, endpointPath: String) -> String? {
        return DaemonProcessJourneySupport.exchange(
            port: port,
            requestText: "POST \(endpointPath) HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    }

    /// Speaks exactly the bytes a command-line client would send and reads
    /// until the daemon closes, bounded by a receive timeout so a wedged
    /// daemon fails the journey instead of hanging it.
    private static func exchange(port: UInt16, requestText: String) -> String? {
        let connectionDescriptor: Int32 = DaemonProcessJourneySupport.openConnectedSocket(port: port);
        guard connectionDescriptor >= 0 else {
            return nil;
        }
        defer { close(connectionDescriptor); }
        let requestData: Data = Data(requestText.utf8);
        requestData.withUnsafeBytes({ (rawBuffer: UnsafeRawBufferPointer) -> Void in
            if let basePointer: UnsafeRawPointer = rawBuffer.baseAddress {
                _ = send(connectionDescriptor, basePointer, rawBuffer.count, 0);
            }
        });
        var receivedData: Data = Data();
        var readBuffer: Array<UInt8> = Array(repeating: 0, count: 4096);
        while true {
            let bytesRead: Int = read(connectionDescriptor, &readBuffer, readBuffer.count);
            if bytesRead <= 0 {
                break;
            }
            receivedData.append(contentsOf: readBuffer[0..<bytesRead]);
        }
        return String(data: receivedData, encoding: .utf8);
    }

    private static func openConnectedSocket(port: UInt16) -> Int32 {
        let connectionDescriptor: Int32 = socket(AF_INET, SOCK_STREAM, 0);
        guard connectionDescriptor >= 0 else {
            return -1;
        }
        var noSigPipeFlag: Int32 = 1;
        _ = setsockopt(
            connectionDescriptor, SOL_SOCKET, SO_NOSIGPIPE,
            &noSigPipeFlag, socklen_t(MemoryLayout<Int32>.size));
        var receiveTimeout: timeval = timeval(tv_sec: 30, tv_usec: 0);
        _ = setsockopt(
            connectionDescriptor, SOL_SOCKET, SO_RCVTIMEO,
            &receiveTimeout, socklen_t(MemoryLayout<timeval>.size));
        var address: sockaddr_in = sockaddr_in();
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size);
        address.sin_family = sa_family_t(AF_INET);
        address.sin_port = in_port_t(port).bigEndian;
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"));
        let connectOutcome: Int32 = withUnsafePointer(to: &address, { (addressPointer: UnsafePointer<sockaddr_in>) -> Int32 in
            return connect(
                connectionDescriptor,
                UnsafePointer<sockaddr>(OpaquePointer(addressPointer)),
                socklen_t(MemoryLayout<sockaddr_in>.size));
        });
        if connectOutcome != 0 {
            close(connectionDescriptor);
            return -1;
        }
        return connectionDescriptor;
    }

    /// Polls the endpoint until the response contains the expected fragment,
    /// bounded by the deadline; returns the last response either way.
    static func waitUntilEndpointContains(
        port: UInt16,
        endpointPath: String,
        expectedFragment: String,
        deadlineSeconds: Double
    ) -> String? {
        let deadline: Date = Date().addingTimeInterval(deadlineSeconds);
        var latestResponse: String? = nil;
        while Date() < deadline {
            latestResponse = DaemonProcessJourneySupport.getEndpoint(port: port, endpointPath: endpointPath);
            if let latestResponse: String = latestResponse, latestResponse.contains(expectedFragment) {
                return latestResponse;
            }
            Thread.sleep(forTimeInterval: 0.05);
        }
        return latestResponse;
    }

    private static func waitWithDeadline(daemonProcess: Process, deadlineSeconds: Double) -> Int32? {
        let deadline: Date = Date().addingTimeInterval(deadlineSeconds);
        while daemonProcess.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05);
        }
        if daemonProcess.isRunning {
            return nil;
        }
        return daemonProcess.terminationStatus;
    }

    private static func readLineWithDeadline(
        pipe: Pipe,
        deadlineSeconds: Double,
        onTimeout: (() -> String)? = nil
    ) throws -> String {
        let lineBox: StartupLineBox = StartupLineBox();
        pipe.fileHandleForReading.readabilityHandler = { (fileHandle: FileHandle) in
            let availableBytes: Data = fileHandle.availableData;
            if availableBytes.isEmpty {
                fileHandle.readabilityHandler = nil;
                lineBox.markStreamEnded();
                return;
            }
            lineBox.append(availableBytes);
            if lineBox.containsLineBreak() {
                fileHandle.readabilityHandler = nil;
            }
        };
        let startupLine: String? = lineBox.waitForLine(deadlineSeconds: deadlineSeconds);
        pipe.fileHandleForReading.readabilityHandler = nil;
        guard let startupLine: String = startupLine else {
            let timeoutDiagnostic: String = onTimeout?() ?? "";
            throw DaemonProcessJourneySupportFailure.startupLineMissing(
                daemonDiagnostic: timeoutDiagnostic);
        }
        return startupLine;
    }

    /// Best-effort stderr capture for a daemon that never served: the read
    /// runs on its own thread because a live daemon never closes the pipe,
    /// and whatever arrived within the grace window explains the stall.
    private static func drainPipeWithGrace(pipe: Pipe, graceSeconds: Double) -> String {
        let drainedBox: DrainableTextBox = DrainableTextBox();
        let drainThread: Thread = Thread {
            let drainedBytes: Data = pipe.fileHandleForReading.readDataToEndOfFile();
            drainedBox.replace(with: drainedBytes);
        };
        drainThread.name = "astronomical-journey-stderr-drain";
        drainThread.start();
        return drainedBox.waitForText(deadlineSeconds: graceSeconds);
    }

    private static func restPort(fromStartupLine startupLine: String) -> UInt16? {
        guard let addressRange: Range<String.Index> = startupLine.range(
            of: DaemonProcessJourneySupport.startupLineRestPrefix) else {
            return nil;
        }
        let addressSuffix: String = String(startupLine[addressRange.upperBound...])
            .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines);
        let addressParts: Array<Substring> = addressSuffix.split(separator: ":");
        guard addressParts.count == 2, let parsedPort: UInt16 = UInt16(addressParts[1]) else {
            return nil;
        }
        return parsedPort;
    }
}

/// Thread-safe accumulator for the daemon's first stdout line.
private final class StartupLineBox: @unchecked Sendable {
    private let stateLock: NSLock = NSLock();
    private var accumulatedBytes: Data = Data();
    private var streamEnded: Bool = false;

    func append(_ additionalBytes: Data) -> Void {
        self.stateLock.lock();
        self.accumulatedBytes.append(additionalBytes);
        self.stateLock.unlock();
    }

    func markStreamEnded() -> Void {
        self.stateLock.lock();
        self.streamEnded = true;
        self.stateLock.unlock();
    }

    func containsLineBreak() -> Bool {
        self.stateLock.lock();
        let containsLineBreak: Bool = self.accumulatedBytes.contains(UInt8(ascii: "\n"));
        self.stateLock.unlock();
        return containsLineBreak;
    }

    func waitForLine(deadlineSeconds: Double) -> String? {
        let deadline: Date = Date().addingTimeInterval(deadlineSeconds);
        while Date() < deadline {
            self.stateLock.lock();
            let currentBytes: Data = self.accumulatedBytes;
            let currentStreamEnded: Bool = self.streamEnded;
            self.stateLock.unlock();
            let currentText: String = String(decoding: currentBytes, as: UTF8.self);
            if let lineBreakIndex: String.Index = currentText.firstIndex(of: "\n") {
                return String(currentText[..<lineBreakIndex]);
            }
            if currentStreamEnded && !currentText.isEmpty {
                return currentText;
            }
            Thread.sleep(forTimeInterval: 0.05);
        }
        return nil;
    }
}

enum DaemonProcessJourneySupportFailure: Error, CustomStringConvertible {
    case missingDaemonExecutable(path: String);
    case startupLineMissing(daemonDiagnostic: String);
    case startupLineMissingRestAddress(startupLine: String);
    case daemonDidNotExitBeforeDeadline;

    var description: String {
        switch (self) {
        case let .missingDaemonExecutable(path):
            return "the built astronomicald binary was not found at \(path)";
        case let .startupLineMissing(daemonDiagnostic):
            return "the daemon never printed its startup line; daemon stderr: \(daemonDiagnostic)";
        case let .startupLineMissingRestAddress(startupLine):
            return "the daemon startup line names no REST address: \(startupLine)";
        case .daemonDidNotExitBeforeDeadline:
            return "the daemon kept running past the journey deadline";
        }
    }
}

/// Thread-safe one-shot text box for stderr drained on a helper thread.
private final class DrainableTextBox: @unchecked Sendable {

    private let stateLock: NSLock = NSLock();
    private var drainedText: String = "";

    func replace(with drainedBytes: Data) -> Void {
        self.stateLock.lock();
        self.drainedText = String(decoding: drainedBytes, as: UTF8.self);
        self.stateLock.unlock();
    }

    func waitForText(deadlineSeconds: Double) -> String {
        let deadline: Date = Date().addingTimeInterval(deadlineSeconds);
        while Date() < deadline {
            self.stateLock.lock();
            let currentText: String = self.drainedText;
            self.stateLock.unlock();
            if !currentText.isEmpty {
                return currentText;
            }
            Thread.sleep(forTimeInterval: 0.05);
        }
        return "";
    }
}
