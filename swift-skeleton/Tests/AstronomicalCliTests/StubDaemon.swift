import Foundation

import Foundation;

@testable import IpcProtocol;

/**
 * Configurable stub daemon speaking the real framed protocol over a real
 * unix socket, porting stub_daemon.rs. One connection serves a scripted
 * exchange sequence, like the real daemon.
 */
final class StubDaemon: @unchecked Sendable {

    struct StubInstalledModel {
        let modelId: String;
        let contextWindow: UInt32?;
        let supportsEmbeddings: Bool;
        let isResident: Bool;

        static func chat(_ modelId: String, isResident: Bool) -> StubInstalledModel {
            return StubInstalledModel(
                modelId: modelId,
                contextWindow: 32768,
                supportsEmbeddings: false,
                isResident: isResident
            );
        }

        static func embeddings(_ modelId: String, isResident: Bool) -> StubInstalledModel {
            return StubInstalledModel(
                modelId: modelId,
                contextWindow: nil,
                supportsEmbeddings: true,
                isResident: isResident
            );
        }
    }

    struct StubCatalogEntry {
        let huggingfaceId: String;
        let requestableModelId: String?;
        var readyOnThisMac: Bool;
        var downloadState: String?;
        let contextWindow: UInt32?;
        let supportsEmbeddings: Bool;

        static func chat(
            _ huggingfaceId: String,
            _ requestableModelId: String,
            _ readyOnThisMac: Bool
        ) -> StubCatalogEntry {
            return StubCatalogEntry(
                huggingfaceId: huggingfaceId,
                requestableModelId: requestableModelId,
                readyOnThisMac: readyOnThisMac,
                downloadState: nil,
                contextWindow: 32768,
                supportsEmbeddings: false
            );
        }

        static func embeddings(
            _ huggingfaceId: String,
            _ requestableModelId: String,
            _ readyOnThisMac: Bool
        ) -> StubCatalogEntry {
            return StubCatalogEntry(
                huggingfaceId: huggingfaceId,
                requestableModelId: requestableModelId,
                readyOnThisMac: readyOnThisMac,
                downloadState: nil,
                contextWindow: nil,
                supportsEmbeddings: true
            );
        }
    }

    enum StubEmbeddingsOutcome {
        case completed
        case contextLengthExceeded
    }

    struct StubDaemonConfig {
        var workerStatus: DaemonWorkerStatus = .ready;
        var installedModels: Array<StubInstalledModel> = [];
        var catalogEntries: Array<StubCatalogEntry> = [];
        var catalogEntriesAfterDownload: Array<StubCatalogEntry>? = nil;
        var defaultModelId: String? = nil;
        var chatFragments: Array<String> = [];
        var embeddingsOutcome: StubEmbeddingsOutcome = .completed;
        var embeddingVector: Array<Float> = [0.25, -0.5];
        var downloadJobs: Array<DaemonDownloadJob?> = [];
    }

    private let stateLock: NSLock = NSLock();
    private let config: StubDaemonConfig;
    private var currentDefaultModelId: String?;
    private var downloadJobIndex: Int = 0;
    private var isDownloadStarted: Bool = false;
    // Captures from the most recent ChatGenerate, shared across connections.
    private var capturedImages: Array<ChatImageInput>? = nil;
    private var capturedMessages: Array<ChatMessage>? = nil;
    private var capturedSettings: ChatGenerationSettings? = nil;
    private var capturedSchemaJson: String? = nil;

    private var listenerFileDescriptor: Int32 = -1;
    private var serveThread: Thread? = nil;
    let socketPath: String;

    init(socketPath: String, config: StubDaemonConfig) {
        self.socketPath = socketPath;
        self.config = config;
        self.currentDefaultModelId = config.defaultModelId;
    }

    /// Binds the socket and starts serving scripted exchanges.
    func start() throws {
        self.listenerFileDescriptor = socket(AF_UNIX, SOCK_STREAM, 0);
        guard self.listenerFileDescriptor >= 0 else {
            throw StubDaemonError.bindFailed;
        }
        var socketAddress: sockaddr_un = sockaddr_un();
        socketAddress.sun_family = sa_family_t(AF_UNIX);
        let socketPathBytes: [UInt8] = Array(self.socketPath.utf8);
        guard socketPathBytes.count < MemoryLayout.size(ofValue: socketAddress.sun_path) else {
            close(self.listenerFileDescriptor);
            throw StubDaemonError.bindFailed;
        }
        let copyOutcome: Int32 = socketPathBytes.withUnsafeBufferPointer { (pathBuffer: UnsafeBufferPointer<UInt8>) -> Int32 in
            return withUnsafeMutableBytes(of: &socketAddress.sun_path) { (destinationBuffer: UnsafeMutableRawBufferPointer) -> Int32 in
                guard let destinationBase: UnsafeMutableRawPointer = destinationBuffer.baseAddress else {
                    return -1;
                }
                memcpy(destinationBase, pathBuffer.baseAddress!, socketPathBytes.count);
                destinationBuffer[socketPathBytes.count] = 0;
                return 0;
            };
        };
        guard copyOutcome == 0,
              bind(self.listenerFileDescriptor, withUnsafePointer(to: &socketAddress) { (addressPointer: UnsafePointer<sockaddr_un>) -> UnsafePointer<sockaddr> in
                  return UnsafeRawPointer(addressPointer).assumingMemoryBound(to: sockaddr.self);
              }, socklen_t(MemoryLayout<sockaddr_un>.size)) == 0,
              listen(self.listenerFileDescriptor, 8) == 0 else {
            close(self.listenerFileDescriptor);
            throw StubDaemonError.bindFailed;
        }
        let serveThread: Thread = Thread { [weak self] in
            self?.serveLoop();
        };
        serveThread.name = "stub-daemon";
        serveThread.start();
        self.stateLock.lock();
        self.serveThread = serveThread;
        self.stateLock.unlock();
    }

    /// Stops the listener; in-flight connections finish their exchanges. A
    /// throwaway self-connection wakes the blocked accept BEFORE the
    /// descriptor is closed, so the fd number can never be reused under a
    /// still-blocked accept that would steal a later stub's connection.
    func stop() {
        self.stateLock.lock();
        let listenerFileDescriptor: Int32 = self.listenerFileDescriptor;
        self.listenerFileDescriptor = -1;
        self.stateLock.unlock();
        if (listenerFileDescriptor >= 0) {
            let wakeupFileDescriptor: Int32 = socket(AF_UNIX, SOCK_STREAM, 0);
            if (wakeupFileDescriptor >= 0) {
                var socketAddress: sockaddr_un = sockaddr_un();
                socketAddress.sun_family = sa_family_t(AF_UNIX);
                let socketPathBytes: [UInt8] = Array(self.socketPath.utf8);
                _ = socketPathBytes.withUnsafeBufferPointer { (pathBuffer: UnsafeBufferPointer<UInt8>) -> Int32 in
                    return withUnsafeMutableBytes(of: &socketAddress.sun_path) { (destinationBuffer: UnsafeMutableRawBufferPointer) -> Int32 in
                        guard let destinationBase: UnsafeMutableRawPointer = destinationBuffer.baseAddress,
                              let sourceBase: UnsafePointer<UInt8> = pathBuffer.baseAddress else {
                            return -1;
                        }
                        memcpy(destinationBase, sourceBase, socketPathBytes.count);
                        destinationBuffer[socketPathBytes.count] = 0;
                        return 0;
                    };
                };
                _ = withUnsafePointer(to: &socketAddress) { (addressPointer: UnsafePointer<sockaddr_un>) -> Int32 in
                    return UnsafeRawPointer(addressPointer).assumingMemoryBound(to: sockaddr.self).withMemoryRebound(to: sockaddr.self, capacity: 1) { (socketPointer: UnsafePointer<sockaddr>) -> Int32 in
                        return connect(wakeupFileDescriptor, socketPointer, socklen_t(MemoryLayout<sockaddr_un>.size));
                    };
                };
                close(wakeupFileDescriptor);
            }
            close(listenerFileDescriptor);
        }
        unlink(self.socketPath);
    }

    private func serveLoop() {
        while (true) {
            self.stateLock.lock();
            let listenerFileDescriptor: Int32 = self.listenerFileDescriptor;
            self.stateLock.unlock();
            if (listenerFileDescriptor < 0) {
                return;
            }
            var peerAddress: sockaddr = sockaddr();
            var peerAddressLength: socklen_t = socklen_t(MemoryLayout<sockaddr>.size);
            let acceptedFileDescriptor: Int32 = accept(listenerFileDescriptor, &peerAddress, &peerAddressLength);
            if (acceptedFileDescriptor < 0) {
                return;
            }
            let connectionThread: Thread = Thread { [weak self] in
                self?.serveConnection(acceptedFileDescriptor);
            };
            connectionThread.name = "stub-daemon-connection";
            connectionThread.start();
        }
    }

    private func serveConnection(_ connectionFileDescriptor: Int32) {
        // The transport owns the descriptor exclusively with its close-once
        // guard; raw close() calls here would double-close after the number
        // is reused by a later connection and steal its bytes.
        let transport: UnixSocketStream = UnixSocketStream(ownedFileDescriptor: connectionFileDescriptor);
        let protocolReader: ProtocolReader = ProtocolReader(transport: transport);
        let protocolWriter: ProtocolWriter = ProtocolWriter(transport: transport);
        while (true) {
            let daemonRequest: DaemonRequest?;
            do {
                daemonRequest = try protocolReader.nextDaemonRequest();
            } catch {
                protocolReader.closeTransportFileDescriptor();
                return;
            }
            guard let request: DaemonRequest = daemonRequest else {
                protocolReader.closeTransportFileDescriptor();
                return;
            }
            do {
                try self.respond(request, protocolWriter: protocolWriter);
            } catch {
                protocolWriter.closeTransportFileDescriptor();
                return;
            }
        }
    }

    private func respond(
        _ daemonRequest: DaemonRequest,
        protocolWriter: ProtocolWriter
    ) throws {
        switch (daemonRequest) {
        case .handshake:
            try protocolWriter.sendDaemonResponse(.handshakeAccepted(
                protocolVersion: DaemonProtocol.protocolVersion,
                applicationName: DaemonProtocol.applicationName
            ));
        case .status:
            self.stateLock.lock();
            let currentDefaultModelId: String? = self.currentDefaultModelId;
            let residentModelId: String? = self.residentModelIdWhileLocked();
            let workerStatus: DaemonWorkerStatus = self.config.workerStatus;
            self.stateLock.unlock();
            try protocolWriter.sendDaemonResponse(.status(
                workerStatus: workerStatus,
                readyModelId: residentModelId,
                defaultModelId: currentDefaultModelId
            ));
        case .modelsList:
            let listedModels: Array<DaemonListedModel> = self.config.installedModels.map { (installedModel: StubInstalledModel) -> DaemonListedModel in
                return DaemonListedModel(
                    modelId: installedModel.modelId,
                    family: "stub",
                    contextWindow: installedModel.contextWindow,
                    supportsEmbeddings: installedModel.supportsEmbeddings,
                    isResident: installedModel.isResident,
                    sizeBytes: 1_000_000_000
                );
            };
            try protocolWriter.sendDaemonResponse(.modelsList(models: listedModels));
        case .catalog:
            self.stateLock.lock();
            let afterDownloadActive: Bool = self.isDownloadStarted;
            self.stateLock.unlock();
            let catalogEntries: Array<StubCatalogEntry> = afterDownloadActive
                ? (self.config.catalogEntriesAfterDownload ?? [])
                : self.config.catalogEntries;
            let entries: Array<DaemonCatalogEntry> = catalogEntries.map { (catalogEntry: StubCatalogEntry) -> DaemonCatalogEntry in
                let displayName: String = catalogEntry.requestableModelId ?? catalogEntry.huggingfaceId;
                return DaemonCatalogEntry(
                    huggingfaceId: catalogEntry.huggingfaceId,
                    displayName: displayName,
                    family: "stub",
                    approximateSizeBytes: 2_000_000_000,
                    readyOnThisMac: catalogEntry.readyOnThisMac,
                    requestableModelId: catalogEntry.requestableModelId,
                    downloadState: catalogEntry.downloadState,
                    contextWindow: catalogEntry.contextWindow,
                    supportsReasoning: false,
                    supportsVision: false,
                    supportsToolCalls: false,
                    supportsImageGeneration: false,
                    supportsEmbeddings: catalogEntry.supportsEmbeddings
                );
            };
            try protocolWriter.sendDaemonResponse(.catalog(entries: entries));
        case let .downloadStart(modelId):
            self.stateLock.lock();
            self.isDownloadStarted = true;
            self.stateLock.unlock();
            try protocolWriter.sendDaemonResponse(.downloadStarted(huggingfaceId: modelId));
        case .downloadStatus:
            self.stateLock.lock();
            let scriptedJob: DaemonDownloadJob?;
            if (self.downloadJobIndex < self.config.downloadJobs.count) {
                scriptedJob = self.config.downloadJobs[self.downloadJobIndex];
            } else {
                scriptedJob = self.config.downloadJobs.last ?? nil;
            }
            self.downloadJobIndex += 1;
            self.stateLock.unlock();
            try protocolWriter.sendDaemonResponse(.downloadStatus(job: scriptedJob));
        case let .defaultModelSet(modelId):
            self.stateLock.lock();
            self.currentDefaultModelId = modelId;
            self.stateLock.unlock();
            try protocolWriter.sendDaemonResponse(.defaultModelSet(defaultModelId: modelId));
        case let .chatGenerate(_, messages, settings, schemaJson):
            self.stateLock.lock();
            self.capturedMessages = messages;
            self.capturedSettings = settings;
            self.capturedSchemaJson = schemaJson;
            if let userImages: Array<ChatImageInput> = messages.first(where: { (message: ChatMessage) -> Bool in
                if case .user = message {
                    return true;
                }
                return false;
            }).flatMap({ (userMessage: ChatMessage) -> Array<ChatImageInput>? in
                if case let .user(_, images) = userMessage {
                    return images;
                }
                return nil;
            }) {
                self.capturedImages = userImages;
            }
            self.stateLock.unlock();
            for chatFragment: String in self.config.chatFragments {
                try protocolWriter.sendDaemonResponse(.chatGenerationText(text: chatFragment));
            }
            try protocolWriter.sendDaemonResponse(.chatGenerationCompleted(
                promptTokenCount: 7,
                generatedTokenCount: 2,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .endOfSequence
            ));
            try protocolWriter.close();
        case let .embedGenerate(model, _, _):
            let embeddingsResponse: DaemonResponse;
            switch (self.config.embeddingsOutcome) {
            case .completed:
                let residentModelId: String = self.residentModelIdWhileLocked() ?? model ?? "stub-model";
                embeddingsResponse = .embeddingsCompleted(
                    model: residentModelId,
                    vectors: [self.config.embeddingVector],
                    inputTokenCounts: [3]
                );
            case .contextLengthExceeded:
                embeddingsResponse = .embeddingsFailed(reason: .contextLengthExceeded(
                    actualTotalContextTokens: 5_000,
                    maximumContextTokens: 2_048
                ));
            }
            try protocolWriter.sendDaemonResponse(embeddingsResponse);
            try protocolWriter.close();
        }
    }

    private func residentModelIdWhileLocked() -> String? {
        return self.config.installedModels.first { (installedModel: StubInstalledModel) -> Bool in
            return installedModel.isResident;
        }?.modelId;
    }

    // MARK: - Capture accessors

    func takeCapturedImages() -> Array<ChatImageInput>? {
        self.stateLock.lock();
        let capturedImages: Array<ChatImageInput>? = self.capturedImages;
        self.capturedImages = nil;
        self.stateLock.unlock();
        return capturedImages;
    }

    func takeCapturedMessages() -> Array<ChatMessage>? {
        self.stateLock.lock();
        let capturedMessages: Array<ChatMessage>? = self.capturedMessages;
        self.capturedMessages = nil;
        self.stateLock.unlock();
        return capturedMessages;
    }

    func takeCapturedSettings() -> ChatGenerationSettings? {
        self.stateLock.lock();
        let capturedSettings: ChatGenerationSettings? = self.capturedSettings;
        self.capturedSettings = nil;
        self.stateLock.unlock();
        return capturedSettings;
    }

    func takeCapturedSchemaJson() -> String? {
        self.stateLock.lock();
        let capturedSchemaJson: String? = self.capturedSchemaJson;
        self.capturedSchemaJson = nil;
        self.stateLock.unlock();
        return capturedSchemaJson;
    }
}

enum StubDaemonError: Error {

    case bindFailed;
}

/// One scripted download job for the stub's `DownloadStatus` sequence.
func stubDownloadJob(
    _ huggingfaceId: String,
    _ state: String,
    _ bytesCompleted: UInt64,
    _ bytesTotal: UInt64,
    _ error: String?
) -> DaemonDownloadJob {
    return DaemonDownloadJob(
        huggingfaceId: huggingfaceId,
        state: state,
        bytesCompleted: bytesCompleted,
        bytesTotal: bytesTotal,
        error: error
    );
}
