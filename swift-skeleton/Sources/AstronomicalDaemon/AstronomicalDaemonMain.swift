import Foundation;

import AstronomicalConfig;
import IpcProtocol;

import Supervisor;

/// The astronomicald daemon entry point.
///
/// Mirrors the startup spine of apps/supervisor/src/main.rs: parse process
/// arguments, resolve the instance paths, take the exclusive instance lock,
/// start the daemon IPC service, and serve until SIGINT or SIGTERM. The REST
/// surface, worker containment, and library slices attach to this spine as
/// they land. Stable's lifecycle stays owned by macOS LaunchAgents; this
/// binary only ever runs the instance its arguments select, and the
/// Development instance is the only one tests may start.
@main
struct AstronomicalDaemonMain {

    static func main() {
        let daemonCommand: DaemonCommand;
        do {
            daemonCommand = try DaemonArguments.parse(processArguments: Array(CommandLine.arguments));
        } catch let argumentError as DaemonArgumentError {
            // The usage block rides the rejection the way the Rust argument
            // parser prints it, so a bad invocation teaches the fix inline.
            FileHandle.standardError.write(Data("astronomicald: \(argumentError)\n".utf8));
            FileHandle.standardError.write(Data("\(DaemonArguments.helpText())\n".utf8));
            exit(2);
        } catch {
            FileHandle.standardError.write(Data("astronomicald: unexpected argument failure\n".utf8));
            exit(2);
        }
        switch (daemonCommand) {
        case .help:
            print(DaemonArguments.helpText());
            exit(0);
        case .version:
            print("astronomicald \(buildIdentityLine())");
            exit(0);
        case let .run(daemonArguments):
            run(daemonArguments: daemonArguments);
        }
    }

    private static func run(daemonArguments: DaemonArguments) -> Never {
        // A local client that disconnects mid-frame must end as a failed
        // write on the serving thread, never as a process-killing SIGPIPE;
        // the Rust daemon inherits this behavior from Rust's default.
        signal(SIGPIPE, SIG_IGN);
        let instancePaths: AstronomicalInstancePaths;
        do {
            instancePaths = try daemonArguments.resolveInstancePaths();
        } catch {
            FileHandle.standardError.write(Data("astronomicald: could not resolve instance paths: \(error)\n".utf8));
            exit(2);
        }
        let instanceLock: AstronomicalInstanceLock;
        do {
            instanceLock = try AstronomicalInstanceLock.acquire(
                lockFilePath: instancePaths.instanceLockFilePath.string);
        } catch AstronomicalInstanceLockError.alreadyRunning {
            FileHandle.standardError.write(Data(
                "astronomicald: Astronomical is already running for the selected instance\n".utf8));
            exit(1);
        } catch {
            FileHandle.standardError.write(Data("astronomicald: could not take the instance lock: \(error)\n".utf8));
            exit(2);
        }
        // The worker-launch configuration resolves before the IPC endpoint
        // opens: an unresolvable config is a startup failure, and the
        // operator sees what the daemon will serve before anything connects.
        let runtimeConfigResolver: ResolvedRuntimeConfigResolver;
        do {
            runtimeConfigResolver = ResolvedRuntimeConfigResolver(
                instancePaths: instancePaths,
                fallbackWorkerExecutablePath: try FallbackWorkerExecutablePath.derive());
        } catch {
            FileHandle.standardError.write(Data("astronomicald: could not locate the worker executable: \(error)\n".utf8));
            exit(2);
        }
        let resolvedRuntimeConfig: ResolvedRuntimeConfig;
        do {
            resolvedRuntimeConfig = try runtimeConfigResolver.load();
        } catch {
            FileHandle.standardError.write(Data("astronomicald: could not resolve the runtime configuration: \(error)\n".utf8));
            exit(2);
        }
        // The attribution plumbing opens before the worker launches: the
        // instance's logs directory receives the per-generation performance
        // rows, the completion rows when the operator enabled that toggle,
        // and the supervisor-owned operation attribution rows when the
        // performance-attribution diagnostics flag is on.
        let loggingDirectory: FilePath = instancePaths.loggingDirectory;
        do {
            try FileManager.default.createDirectory(
                atPath: loggingDirectory.string,
                withIntermediateDirectories: true);
        } catch {
            FileHandle.standardError.write(Data(
                "astronomicald: could not create the logs directory: \(error)\n".utf8));
            exit(2);
        }
        let supervisorAttributionLog: SupervisorPerformanceAttributionLog;
        do {
            supervisorAttributionLog = try SupervisorPerformanceAttributionLog.open(
                logDirectory: loggingDirectory,
                performanceAttributionEnabled: resolvedRuntimeConfig.performanceAttributionEnabled);
        } catch {
            FileHandle.standardError.write(Data(
                "astronomicald: failed to create the supervisor performance-attribution log: \(error)\n".utf8));
            exit(2);
        }
        let generationPerformanceLog: GenerationPerformanceLog;
        let completionAttributionLog: CompletionAttributionLog;
        do {
            generationPerformanceLog = try GenerationPerformanceLog.open(
                logDirectory: loggingDirectory);
            completionAttributionLog = try CompletionAttributionLog.open(
                logDirectory: loggingDirectory,
                completionAttributionEnabled: resolvedRuntimeConfig.completionAttributionEnabled);
        } catch {
            FileHandle.standardError.write(Data(
                "astronomicald: failed to create the generation or completion attribution logs: \(error)\n".utf8));
            exit(2);
        }
        // The operator log keeps the supervisor's own worker-failure
        // diagnostics — exit status and stderr tail — durable in the same
        // logs directory, lossy so a slow disk never stalls serving.
        let supervisorOperatorLog: SupervisorOperatorLog = SupervisorOperatorLog.open(
            logDirectory: loggingDirectory);
        // The supervisor owns the one live health snapshot: the REST routes
        // and the daemon IPC status verb read from it, so every surface sees
        // the same worker facts.
        let workerSupervisor: WorkerSupervisor;
        do {
            workerSupervisor = try WorkerSupervisor.launch(
                workerExecutablePath: resolvedRuntimeConfig.workerExecutablePath.string,
                workerArguments: [],
                workerStartupConfiguration: resolvedRuntimeConfig.workerStartupConfiguration(),
                modelPolicyCatalog: resolvedRuntimeConfig.modelPolicyCatalog,
                modelLoadTimeout: AstronomicalDaemonMain.workerModelLoadTimeoutSeconds,
                generationPerformanceLog: generationPerformanceLog,
                completionAttributionLog: completionAttributionLog,
                operatorLog: supervisorOperatorLog);
        } catch {
            FileHandle.standardError.write(Data(
                "astronomicald worker unavailable: \(error)\n".utf8));
            workerSupervisor = WorkerSupervisor.unavailable(
                modelPolicyCatalog: resolvedRuntimeConfig.modelPolicyCatalog);
        }
        let workerHealthState: WorkerHealthState = workerSupervisor.ownedHealthState();
        let chatRequestIdAllocator: ChatRequestIdAllocator = ChatRequestIdAllocator();
        // The menu bar app requests the same graceful close over HTTP.
        let shutdownController: ShutdownController = ShutdownController();
        let configTransitionState: ConfigTransitionState = ConfigTransitionState(
            reloadableConfig: resolvedRuntimeConfig,
            configuredConfigSnapshot: resolvedRuntimeConfig);
        // The Library download manager: the bundled release catalog joined
        // with live discovery, transferring through the pinned Hub client.
        // Recovery runs before the REST endpoint accepts anything, so a job
        // interrupted mid-publish resumes under the same identity.
        let libraryDownloadCatalog: DownloadCatalog;
        do {
            // Catalog loading is a supervisor-owned operation: its duration
            // and entry count land in the supervisor attribution log when the
            // diagnostics flag is on, and never block the load otherwise.
            let catalogOutcome: Result<DownloadCatalog, any Error> = try supervisorAttributionLog.measureOperation(
                operation: .libraryCatalogLoad,
                measuredOperation: { () -> Result<DownloadCatalog, any Error> in
                    return Result(catching: { () throws -> DownloadCatalog in
                        return try DownloadCatalog.loadBundled();
                    });
                },
                describeMeasurement: { (outcome: Result<DownloadCatalog, any Error>) -> SupervisorPerformanceMeasurement in
                    switch (outcome) {
                    case let .success(loadedCatalog):
                        return SupervisorPerformanceMeasurement.successfulCatalogLoad(
                            catalogEntryCount: loadedCatalog.entryCount);
                    case .failure:
                        return SupervisorPerformanceMeasurement.failure();
                    }
                });
            libraryDownloadCatalog = try catalogOutcome.get();
        } catch {
            FileHandle.standardError.write(Data(
                "astronomicald: the bundled library catalog is invalid: \(error)\n".utf8));
            exit(2);
        }
        let libraryJobRecordStore: LibraryDownloadJobRecordStore = LibraryDownloadJobRecordStore(
            stateDirectory: instancePaths.stateDirectory);
        let libraryDownloadCoordinator: LibraryDownloadCoordinator = LibraryDownloadCoordinator(
            downloadCatalog: libraryDownloadCatalog,
            modelsDirectory: instancePaths.modelsDirectory,
            stateDirectory: instancePaths.stateDirectory,
            hubEndpoint: HubDownloadService.productionHubEndpoint,
            discoveryRefresh: {
                // Publication refresh re-resolves discovery so the models
                // advertisement and catalog readiness follow the new files.
                let refreshedRuntimeConfig: ResolvedRuntimeConfig = try runtimeConfigResolver.load();
                configTransitionState.replaceReloadableConfig(refreshedRuntimeConfig);
            },
            availableCapacityBytes: { (volumeDirectory: FilePath) -> UInt64? in
                let volumeUrl: URL = URL(fileURLWithPath: volumeDirectory.string);
                guard let volumeValues: URLResourceValues = try? volumeUrl.resourceValues(
                    forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
                    let availableCapacity: Int64 = volumeValues.volumeAvailableCapacityForImportantUsage
                else {
                    return nil;
                }
                return UInt64(availableCapacity);
            });
        let libraryRecoverySemaphore: DispatchSemaphore = DispatchSemaphore(value: 0);
        Task<Void, Never> {
            await libraryDownloadCoordinator.recoverStartupState();
            libraryRecoverySemaphore.signal();
        }
        libraryRecoverySemaphore.wait();
        let restServer: RestHttpServer;
        do {
            restServer = try RestHttpServer.start(
                bindEndpoint: resolvedRuntimeConfig.bindEndpoint,
                routeTable: RestEndpointRoutes.servingRouteTable(
                    resolvedRuntimeConfig: resolvedRuntimeConfig,
                    workerHealthState: workerHealthState,
                    instancePaths: instancePaths,
                    buildIdentity: ApplicationBuildIdentity.current(),
                    chatContext: RestChatRouteContext(
                        chatExecutor: workerSupervisor,
                        requestIdAllocator: chatRequestIdAllocator,
                        resolvedRuntimeConfig: resolvedRuntimeConfig,
                        instancePaths: instancePaths,
                        liveResolvedRuntimeConfigProvider: {
                            return configTransitionState.currentReloadableConfig();
                        }),
                    responsesContext: RestResponsesRouteContext(
                        responsesExecutor: workerSupervisor,
                        requestIdAllocator: chatRequestIdAllocator,
                        resolvedRuntimeConfig: resolvedRuntimeConfig,
                        instancePaths: instancePaths,
                        liveResolvedRuntimeConfigProvider: {
                            return configTransitionState.currentReloadableConfig();
                        }),
                    cacheClearContext: RestCacheClearRouteContext(
                        cacheClearExecutor: workerSupervisor),
                    shutdownController: shutdownController,
                    memoryContext: RestMaximumMlxMemoryRouteContext(
                        workerControl: workerSupervisor,
                        runtimeConfigResolver: runtimeConfigResolver,
                        transitionState: configTransitionState),
                    configReloadContext: RestConfigReloadRouteContext(
                        transitionState: configTransitionState,
                        runtimeConfigResolver: runtimeConfigResolver,
                        workerControl: workerSupervisor,
                        workerHealthState: workerHealthState,
                        generationActivityIdleProvider: { return true }),
                    configRevealContext: RestConfigRevealRouteContext(revealActiveConfig: {
                        let configFilePath: FilePath = instancePaths.configFilePath;
                        return ConfigRevealOpener.revealInFinder(configFilePath: configFilePath);
                    }),
                    libraryCatalogContext: RestLibraryCatalogRouteContext(
                        downloadCatalog: libraryDownloadCatalog,
                        discoveredModelsProvider: {
                            return configTransitionState.currentReloadableConfig().discoveredModels;
                        },
                        validatedPublicationsProvider: {
                            return libraryDownloadCoordinator.validatedPublications.snapshot();
                        },
                        currentJobProvider: {
                            guard let jobRecord: LibraryDownloadJobRecord = try? libraryJobRecordStore.load() else {
                                return nil;
                            }
                            return LibraryDownloadJobSummary(
                                huggingfaceId: jobRecord.huggingfaceId,
                                stateName: jobRecord.state.rawValue);
                        },
                        destinationDirectoryProvider: { (huggingfaceId: String) -> String? in
                            return libraryDownloadCoordinator.destinationDirectory(huggingfaceId: huggingfaceId).string;
                        }),
                    libraryDownloadContext: RestLibraryDownloadRouteContext(
                        coordinator: libraryDownloadCoordinator)),
                corsPolicy: RestCorsPolicy.canvasShell());
        } catch {
            FileHandle.standardError.write(Data("astronomicald: could not start the REST endpoint: \(error)\n".utf8));
            exit(2);
        }
        let service: DaemonIpcService;
        do {
            service = try DaemonIpcService.start(
                instancePaths: instancePaths,
                healthProvider: { return workerHealthState.daemonStatusReport(); },
                chatContext: DaemonIpcChatContext(
                    chatExecutor: workerSupervisor,
                    resolvedRuntimeConfig: resolvedRuntimeConfig,
                    instancePaths: instancePaths),
                modelsContext: DaemonIpcModelsContext(
                    embeddingsExecutor: workerSupervisor,
                    liveResolvedRuntimeConfigProvider: {
                        return configTransitionState.currentReloadableConfig();
                    },
                    downloadCatalog: libraryDownloadCatalog,
                    libraryDownloadCoordinator: libraryDownloadCoordinator,
                    instancePaths: instancePaths));
        } catch {
            FileHandle.standardError.write(Data("astronomicald: could not start the daemon IPC service: \(error)\n".utf8));
            exit(2);
        }
        // Live progress instead of silent waits: the operator sees the
        // resolved serving identity and the endpoints the daemon answers on
        // before anything can talk to it.
        let generationPrefix: String = String(resolvedRuntimeConfig.configurationGeneration.prefix(12));
        print("astronomicald \(daemonArguments.runtimeInstance.rawInstanceName) resolved "
            + "\(resolvedRuntimeConfig.discoveredModels.count) models, "
            + "generation \(generationPrefix), "
            + "serving IPC on \(service.socketPath), "
            + "serving REST on http://\(restServer.boundEndpoint.description)");
        fflush(stdout);

        let shutdownSemaphore = DispatchSemaphore(value: 0);
        let signalSourceShutdown: @Sendable () -> Void = {
            restServer.stop();
            service.shutdown();
            _ = try? workerSupervisor.shutdown();
            shutdownSemaphore.signal();
        };
        shutdownController.subscribe(signalSourceShutdown);
        for signalNumber: Int32 in [SIGINT, SIGTERM] {
            let signalSource: DispatchSourceSignal = DispatchSource.makeSignalSource(
                signal: signalNumber,
                queue: DispatchQueue.global());
            signal(signalNumber, SIG_IGN);
            signalSource.setEventHandler(handler: signalSourceShutdown);
            signalSource.resume();
            // Keep the source alive for the process lifetime.
            AllSignalSources.append(signalSource);
        }

        shutdownSemaphore.wait();
        // The requesting handler signals shutdown from inside its own call,
        // before its 202 has been written; this bounded beat lets the
        // per-connection thread flush that reply before the process exits.
        // Half a second absorbs a fully loaded machine and stays
        // imperceptible next to process teardown.
        Thread.sleep(forTimeInterval: 0.5);
        // The lock releases when this process exits; instanceLock is held so
        // the compiler keeps it alive for the whole serving lifetime.
        _ = instanceLock;
        exit(0);
    }

    private static func buildIdentityLine() -> String {
        let buildIdentity: String? = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleVersion") as? String;
        return buildIdentity.map { (identity: String) -> String in identity } ?? "0.0.0-dev";
    }
}

/// Signal sources must stay retained or Dispatch tears them down immediately.
/// The daemon is single-threaded at this boundary, so the escape is safe here.
private nonisolated(unsafe) var AllSignalSources: Array<DispatchSourceSignal> = Array();

extension AstronomicalDaemonMain {

    /// The bounded wait for a worker's startup or model-load acknowledgement,
    /// mirroring apps/supervisor/src/main.rs's WORKER_MODEL_LOAD_TIMEOUT.
    static let workerModelLoadTimeoutSeconds: TimeInterval = 60;
}
