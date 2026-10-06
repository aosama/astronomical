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
            FileHandle.standardError.write(Data("astronomicald: \(argumentError)\n".utf8));
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
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        let restServer: RestHttpServer;
        do {
            restServer = try RestHttpServer.start(
                bindEndpoint: resolvedRuntimeConfig.bindEndpoint,
                routeTable: RestEndpointRoutes.foundationRouteTable(readinessProvider: {
                    return workerHealthState.currentSnapshot().status;
                }));
        } catch {
            FileHandle.standardError.write(Data("astronomicald: could not start the REST endpoint: \(error)\n".utf8));
            exit(2);
        }
        let service: DaemonIpcService;
        do {
            service = try DaemonIpcService.start(
                instancePaths: instancePaths,
                healthProvider: { return workerHealthState.daemonStatusReport(); });
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
        let signalSourceShutdown: () -> Void = {
            restServer.stop();
            service.shutdown();
            shutdownSemaphore.signal();
        };
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
