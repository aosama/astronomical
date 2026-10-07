import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic journeys for the POST /v1/control/shutdown surface, migrating
 * apps/supervisor/tests/rest_api/config_reload/shutdown.rs: the endpoint
 * triggers the daemon's graceful-shutdown signal exactly once, the request
 * persists for observers that attach later, the endpoint is POST-only, and
 * it is absent without a controller.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestShutdownControlTests {

    @Test
    func should_trigger_the_graceful_shutdown_signal_when_the_endpoint_is_called() throws {
        let shutdownObserver: ShutdownSignalObserver = ShutdownSignalObserver();
        let shutdownController: ShutdownController = ShutdownController();
        shutdownController.subscribe({ shutdownObserver.recordSignal() });
        let routeTable: RestRouteTable = try RestShutdownControlTests.routeTable(shutdownController);

        let shutdownResponse: RestHttpResponse = try RestShutdownControlTests.postShutdown(
            routeTable: routeTable);

        #expect(shutdownResponse.statusCode == 202);
        let shutdownDocument: [String: Any] = try RestShutdownControlTests.decodeObject(shutdownResponse);
        #expect(shutdownDocument["status"] as? String == "shutting_down");
        #expect(shutdownObserver.awaitSignal(within: 2));

        // Repeated requests stay idempotent and still answer 202.
        let repeatResponse: RestHttpResponse = try RestShutdownControlTests.postShutdown(
            routeTable: routeTable);
        #expect(repeatResponse.statusCode == 202);
        #expect(shutdownObserver.recordedSignalCount == 1);
    }

    @Test
    func should_persist_the_shutdown_request_for_observers_that_attach_later() throws {
        let shutdownObserver: ShutdownSignalObserver = ShutdownSignalObserver();
        let shutdownController: ShutdownController = ShutdownController();

        #expect(shutdownController.requestShutdown());
        #expect(shutdownController.requestShutdown() == false);

        shutdownController.subscribe({ shutdownObserver.recordSignal() });
        #expect(shutdownObserver.awaitSignal(within: 2));
        #expect(shutdownObserver.recordedSignalCount == 1);
    }

    @Test
    func should_keep_the_shutdown_endpoint_post_only() throws {
        let routeTable: RestRouteTable = try RestShutdownControlTests.routeTable(ShutdownController());

        let routeOutcome: RestRouteOutcome = routeTable.outcome(
            method: "GET",
            path: RestShutdownControlEndpoint.routePath);
        guard case .methodNotAllowed = routeOutcome else {
            Issue.record("the shutdown endpoint must be POST-only, got \(routeOutcome)");
            return;
        }
    }

    @Test
    func should_not_offer_shutdown_without_a_controller() throws {
        let routeTable: RestRouteTable = try RestShutdownControlTests.routeTable(nil);

        let routeOutcome: RestRouteOutcome = routeTable.outcome(
            method: "POST",
            path: RestShutdownControlEndpoint.routePath);
        guard case .notFound = routeOutcome else {
            Issue.record("the shutdown route must be absent without a controller, got \(routeOutcome)");
            return;
        }
    }
}

/// Captures the shutdown signal firings so journeys can observe them with a
/// bounded wait.
final class ShutdownSignalObserver: @unchecked Sendable {

    private let signalLock: NSLock;
    private let signalDispatchGroup: DispatchGroup;
    private var recordedSignalCountValue: Int;

    init() {
        self.signalLock = NSLock();
        self.signalDispatchGroup = DispatchGroup();
        self.signalDispatchGroup.enter();
        self.recordedSignalCountValue = 0;
    }

    var recordedSignalCount: Int {
        self.signalLock.lock();
        defer { self.signalLock.unlock(); }
        return self.recordedSignalCountValue;
    }

    func recordSignal() -> Void {
        self.signalLock.lock();
        self.recordedSignalCountValue += 1;
        self.signalLock.unlock();
        self.signalDispatchGroup.leave();
    }

    func awaitSignal(within maximumWait: TimeInterval) -> Bool {
        return self.signalDispatchGroup.wait(
            timeout: .now() + maximumWait) == .success;
    }
}

extension RestShutdownControlTests {

    static func routeTable(_ shutdownController: ShutdownController?) throws -> RestRouteTable {
        return RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            workerHealthState: WorkerHealthState(),
            instancePaths: RestChatJourneySupport.journeyInstancePaths(),
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            shutdownController: shutdownController);
    }

    static func postShutdown(routeTable: RestRouteTable) throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = routeTable.outcome(
            method: "POST",
            path: RestShutdownControlEndpoint.routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            Issue.record("the shutdown route must exist, got \(routeOutcome)");
            throw RestShutdownControlTestsFailure.shutdownRouteMissing;
        }
        return try routeHandler(RestHttpRequest(
            method: "POST",
            path: RestShutdownControlEndpoint.routePath,
            requestTarget: RestShutdownControlEndpoint.routePath,
            headersByLowercasedName: [:],
            bodyBytes: Data()));
    }

    static func decodeObject(_ response: RestHttpResponse) throws -> [String: Any] {
        let responseDocument: Any = try JSONSerialization.jsonObject(
            with: response.bodyBytes,
            options: []);
        guard let responseDocumentObject = responseDocument as? [String: Any] else {
            throw RestShutdownControlTestsFailure.responseNotAnObject;
        }
        return responseDocumentObject;
    }
}

/// Typed failures of the shutdown journey plumbing.
enum RestShutdownControlTestsFailure: Error {

    case shutdownRouteMissing;
    case responseNotAnObject;
}
