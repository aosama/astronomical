import Foundation;

/**
 * The endpoint routes the daemon serves at startup. Mirrors the always-on
 * spine of the Rust application router: the health liveness probe and the
 * readiness probe driven by the worker health state. Generation and model
 * endpoints attach as their slices land.
 */
public enum RestEndpointRoutes {

    public static func foundationRouteTable(
        readinessProvider: @escaping @Sendable () -> WorkerHealthStatus
    ) -> RestRouteTable {
        var routeTable: RestRouteTable = RestRouteTable();
        routeTable.register(
            method: "GET",
            path: "/health",
            handler: { (_ request: RestHttpRequest) -> RestHttpResponse in
                return RestHttpResponse.text(statusCode: 200, body: "ok");
            });
        routeTable.register(
            method: "GET",
            path: "/ready",
            handler: { (_ request: RestHttpRequest) -> RestHttpResponse in
                let workerHealthStatus: WorkerHealthStatus = readinessProvider();
                let readinessStatusCode: Int = workerHealthStatus.isReady() ? 200 : 503;
                return RestHttpResponse.text(statusCode: readinessStatusCode, body: workerHealthStatus.readinessText());
            });
        return routeTable;
    }
}
