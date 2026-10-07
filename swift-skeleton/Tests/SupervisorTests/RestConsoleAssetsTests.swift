import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Embedded Observatory console journeys, migrating
 * apps/supervisor/tests/rest_api/application/observatory_assets.rs and
 * observatory_library.rs: the single-page shell renders at the root and
 * every named deep link with its labelled regions intact, the removed
 * destinations stay 404, each referenced script and stylesheet serves from
 * the bundle with its exact content type, the render stack serves under
 * /render/ with webfonts, and unknown REST paths stay 404.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestConsoleAssetsTests {

    static let htmlTitleMarker: String = "<title>Astronomical Observatory</title>";
    static let chatTranscriptMarker: String =
        "<section id=\"chat-transcript\" class=\"chat-transcript\" aria-live=\"polite\" aria-label=\"Chat transcript\"></section>";
    static let navigationLabelMarker: String = "aria-label=\"Observatory sections\"";
    static let connectRegionMarker: String =
        "data-observatory-view=\"connect\" aria-labelledby=\"connect-title\"";
    static let connectNavigationMarker: String = "data-observatory-destination=\"connect\"";

    @Test
    func should_serve_the_embedded_observatory_index_html_at_root() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let shellResponse: RestHttpResponse = try consoleJourney.get("/");

        #expect(shellResponse.statusCode == 200);
        #expect(shellResponse.contentType.hasPrefix("text/html"));
        let shellText: String = shellResponse.bodyText();
        #expect(shellText.contains(RestConsoleAssetsTests.htmlTitleMarker),
            "the observatory shell should declare its title");
        #expect(shellText.contains(RestConsoleAssetsTests.chatTranscriptMarker),
            "the chat transcript should be a named live region");
        #expect(shellText.contains(RestConsoleAssetsTests.navigationLabelMarker));
        #expect(shellText.contains(RestConsoleAssetsTests.connectRegionMarker),
            "the Observatory should expose the coding-agent connection region");
        #expect(shellText.contains(RestConsoleAssetsTests.connectNavigationMarker));
        #expect(shellText.contains("src=\"/connect.js\""),
            "the served shell should load the connection material script");
    }

    @Test
    func should_serve_the_observatory_shell_at_each_named_deep_link() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        for observatoryPath: String in [
            "/overview", "/chat", "/model", "/library", "/connect", "/settings",
        ] {
            let shellResponse: RestHttpResponse = try consoleJourney.get(observatoryPath);
            #expect(
                shellResponse.statusCode == 200,
                Comment(rawValue: "path: \(observatoryPath)"));
            #expect(
                shellResponse.contentType.hasPrefix("text/html"),
                Comment(rawValue: "deep-link response should be HTML for \(observatoryPath)"));
        }
    }

    @Test
    func should_not_serve_removed_memory_cache_and_optimizer_destinations() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        for removedObservatoryPath: String in ["/memory", "/cache", "/optimizer"] {
            let removedOutcome: RestRouteOutcome = consoleJourney.routeTable.outcome(
                method: "GET",
                path: removedObservatoryPath);
            guard case .notFound = removedOutcome else {
                Issue.record(
                    Comment(rawValue: "path: \(removedObservatoryPath) must stay 404"));
                continue;
            }
        }
    }

    @Test
    func should_return_not_found_for_an_unknown_rest_path() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let unknownOutcome: RestRouteOutcome = consoleJourney.routeTable.outcome(
            method: "GET",
            path: "/this/is/not/a/route");

        guard case .notFound = unknownOutcome else {
            Issue.record("unknown REST paths must stay 404");
            return;
        }
    }

    @Test
    func should_serve_the_embedded_observatory_javascript_with_correct_content_type() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let scriptResponse: RestHttpResponse = try consoleJourney.get("/console.js");

        #expect(scriptResponse.statusCode == 200);
        #expect(scriptResponse.contentType.hasPrefix("application/javascript"));
        #expect(scriptResponse.bodyText().isEmpty == false);
    }

    @Test
    func should_serve_the_embedded_compact_overview_javascript() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let scriptResponse: RestHttpResponse = try consoleJourney.get("/overview-compact.js");

        #expect(scriptResponse.statusCode == 200);
        #expect(scriptResponse.contentType.hasPrefix("application/javascript"));
        #expect(
            scriptResponse.bodyText().contains("reconciledMlxMemorySegmentBytes"),
            "the compact overview script should carry the memory segment rendering");
    }

    @Test
    func should_serve_the_embedded_connection_material_script() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let scriptResponse: RestHttpResponse = try consoleJourney.get("/connect.js");

        #expect(scriptResponse.statusCode == 200);
        #expect(scriptResponse.contentType.hasPrefix("application/javascript"));
        let scriptText: String = scriptResponse.bodyText();
        #expect(scriptText.contains("connect-api-base-url"));
        #expect(scriptText.contains("openai-completions"));
    }

    @Test
    func should_serve_the_embedded_memory_control_script() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let scriptResponse: RestHttpResponse = try consoleJourney.get("/memory-control.js");

        #expect(scriptResponse.statusCode == 200);
        #expect(
            scriptResponse.bodyText().contains("maximum-mlx-memory"),
            "the memory control script should drive the ceiling control");
    }

    @Test
    func should_serve_the_embedded_observatory_playground_javascript() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let scriptResponse: RestHttpResponse = try consoleJourney.get("/playground.js");

        #expect(scriptResponse.statusCode == 200);
        #expect(scriptResponse.bodyText().isEmpty == false);
    }

    @Test
    func should_serve_the_embedded_observatory_stylesheet_with_correct_content_type() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let stylesheetResponse: RestHttpResponse = try consoleJourney.get("/console.css");

        #expect(stylesheetResponse.statusCode == 200);
        #expect(stylesheetResponse.contentType.hasPrefix("text/css"));
        #expect(stylesheetResponse.bodyText().isEmpty == false);
    }

    @Test
    func should_serve_the_render_stack_and_its_webfonts() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let renderScriptResponse: RestHttpResponse = try consoleJourney.get("/render/canvas-render.js");
        #expect(renderScriptResponse.statusCode == 200);
        #expect(renderScriptResponse.contentType.hasPrefix("application/javascript"));
        #expect(renderScriptResponse.bodyText().isEmpty == false);

        let vendorScriptResponse: RestHttpResponse = try consoleJourney.get("/render/vendor/morphdom-umd.min.js");
        #expect(vendorScriptResponse.statusCode == 200);
        #expect(vendorScriptResponse.bodyText().isEmpty == false);

        let stylesheetResponse: RestHttpResponse = try consoleJourney.get("/render/vendor/katex/katex.min.css");
        #expect(stylesheetResponse.statusCode == 200);
        #expect(stylesheetResponse.contentType.hasPrefix("text/css"));

        let fontResponse: RestHttpResponse = try consoleJourney.get(
            "/render/vendor/katex/fonts/KaTeX_Main-Regular.woff2");
        #expect(fontResponse.statusCode == 200);
        #expect(fontResponse.contentType.hasPrefix("font/woff2"));
        #expect(fontResponse.bodyBytes.isEmpty == false);

        let unknownRenderOutcome: RestRouteOutcome = consoleJourney.routeTable.outcome(
            method: "GET",
            path: "/render/not-an-asset.txt");
        guard case .handler = unknownRenderOutcome else {
            Issue.record("the render prefix must own every path under it");
            return;
        }
        let unknownRenderResponse: RestHttpResponse = try consoleJourney.get("/render/not-an-asset.txt");
        #expect(unknownRenderResponse.statusCode == 404);
    }

    @Test
    func should_serve_a_labelled_library_destination_at_its_deep_link() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let shellResponse: RestHttpResponse = try consoleJourney.get("/library");

        #expect(shellResponse.statusCode == 200);
        #expect(shellResponse.contentType.hasPrefix("text/html"));
        let shellText: String = shellResponse.bodyText();
        let libraryNavigationTag: String = try RestConsoleAssetsTests.openingTagWithAttribute(
            shellText, attribute: "data-observatory-destination=\"library\"");
        #expect(libraryNavigationTag.contains("aria-controls=\"library-view\""));
        let libraryViewTag: String = try RestConsoleAssetsTests.openingTagWithId(
            shellText, elementId: "library-view");
        #expect(libraryViewTag.contains("data-observatory-view=\"library\""));
        #expect(libraryViewTag.contains("aria-labelledby=\"library-title\""));
        let libraryCatalogTag: String = try RestConsoleAssetsTests.openingTagWithId(
            shellText, elementId: "library-catalog");
        #expect(libraryCatalogTag.contains("role=\"status\"") == false);
        #expect(libraryCatalogTag.contains("aria-live=") == false);
        let libraryStatusTag: String = try RestConsoleAssetsTests.openingTagWithId(
            shellText, elementId: "library-catalog-status");
        #expect(libraryStatusTag.contains("role=\"status\""));
        #expect(libraryStatusTag.contains("aria-live=\"polite\""));
    }

    @Test
    func should_serve_the_embedded_library_javascript() throws {
        let consoleJourney: ConsoleAssetsJourney = try ConsoleAssetsJourney.launch();
        defer { consoleJourney.dispose() }

        let libraryResponse: RestHttpResponse = try consoleJourney.get("/library.js");

        #expect(libraryResponse.statusCode == 200);
        #expect(libraryResponse.contentType.hasPrefix("application/javascript"));
        #expect(libraryResponse.bodyText().isEmpty == false);
    }

    // MARK: Support

    private static func openingTagWithId(
        _ htmlDocument: String,
        elementId: String
    ) throws -> String {
        return try RestConsoleAssetsTests.openingTagWithAttribute(
            htmlDocument, attribute: "id=\"\(elementId)\"");
    }

    private static func openingTagWithAttribute(
        _ htmlDocument: String,
        attribute: String
    ) throws -> String {
        guard let attributeRange: Range<String.Index> = htmlDocument.range(of: attribute) else {
            throw ConsoleAssetsJourneyFailure.attributeMissing(attribute);
        }
        guard let tagStartIndex: String.Index = htmlDocument[..<attributeRange.lowerBound]
            .lastIndex(of: "<") else {
            throw ConsoleAssetsJourneyFailure.openingTagMissing(attribute);
        }
        guard let tagEndIndex: String.Index = htmlDocument[attributeRange.lowerBound...]
            .firstIndex(of: ">") else {
            throw ConsoleAssetsJourneyFailure.openingTagUnterminated(attribute);
        }
        return String(htmlDocument[tagStartIndex...tagEndIndex]);
    }
}

/// One console journey: the serving route table over a fresh health state.
final class ConsoleAssetsJourney {

    let routeTable: RestRouteTable;
    private let homeDirectoryUrl: URL;

    private init(routeTable: RestRouteTable, homeDirectoryUrl: URL) {
        self.routeTable = routeTable;
        self.homeDirectoryUrl = homeDirectoryUrl;
    }

    static func launch() throws -> ConsoleAssetsJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-console-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            workerHealthState: WorkerHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity());
        return ConsoleAssetsJourney(routeTable: routeTable, homeDirectoryUrl: homeDirectoryUrl);
    }

    func dispose() -> Void {
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    func get(_ routePath: String) throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(method: "GET", path: routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ConsoleAssetsJourneyFailure.routeMissing(routePath);
        }
        return try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: "GET",
            path: routePath));
    }
}

extension RestHttpResponse {

    /// The response body as UTF-8 text, the console journeys' read shape.
    func bodyText() -> String {
        return String(decoding: self.bodyBytes, as: UTF8.self);
    }
}

/// Typed failures of the console journeys.
enum ConsoleAssetsJourneyFailure: Error {

    case routeMissing(String);
    case attributeMissing(String);
    case openingTagMissing(String);
    case openingTagUnterminated(String);
}
