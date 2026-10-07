import Foundation;

/**
 * The embedded Observatory console, porting apps/supervisor/src/
 * console_assets.rs + console_render_assets.rs. The shell and its assets
 * live once at apps/supervisor/console — reached through a symlinked
 * resource directory the Swift package copies into the Supervisor bundle
 * at build time — so serving has no filesystem dependency and no separate
 * frontend build step.
 */
enum ConsoleAssets {

    static let htmlContentType: String = "text/html; charset=utf-8";
    static let javascriptContentType: String = "application/javascript; charset=utf-8";
    static let cssContentType: String = "text/css; charset=utf-8";
    static let woff2ContentType: String = "font/woff2";

    /// The deep links that render the single-page shell.
    static let shellRoutes: Array<String> = [
        "/", "/overview", "/chat", "/model", "/library", "/connect", "/settings",
    ];

    /// The exact asset routes the shell references, mapped to their file
    /// names inside the bundled console directory.
    static let assetRoutes: Array<(routePath: String, relativePath: String)> = [
        ("/console.js", "console.js"),
        ("/library.js", "library.js"),
        ("/library-render.js", "library-render.js"),
        ("/overview-compact.js", "overview-compact.js"),
        ("/memory-control.js", "memory-control.js"),
        ("/connect.js", "connect.js"),
        ("/playground.js", "playground.js"),
        ("/console.css", "console.css"),
        ("/library.css", "library.css"),
    ];

    /// Serves one bundled console file as UTF-8 text with its content type.
    static func textResponse(
        relativePath: String,
        contentType: String
    ) -> RestHttpResponse? {
        guard let consoleText: String = ConsoleAssets.readConsoleText(relativePath) else {
            return nil;
        }
        return RestHttpResponse(
            statusCode: 200,
            contentType: contentType,
            bodyBytes: Data(consoleText.utf8));
    }

    /// Serves the render stack: JavaScript and stylesheet entries under
    /// render/, and the KaTeX webfonts beside them. Everything else —
    /// including any path escaping the render directory — is absent.
    static func renderAssetResponse(renderRelativePath: String) -> RestHttpResponse? {
        if renderRelativePath.hasSuffix(".js") || renderRelativePath.hasSuffix(".css") {
            return ConsoleAssets.textResponse(
                relativePath: "render/\(renderRelativePath)",
                contentType: renderRelativePath.hasSuffix(".css")
                    ? ConsoleAssets.cssContentType
                    : ConsoleAssets.javascriptContentType);
        }
        if renderRelativePath.hasPrefix("vendor/katex/fonts/"),
           renderRelativePath.hasSuffix(".woff2") {
            return ConsoleAssets.fontResponse(relativePath: "render/\(renderRelativePath)");
        }
        return nil;
    }

    private static func fontResponse(relativePath: String) -> RestHttpResponse? {
        guard let consoleResourceUrl: URL = ConsoleAssets.consoleResourceUrl(relativePath),
              let fontBytes: Data = try? Data(contentsOf: consoleResourceUrl) else {
            return nil;
        }
        return RestHttpResponse(
            statusCode: 200,
            contentType: ConsoleAssets.woff2ContentType,
            bodyBytes: fontBytes);
    }

    private static func readConsoleText(_ relativePath: String) -> String? {
        guard let consoleResourceUrl: URL = ConsoleAssets.consoleResourceUrl(relativePath) else {
            return nil;
        }
        return try? String(contentsOf: consoleResourceUrl, encoding: .utf8);
    }

    /// Resolves one file inside the bundled console directory, rejecting any
    /// path that climbs out of it.
    private static func consoleResourceUrl(_ relativePath: String) -> URL? {
        let consoleDirectoryUrl: URL? = Bundle.module.url(
            forResource: "console",
            withExtension: nil);
        guard let consoleDirectoryUrl = consoleDirectoryUrl else {
            return nil;
        }
        let candidateUrl: URL = consoleDirectoryUrl.appendingPathComponent(relativePath);
        // A resolved path that leaves the console directory never serves.
        guard candidateUrl.standardizedFileURL.path
            .hasPrefix(consoleDirectoryUrl.standardizedFileURL.path) else {
            return nil;
        }
        return candidateUrl;
    }
}
