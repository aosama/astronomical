// swift-tools-version: 6.0
import PackageDescription

// A dedicated thin chat application. The pure, fully-tested logic lives in the
// ThinTalkCore target (no SwiftUI, no AppKit); the executable target is a thin
// SwiftUI layer that only renders state and forwards user actions.
//
// ThinTalkCanvas owns the conversation canvas: one WKWebView plus the offline
// shell it renders with. It is a separate target so the canvas can be driven
// directly in tests with a real web view and the real bundled assets.
//
// Tests in ThinTalkContractTests exercise the REST chat contract against a
// stubbed wire, and the canvas against the real shell, so a live supervisor is
// never required to verify behaviour.
let package = Package(
    name: "ThinTalk",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "ThinTalk", targets: ["ThinTalk"]),
        // The chat surface is also linked by astronomical-menu, which opens the
        // conversation as one window of the one Astronomical app (issue 647
        // decision revision, 2026-09-20). One module set, two hosts.
        .library(name: "ThinTalkChat", targets: ["ThinTalkCore", "ThinTalkCanvas", "ThinTalkUI"]),
    ],
    targets: [
        .target(name: "ThinTalkCore", path: "Sources/ThinTalkCore"),
        .target(
            name: "ThinTalkCanvas",
            dependencies: ["ThinTalkCore"],
            path: "Sources/ThinTalkCanvas",
            resources: [.copy("Resources/web")]
        ),
        .target(
            name: "ThinTalkUI",
            dependencies: ["ThinTalkCore", "ThinTalkCanvas"],
            path: "Sources/ThinTalkUI"
        ),
        .executableTarget(
            name: "ThinTalk",
            dependencies: ["ThinTalkUI"],
            path: "Sources/ThinTalk"
        ),
        .testTarget(
            name: "ThinTalkContractTests",
            dependencies: ["ThinTalkCore", "ThinTalkCanvas", "ThinTalkUI"],
            path: "Tests/ThinTalkContractTests"
        ),
    ]
)