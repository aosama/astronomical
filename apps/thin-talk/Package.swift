// swift-tools-version: 6.0
import PackageDescription

// A dedicated thin chat application. The pure, fully-tested logic lives in the
// ThinTalkCore target (no SwiftUI, no AppKit); the executable target is a thin
// SwiftUI layer that only renders state and forwards user actions. Tests in
// ThinTalkContractTests exercise the REST chat contract against a stubbed wire
// so a live supervisor is never required to verify behaviour.
let package = Package(
    name: "ThinTalk",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "ThinTalk", targets: ["ThinTalk"]),
    ],
    targets: [
        .target(name: "ThinTalkCore", path: "Sources/ThinTalkCore"),
        .executableTarget(
            name: "ThinTalk",
            dependencies: ["ThinTalkCore"],
            path: "Sources/ThinTalk"
        ),
        .testTarget(
            name: "ThinTalkContractTests",
            dependencies: ["ThinTalkCore"],
            path: "Tests/ThinTalkContractTests"
        ),
    ]
)
