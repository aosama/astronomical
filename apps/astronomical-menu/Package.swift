// swift-tools-version: 6.2
import PackageDescription

// One package, two processes of one application bundle.
//
// The main app (`astronomical-app`) is the regular Mac application: the Thin
// Talk window, the menu bar, updates. The agent (`astronomical-menu`) is the
// menu-bar item that owns the engine; it lives in the bundle's
// Contents/Library/LoginItems and can start at login. Neither ever changes
// its role while running.
//
// The main app has two executables: the direct channel links Sparkle; the App
// Store channel links only the core, whose default update controller carries
// store semantics. This is the SwiftPM equivalent of the dual-target pattern
// required by App Review guideline 2.4.5(vii). The agent links neither Sparkle
// nor Thin Talk, so a single agent executable serves every channel.
let package = Package(
    name: "AstronomicalMenu",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "astronomical-app", targets: ["AstronomicalApp"]),
        .executable(name: "astronomical-app-app-store", targets: ["AstronomicalAppAppStore"]),
        .executable(name: "astronomical-menu", targets: ["AstronomicalMenu"]),
    ],
    dependencies: [
        .package(url: "https://github.com/sparkle-project/Sparkle.git", from: "2.9.6"),
        // The main app hosts the conversation as its window: the persistent
        // Thin Talk chat surface (issue 647 decision revision, 2026-09-20).
        .package(path: "../thin-talk"),
    ],
    targets: [
        // What both processes agree on: which instance they belong to, where
        // things sit in the bundle, and the commands they send each other.
        .target(
            name: "AstronomicalShared",
            path: "Sources/AstronomicalShared"
        ),
        .target(
            name: "AstronomicalAppCore",
            dependencies: [
                "AstronomicalShared",
                .product(name: "ThinTalkChat", package: "thin-talk"),
            ],
            path: "Sources/AstronomicalAppCore"
        ),
        .target(
            name: "AstronomicalSparkleUpdateController",
            dependencies: [
                "AstronomicalAppCore",
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/AstronomicalSparkleUpdateController"
        ),
        .executableTarget(
            name: "AstronomicalApp",
            dependencies: ["AstronomicalAppCore", "AstronomicalSparkleUpdateController"],
            path: "Sources/AstronomicalApp"
        ),
        .executableTarget(
            name: "AstronomicalAppAppStore",
            dependencies: ["AstronomicalAppCore"],
            path: "Sources/AstronomicalAppAppStore"
        ),
        .target(
            name: "AstronomicalMenuCore",
            dependencies: ["AstronomicalShared"],
            path: "Sources/AstronomicalMenuCore"
        ),
        .executableTarget(
            name: "AstronomicalMenu",
            dependencies: ["AstronomicalMenuCore"],
            path: "Sources/AstronomicalMenu"
        ),
        .testTarget(
            name: "AstronomicalMenuContractTests",
            dependencies: [
                "AstronomicalShared", "AstronomicalMenuCore", "AstronomicalAppCore",
                "AstronomicalSparkleUpdateController",
            ],
            resources: [.process("Fixtures")]
        ),
    ]
)
