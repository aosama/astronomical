// swift-tools-version: 6.2
import PackageDescription

// One menu application source tree, two executables: the direct channel links
// Sparkle; the App Store channel links only the core, whose default update
// controller carries store semantics. This is the SwiftPM equivalent of the
// dual-target pattern required by App Review guideline 2.4.5(vii).
let package = Package(
    name: "AstronomicalMenu",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "astronomical-menu", targets: ["AstronomicalMenu"]),
        .executable(name: "astronomical-menu-app-store", targets: ["AstronomicalMenuAppStore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/sparkle-project/Sparkle.git", from: "2.9.6"),
    ],
    targets: [
        .target(name: "AstronomicalMenuCore", path: "Sources/AstronomicalMenuCore"),
        .target(
            name: "AstronomicalMenuSparkleUpdateController",
            dependencies: [
                "AstronomicalMenuCore",
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/AstronomicalMenuSparkleUpdateController"
        ),
        .executableTarget(
            name: "AstronomicalMenu",
            dependencies: ["AstronomicalMenuCore", "AstronomicalMenuSparkleUpdateController"],
            path: "Sources/AstronomicalMenu"
        ),
        .executableTarget(
            name: "AstronomicalMenuAppStore",
            dependencies: ["AstronomicalMenuCore"],
            path: "Sources/AstronomicalMenuAppStore"
        ),
        .testTarget(
            name: "AstronomicalMenuContractTests",
            dependencies: ["AstronomicalMenuCore", "AstronomicalMenuSparkleUpdateController"],
            resources: [.process("Fixtures")]
        ),
    ]
)
