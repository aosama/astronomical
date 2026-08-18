// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "AstronomicalMenu",
    platforms: [.macOS(.v26)],
    products: [.executable(name: "astronomical-menu", targets: ["AstronomicalMenu"])],
    dependencies: [
        .package(url: "https://github.com/sparkle-project/Sparkle.git", from: "2.9.6"),
    ],
    targets: [
        .executableTarget(
            name: "AstronomicalMenu",
            dependencies: [.product(name: "Sparkle", package: "Sparkle")]
        ),
        .testTarget(name: "AstronomicalMenuContractTests", dependencies: ["AstronomicalMenu"]),
    ]
)
