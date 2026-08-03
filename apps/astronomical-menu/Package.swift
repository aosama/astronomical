// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "AstronomicalMenu",
    platforms: [.macOS(.v26)],
    products: [.executable(name: "astronomical-menu", targets: ["AstronomicalMenu"])],
    targets: [
        .executableTarget(name: "AstronomicalMenu"),
        .testTarget(name: "AstronomicalMenuContractTests", dependencies: ["AstronomicalMenu"]),
    ]
)
