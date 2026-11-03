// Proves the app icon renderer draws a clean, deterministic Stable icon,
// a visibly distinct Development icon, a complete icon family iconutil
// accepts, and refuses unsupported or duplicated options.
//
// Swift rather than shell or Rust: the renderer is a Swift script and the
// contract runs on the CI job that already owns the Swift toolchain, so it
// adds no toolchain install or compile to any job.
//
// Usage: swift scripts/internal/render-macos-app-icon-contract.swift

import Foundation

let requiredIconNames = [
    "icon_16x16.png", "icon_16x16@2x.png",
    "icon_32x32.png", "icon_32x32@2x.png",
    "icon_128x128.png", "icon_128x128@2x.png",
    "icon_256x256.png", "icon_256x256@2x.png",
    "icon_512x512.png", "icon_512x512@2x.png",
]

struct ContractFailure: Error, CustomStringConvertible {
    let description: String
}

func runTool(_ executablePath: String, _ toolArguments: [String]) throws -> Int32 {
    let toolProcess = Process()
    toolProcess.executableURL = URL(fileURLWithPath: executablePath)
    toolProcess.arguments = toolArguments
    toolProcess.standardOutput = FileHandle.nullDevice
    toolProcess.standardError = FileHandle.nullDevice
    try toolProcess.run()
    toolProcess.waitUntilExit()
    return toolProcess.terminationStatus
}

/// The renderer is compiled once and then run directly; running it six
/// times through `swift script` would recompile it six times.
func compileRenderer(_ rendererSource: String, into rendererBinary: String) throws {
    guard try runTool("/usr/bin/xcrun", ["swiftc", "-O", rendererSource, "-o", rendererBinary]) == 0 else {
        throw ContractFailure(description: "the icon renderer did not compile")
    }
}

func render(_ rendererBinary: String, channel: String, into iconsetDirectory: String) throws {
    let renderStatus = try runTool(rendererBinary, ["--output-directory", iconsetDirectory, "--channel", channel])
    guard renderStatus == 0 else {
        throw ContractFailure(description: "the renderer failed for the \(channel) channel")
    }
}

func fileContentsAreEqual(_ firstPath: String, _ secondPath: String) -> Bool {
    FileManager.default.contentsEqual(atPath: firstPath, andPath: secondPath)
}

func isNonEmptyFile(_ filePath: String) -> Bool {
    guard let fileAttributes = try? FileManager.default.attributesOfItem(atPath: filePath),
        let fileSize = fileAttributes[.size] as? NSNumber
    else {
        return false
    }
    return fileSize.intValue > 0
}

func checkIconFamilies(stableA: String, stableB: String, developmentA: String, developmentB: String) throws {
    for iconName in requiredIconNames {
        let stableIcon = "\(stableA)/\(iconName)"
        guard isNonEmptyFile(stableIcon) else {
            throw ContractFailure(description: "rendered icon family is missing \(iconName)")
        }
        guard fileContentsAreEqual(stableIcon, "\(stableB)/\(iconName)") else {
            throw ContractFailure(description: "Stable icon is not deterministic: \(iconName)")
        }
        guard fileContentsAreEqual("\(developmentA)/\(iconName)", "\(developmentB)/\(iconName)") else {
            throw ContractFailure(description: "Development icon is not deterministic: \(iconName)")
        }
        guard !fileContentsAreEqual(stableIcon, "\(developmentA)/\(iconName)") else {
            throw ContractFailure(description: "Development identity is absent from \(iconName)")
        }
    }
}

func checkDeterministicChannels(rendererBinary: String, sandboxDirectory: String) throws {
    print("[app-icon-test] case=stable-icon-is-clean-and-deterministic status=start")
    let stableA = "\(sandboxDirectory)/Stable-A.iconset"
    let stableB = "\(sandboxDirectory)/Stable-B.iconset"
    let developmentA = "\(sandboxDirectory)/Development-A.iconset"
    let developmentB = "\(sandboxDirectory)/Development-B.iconset"
    try render(rendererBinary, channel: "stable", into: stableA)
    try render(rendererBinary, channel: "stable", into: stableB)
    try render(rendererBinary, channel: "development", into: developmentA)
    try render(rendererBinary, channel: "development", into: developmentB)
    try checkIconFamilies(stableA: stableA, stableB: stableB, developmentA: developmentA, developmentB: developmentB)
    let icnsPath = "\(sandboxDirectory)/Astronomical.icns"
    let iconutilStatus = try runTool("/usr/bin/iconutil", ["--convert", "icns", "--output", icnsPath, stableA])
    guard iconutilStatus == 0, isNonEmptyFile(icnsPath) else {
        throw ContractFailure(description: "iconutil did not produce Astronomical.icns")
    }
    print("[app-icon-test] case=stable-icon-is-clean-and-deterministic status=success")
}

func checkRejectedOptions(rendererBinary: String, sandboxDirectory: String) throws {
    print("[app-icon-test] case=invalid-channel-is-rejected status=start")
    let invalidArguments = ["--output-directory", "\(sandboxDirectory)/invalid.iconset", "--channel", "nightly"]
    guard try runTool(rendererBinary, invalidArguments) != 0 else {
        throw ContractFailure(description: "icon renderer unexpectedly accepted an unsupported channel")
    }
    print("[app-icon-test] case=invalid-channel-is-rejected status=success")
    print("[app-icon-test] case=duplicate-options-are-rejected status=start")
    let duplicateArguments = [
        "--output-directory", "\(sandboxDirectory)/duplicate.iconset", "--channel", "stable", "--channel", "development",
    ]
    guard try runTool(rendererBinary, duplicateArguments) != 0 else {
        throw ContractFailure(description: "icon renderer unexpectedly accepted a duplicate channel")
    }
    print("[app-icon-test] case=duplicate-options-are-rejected status=success")
}

let contractSource = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
let rendererSource = contractSource.deletingLastPathComponent().appendingPathComponent("render-macos-app-icon.swift").path
let sandboxDirectory = FileManager.default.temporaryDirectory
    .appendingPathComponent("astronomical-app-icon-\(ProcessInfo.processInfo.processIdentifier)").path
do {
    try FileManager.default.createDirectory(atPath: sandboxDirectory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(atPath: sandboxDirectory) }
    let rendererBinary = "\(sandboxDirectory)/render-macos-app-icon"
    try compileRenderer(rendererSource, into: rendererBinary)
    try checkDeterministicChannels(rendererBinary: rendererBinary, sandboxDirectory: sandboxDirectory)
    try checkRejectedOptions(rendererBinary: rendererBinary, sandboxDirectory: sandboxDirectory)
} catch {
    FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
    exit(1)
}
