import Foundation;
import XCTest;
import AstronomicalConfig;

final class AstronomicalInstancePathsTests: XCTestCase {
    private var temporaryDirectoryFixture: TemporaryDirectoryFixture?;

    override func tearDown() {
        guard let fixture: TemporaryDirectoryFixture = self.temporaryDirectoryFixture else {
            super.tearDown();
            return;
        }
        do {
            try fixture.destroy();
        } catch {
            XCTFail("temporary directory should be removed: \(error)");
        }
        self.temporaryDirectoryFixture = nil;
        super.tearDown();
    }

    private func makeTemporaryDirectoryFixture() throws -> TemporaryDirectoryFixture {
        let fixture: TemporaryDirectoryFixture = try TemporaryDirectoryFixture();
        self.temporaryDirectoryFixture = fixture;
        return fixture;
    }

    func testShouldKeepStableAndDevelopmentStateAndEndpointsSeparate() throws -> Void {
        let fictionalHomeDirectory: FilePath = try TestPaths.fromLiteral("/Users/example");
        let stableHomeStateDirectory: FilePath = fictionalHomeDirectory.appending(component: ".astronomical");
        let developmentHomeStateDirectory: FilePath = fictionalHomeDirectory.appending(component: ".astronomical-dev");

        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        XCTAssertEqual(stablePaths.stateDirectory, stableHomeStateDirectory);
        XCTAssertEqual(developmentPaths.stateDirectory, developmentHomeStateDirectory);
        XCTAssertEqual(stablePaths.defaultBindAddress.description, "127.0.0.1:6732");
        XCTAssertEqual(developmentPaths.defaultBindAddress.description, "127.0.0.1:6733");
        XCTAssertNotEqual(stablePaths.configFilePath, developmentPaths.configFilePath);
        XCTAssertNotEqual(stablePaths.promptCacheDirectory, developmentPaths.promptCacheDirectory);
        XCTAssertEqual(stablePaths.modelsDirectory, stableHomeStateDirectory.appending(component: "models"));
        XCTAssertEqual(developmentPaths.modelsDirectory, developmentHomeStateDirectory.appending(component: "models"));
        XCTAssertNotEqual(stablePaths.modelsDirectory, developmentPaths.modelsDirectory);
        XCTAssertNotEqual(stablePaths.loggingDirectory, developmentPaths.loggingDirectory);
        XCTAssertNotEqual(stablePaths.instanceLockFilePath, developmentPaths.instanceLockFilePath);
        XCTAssertTrue(stablePaths.isStandardStateDirectory);
        XCTAssertTrue(developmentPaths.isStandardStateDirectory);
    }

    func testShouldKeepEveryWritablePathBeneathAnExplicitTestStateDirectory() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let testStateDirectory: FilePath = testStateDirectoryFixture.rootDirectoryPath;
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            testStateDirectory,
            defaultBindAddress: SocketEndpoint.loopback(port: 0)
        );
        XCTAssertFalse(instancePaths.isStandardStateDirectory);
        XCTAssertEqual(instancePaths.modelsDirectory, testStateDirectory.appending(component: "models"));

        let writablePaths: Array<FilePath> = [
            instancePaths.configFilePath,
            instancePaths.promptCacheDirectory,
            instancePaths.modelsDirectory,
            instancePaths.loggingDirectory,
            instancePaths.daemonOwnershipFilePath,
            instancePaths.instanceLockFilePath,
            instancePaths.qwenThinkingChannelSeedFilePath
        ];
        let stateRootPrefix: String = testStateDirectory.string + "/";
        for writablePath: FilePath in writablePaths {
            XCTAssertTrue(writablePath.string.hasPrefix(stateRootPrefix));
        }
    }

    func testShouldAssignAnEphemeralEndpointToACustomChannelStateDirectory() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        XCTAssertEqual(instancePaths.defaultBindAddress.description, "127.0.0.1:0");
        XCTAssertFalse(instancePaths.isStandardStateDirectory);
    }

    func testShouldCanonicalizeAValidUserHomeBeforeDerivingStandardState() throws -> Void {
        let userHomeDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let userHomeDirectory: FilePath = userHomeDirectoryFixture.rootDirectoryPath;

        let developmentPaths: AstronomicalInstancePaths = try AstronomicalInstancePaths.forUserHomeDirectory(
            userHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        let canonicalHomeDirectory: FilePath = try TestPaths.canonicalized(userHomeDirectory);
        XCTAssertEqual(developmentPaths.stateDirectory, canonicalHomeDirectory.appending(component: ".astronomical-dev"));
    }

    func testShouldRejectARelativeUserHomeBeforeDerivingStandardState() throws -> Void {
        let relativeHomeDirectory: FilePath = try TestPaths.fromLiteral("relative-home");

        XCTAssertThrowsError(
            try AstronomicalInstancePaths.forUserHomeDirectory(
                relativeHomeDirectory,
                runtimeInstance: AstronomicalRuntimeInstance.stable
            ),
            "relative HOME must be rejected",
            { (caughtError: any Error) in
                guard let configError: AstronomicalConfigError = caughtError as? AstronomicalConfigError else {
                    XCTFail("expected AstronomicalConfigError, got \(caughtError)");
                    return;
                }
                guard case let AstronomicalConfigError.pathMustBeAbsolute(fieldName, configuredPath) = configError else {
                    XCTFail("expected pathMustBeAbsolute, got \(configError)");
                    return;
                }
                XCTAssertEqual(fieldName, "HOME");
                XCTAssertEqual(configuredPath, relativeHomeDirectory);
            }
        );
    }

    func testShouldRejectTheFilesystemRootAsAUserHome() throws -> Void {
        let rootHomeDirectory: FilePath = try TestPaths.fromLiteral("/");

        XCTAssertThrowsError(
            try AstronomicalInstancePaths.forUserHomeDirectory(
                rootHomeDirectory,
                runtimeInstance: AstronomicalRuntimeInstance.stable
            ),
            "filesystem root must not become Astronomical user state",
            { (caughtError: any Error) in
                guard let configError: AstronomicalConfigError = caughtError as? AstronomicalConfigError else {
                    XCTFail("expected AstronomicalConfigError, got \(caughtError)");
                    return;
                }
                guard case AstronomicalConfigError.homeDirectoryMustNotBeRoot = configError else {
                    XCTFail("expected homeDirectoryMustNotBeRoot, got \(configError)");
                    return;
                }
            }
        );
    }

    func testShouldRejectANonexistentUserHomeBeforeDerivingStandardState() throws -> Void {
        let nonexistentHomeDirectory: FilePath = try TestPaths.fromLiteral("/nonexistent-astronomical-home-\(UUID().uuidString)");

        XCTAssertThrowsError(
            try AstronomicalInstancePaths.forUserHomeDirectory(
                nonexistentHomeDirectory,
                runtimeInstance: AstronomicalRuntimeInstance.stable
            ),
            "nonexistent HOME must be rejected",
            { (caughtError: any Error) in
                guard let configError: AstronomicalConfigError = caughtError as? AstronomicalConfigError else {
                    XCTFail("expected AstronomicalConfigError, got \(caughtError)");
                    return;
                }
                guard case let AstronomicalConfigError.resolveHomeDirectory(homeDirectory, _) = configError else {
                    XCTFail("expected resolveHomeDirectory, got \(configError)");
                    return;
                }
                XCTAssertEqual(homeDirectory, nonexistentHomeDirectory);
            }
        );
    }

    func testShouldResolveThinkingMarkdownUnderTheInstanceStateDirectory() throws -> Void {
        let fictionalHomeDirectory: FilePath = try TestPaths.fromLiteral("/Users/example");
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );

        XCTAssertEqual(
            developmentPaths.qwenThinkingChannelSeedFilePath,
            fictionalHomeDirectory.appending(component: ".astronomical-dev").appending(component: "thinking.md")
        );
        XCTAssertEqual(
            stablePaths.qwenThinkingChannelSeedFilePath,
            fictionalHomeDirectory.appending(component: ".astronomical").appending(component: "thinking.md")
        );
    }

    func testShouldResolveAppStoreStateBeneathTheApplicationSupportDirectory() throws -> Void {
        let fictionalApplicationSupportDirectory: FilePath =
            try TestPaths.fromLiteral("/Users/example/Library/Application Support");

        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forApplicationSupportDirectory(
            fictionalApplicationSupportDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );
        let stableStateDirectory: FilePath = fictionalApplicationSupportDirectory.appending(component: "Astronomical");

        XCTAssertEqual(stablePaths.stateDirectory, stableStateDirectory);
        XCTAssertEqual(stablePaths.modelsDirectory, stableStateDirectory.appending(component: "models"));
        XCTAssertEqual(stablePaths.promptCacheDirectory, stableStateDirectory.appending(component: "cache"));
        XCTAssertEqual(stablePaths.loggingDirectory, stableStateDirectory.appending(component: "logs"));
        XCTAssertEqual(stablePaths.configFilePath, stableStateDirectory.appending(component: "config.json"));
        // Standard-instance endpoint guards must carry over so the store
        // build keeps the same loopback discipline as the direct channel.
        XCTAssertTrue(stablePaths.isStandardStateDirectory);
        XCTAssertEqual(stablePaths.defaultBindAddress.description, "127.0.0.1:6732");
    }

    func testShouldKeepAppStoreStableAndDevelopmentStateSeparate() throws -> Void {
        let fictionalApplicationSupportDirectory: FilePath =
            try TestPaths.fromLiteral("/Users/example/Library/Application Support");

        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forApplicationSupportDirectory(
            fictionalApplicationSupportDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forApplicationSupportDirectory(
            fictionalApplicationSupportDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        XCTAssertEqual(
            stablePaths.stateDirectory,
            fictionalApplicationSupportDirectory.appending(component: "Astronomical")
        );
        XCTAssertEqual(
            developmentPaths.stateDirectory,
            fictionalApplicationSupportDirectory.appending(component: "Astronomical Development")
        );
        XCTAssertNotEqual(stablePaths.stateDirectory, developmentPaths.stateDirectory);
        XCTAssertNotEqual(stablePaths.defaultBindAddress, developmentPaths.defaultBindAddress);
    }

    func testShouldNeverShareAppStoreStateWithTheHomeDotFolderChannel() throws -> Void {
        let fictionalHomeDirectory: FilePath = try TestPaths.fromLiteral("/Users/example");
        let fictionalApplicationSupportDirectory: FilePath =
            try TestPaths.fromLiteral("/Users/example/Library/Application Support");

        let directPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );
        let appStorePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forApplicationSupportDirectory(
            fictionalApplicationSupportDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );

        XCTAssertNotEqual(directPaths.stateDirectory, appStorePaths.stateDirectory);
    }

    func testShouldExposeDaemonIpcSocketPathBesideTheInstanceLockForEveryInstance() throws -> Void {
        let fictionalHomeDirectory: FilePath = try TestPaths.fromLiteral("/Users/example");
        let fictionalTestStateDirectory: FilePath = try TestPaths.fromLiteral("/tmp/astronomical-test-instance");

        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        let explicitTestPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            fictionalTestStateDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        XCTAssertEqual(
            stablePaths.ipcSocketFilePath,
            fictionalHomeDirectory.appending(component: ".astronomical").appending(component: "ipc.sock")
        );
        XCTAssertEqual(
            developmentPaths.ipcSocketFilePath,
            fictionalHomeDirectory.appending(component: ".astronomical-dev").appending(component: "ipc.sock")
        );
        XCTAssertEqual(
            explicitTestPaths.ipcSocketFilePath,
            fictionalTestStateDirectory.appending(component: "ipc.sock")
        );
        XCTAssertNotEqual(stablePaths.ipcSocketFilePath, developmentPaths.ipcSocketFilePath);
        XCTAssertNotEqual(stablePaths.ipcSocketFilePath, stablePaths.instanceLockFilePath);
    }

    func testShouldRejectAForeignBindAddressForAStandardStateDirectory() throws -> Void {
        let fictionalHomeDirectory: FilePath = try TestPaths.fromLiteral("/Users/example");
        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );
        let foreignEndpoint: SocketEndpoint = SocketEndpoint.loopback(port: 6733);

        XCTAssertThrowsError(
            try stablePaths.validateConfiguredBindAddress(foreignEndpoint),
            "standard instances must keep their own loopback endpoint",
            { (caughtError: any Error) in
                guard let configError: AstronomicalConfigError = caughtError as? AstronomicalConfigError else {
                    XCTFail("expected AstronomicalConfigError, got \(caughtError)");
                    return;
                }
                guard case let AstronomicalConfigError.standardInstanceBindAddressMismatch(
                    configuredBindAddress,
                    expectedBindAddress
                ) = configError else {
                    XCTFail("expected standardInstanceBindAddressMismatch, got \(configError)");
                    return;
                }
                XCTAssertEqual(configuredBindAddress, foreignEndpoint);
                XCTAssertEqual(expectedBindAddress.description, "127.0.0.1:6732");
            }
        );
    }

    func testShouldPermitAnyLoopbackBindAddressForAnExplicitTestStateDirectory() throws -> Void {
        let fictionalTestStateDirectory: FilePath = try TestPaths.fromLiteral("/tmp/astronomical-test-instance");
        let explicitPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            fictionalTestStateDirectory,
            defaultBindAddress: SocketEndpoint.loopback(port: 0)
        );
        let customEndpoint: SocketEndpoint = SocketEndpoint(host: "127.0.0.1", port: 9999);

        let acceptedEndpoint: SocketEndpoint = try explicitPaths.validateConfiguredBindAddress(customEndpoint);

        XCTAssertEqual(acceptedEndpoint, customEndpoint);
    }
}
