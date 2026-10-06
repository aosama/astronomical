import Foundation;

import Testing;

import AstronomicalConfig;
import JourneyCategories;

@Suite(.tags(.hermeticJourney))
final class AstronomicalInstancePathsTests {
    private var temporaryDirectoryFixture: TemporaryDirectoryFixture?;

    deinit {
        if let fixture: TemporaryDirectoryFixture = self.temporaryDirectoryFixture {
            try? fixture.destroy();
        }
    }

    private func makeTemporaryDirectoryFixture() throws -> TemporaryDirectoryFixture {
        let fixture: TemporaryDirectoryFixture = try TemporaryDirectoryFixture();
        self.temporaryDirectoryFixture = fixture;
        return fixture;
    }

    @Test
    func should_keep_stable_and_development_state_and_endpoints_separate() throws -> Void {
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

        #expect(stablePaths.stateDirectory == stableHomeStateDirectory);
        #expect(developmentPaths.stateDirectory == developmentHomeStateDirectory);
        #expect(stablePaths.defaultBindAddress.description == "127.0.0.1:6732");
        #expect(developmentPaths.defaultBindAddress.description == "127.0.0.1:6733");
        #expect(stablePaths.configFilePath != developmentPaths.configFilePath);
        #expect(stablePaths.promptCacheDirectory != developmentPaths.promptCacheDirectory);
        #expect(stablePaths.modelsDirectory == stableHomeStateDirectory.appending(component: "models"));
        #expect(developmentPaths.modelsDirectory == developmentHomeStateDirectory.appending(component: "models"));
        #expect(stablePaths.modelsDirectory != developmentPaths.modelsDirectory);
        #expect(stablePaths.loggingDirectory != developmentPaths.loggingDirectory);
        #expect(stablePaths.instanceLockFilePath != developmentPaths.instanceLockFilePath);
        #expect(stablePaths.isStandardStateDirectory);
        #expect(developmentPaths.isStandardStateDirectory);
    }

    @Test
    func should_keep_every_writable_path_beneath_an_explicit_test_state_directory() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let testStateDirectory: FilePath = testStateDirectoryFixture.rootDirectoryPath;
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            testStateDirectory,
            defaultBindAddress: SocketEndpoint.loopback(port: 0)
        );
        #expect(!instancePaths.isStandardStateDirectory);
        #expect(instancePaths.modelsDirectory == testStateDirectory.appending(component: "models"));

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
            #expect(writablePath.string.hasPrefix(stateRootPrefix));
        }
    }

    @Test
    func should_assign_an_ephemeral_endpoint_to_a_custom_channel_state_directory() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        #expect(instancePaths.defaultBindAddress.description == "127.0.0.1:0");
        #expect(!instancePaths.isStandardStateDirectory);
    }

    @Test
    func should_canonicalize_a_valid_user_home_before_deriving_standard_state() throws -> Void {
        let userHomeDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let userHomeDirectory: FilePath = userHomeDirectoryFixture.rootDirectoryPath;

        let developmentPaths: AstronomicalInstancePaths = try AstronomicalInstancePaths.forUserHomeDirectory(
            userHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        let canonicalHomeDirectory: FilePath = try TestPaths.canonicalized(userHomeDirectory);
        #expect(developmentPaths.stateDirectory == canonicalHomeDirectory.appending(component: ".astronomical-dev"));
    }

    @Test
    func should_reject_a_relative_user_home_before_deriving_standard_state() throws -> Void {
        let relativeHomeDirectory: FilePath = try TestPaths.fromLiteral("relative-home");

        do {
            _ = try AstronomicalInstancePaths.forUserHomeDirectory(
                relativeHomeDirectory,
                runtimeInstance: AstronomicalRuntimeInstance.stable
            );
            Issue.record("relative HOME must be rejected");
        } catch let configError as AstronomicalConfigError {
            guard case let .pathMustBeAbsolute(fieldName, configuredPath) = configError else {
                Issue.record(Comment(stringLiteral: "expected pathMustBeAbsolute, got \(configError)"));
                return;
            }
            #expect(fieldName == "HOME");
            #expect(configuredPath == relativeHomeDirectory);
        }
    }

    @Test
    func should_reject_the_filesystem_root_as_a_user_home() throws -> Void {
        let rootHomeDirectory: FilePath = try TestPaths.fromLiteral("/");

        do {
            _ = try AstronomicalInstancePaths.forUserHomeDirectory(
                rootHomeDirectory,
                runtimeInstance: AstronomicalRuntimeInstance.stable
            );
            Issue.record("filesystem root must not become Astronomical user state");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.homeDirectoryMustNotBeRoot = configError else {
                Issue.record(Comment(stringLiteral: "expected homeDirectoryMustNotBeRoot, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_reject_a_nonexistent_user_home_before_deriving_standard_state() throws -> Void {
        let nonexistentHomeDirectory: FilePath = try TestPaths.fromLiteral("/nonexistent-astronomical-home-\(UUID().uuidString)");

        do {
            _ = try AstronomicalInstancePaths.forUserHomeDirectory(
                nonexistentHomeDirectory,
                runtimeInstance: AstronomicalRuntimeInstance.stable
            );
            Issue.record("nonexistent HOME must be rejected");
        } catch let configError as AstronomicalConfigError {
            guard case let .resolveHomeDirectory(homeDirectory, _) = configError else {
                Issue.record(Comment(stringLiteral: "expected resolveHomeDirectory, got \(configError)"));
                return;
            }
            #expect(homeDirectory == nonexistentHomeDirectory);
        }
    }

    @Test
    func should_resolve_thinking_markdown_under_the_instance_state_directory() throws -> Void {
        let fictionalHomeDirectory: FilePath = try TestPaths.fromLiteral("/Users/example");
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );

        #expect(
            developmentPaths.qwenThinkingChannelSeedFilePath
                == fictionalHomeDirectory.appending(component: ".astronomical-dev").appending(component: "thinking.md"));
        #expect(
            stablePaths.qwenThinkingChannelSeedFilePath
                == fictionalHomeDirectory.appending(component: ".astronomical").appending(component: "thinking.md"));
    }

    @Test
    func should_resolve_app_store_state_beneath_the_application_support_directory() throws -> Void {
        let fictionalApplicationSupportDirectory: FilePath =
            try TestPaths.fromLiteral("/Users/example/Library/Application Support");

        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forApplicationSupportDirectory(
            fictionalApplicationSupportDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );
        let stableStateDirectory: FilePath = fictionalApplicationSupportDirectory.appending(component: "Astronomical");

        #expect(stablePaths.stateDirectory == stableStateDirectory);
        #expect(stablePaths.modelsDirectory == stableStateDirectory.appending(component: "models"));
        #expect(stablePaths.promptCacheDirectory == stableStateDirectory.appending(component: "cache"));
        #expect(stablePaths.loggingDirectory == stableStateDirectory.appending(component: "logs"));
        #expect(stablePaths.configFilePath == stableStateDirectory.appending(component: "config.json"));
        // Standard-instance endpoint guards must carry over so the store
        // build keeps the same loopback discipline as the direct channel.
        #expect(stablePaths.isStandardStateDirectory);
        #expect(stablePaths.defaultBindAddress.description == "127.0.0.1:6732");
    }

    @Test
    func should_keep_app_store_stable_and_development_state_separate() throws -> Void {
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

        #expect(
            stablePaths.stateDirectory
                == fictionalApplicationSupportDirectory.appending(component: "Astronomical"));
        #expect(
            developmentPaths.stateDirectory
                == fictionalApplicationSupportDirectory.appending(component: "Astronomical Development"));
        #expect(stablePaths.stateDirectory != developmentPaths.stateDirectory);
        #expect(stablePaths.defaultBindAddress != developmentPaths.defaultBindAddress);
    }

    @Test
    func should_never_share_app_store_state_with_the_home_dot_folder_channel() throws -> Void {
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

        #expect(directPaths.stateDirectory != appStorePaths.stateDirectory);
    }

    @Test
    func should_expose_the_daemon_ipc_socket_path_beside_the_instance_lock_for_every_instance() throws -> Void {
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

        #expect(
            stablePaths.ipcSocketFilePath
                == fictionalHomeDirectory.appending(component: ".astronomical").appending(component: "ipc.sock"));
        #expect(
            developmentPaths.ipcSocketFilePath
                == fictionalHomeDirectory.appending(component: ".astronomical-dev").appending(component: "ipc.sock"));
        #expect(
            explicitTestPaths.ipcSocketFilePath
                == fictionalTestStateDirectory.appending(component: "ipc.sock"));
        #expect(stablePaths.ipcSocketFilePath != developmentPaths.ipcSocketFilePath);
        #expect(stablePaths.ipcSocketFilePath != stablePaths.instanceLockFilePath);
    }

    @Test
    func should_reject_a_foreign_bind_address_for_a_standard_state_directory() throws -> Void {
        let fictionalHomeDirectory: FilePath = try TestPaths.fromLiteral("/Users/example");
        let stablePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectory,
            runtimeInstance: AstronomicalRuntimeInstance.stable
        );
        let foreignEndpoint: SocketEndpoint = SocketEndpoint.loopback(port: 6733);

        do {
            _ = try stablePaths.validateConfiguredBindAddress(foreignEndpoint);
            Issue.record("standard instances must keep their own loopback endpoint");
        } catch let configError as AstronomicalConfigError {
            guard case let .standardInstanceBindAddressMismatch(
                configuredBindAddress,
                expectedBindAddress) = configError else {
                Issue.record(Comment(stringLiteral: "expected standardInstanceBindAddressMismatch, got \(configError)"));
                return;
            }
            #expect(configuredBindAddress == foreignEndpoint);
            #expect(expectedBindAddress.description == "127.0.0.1:6732");
        }
    }

    @Test
    func should_permit_any_loopback_bind_address_for_an_explicit_test_state_directory() throws -> Void {
        let fictionalTestStateDirectory: FilePath = try TestPaths.fromLiteral("/tmp/astronomical-test-instance");
        let explicitPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            fictionalTestStateDirectory,
            defaultBindAddress: SocketEndpoint.loopback(port: 0)
        );
        let customEndpoint: SocketEndpoint = SocketEndpoint(host: "127.0.0.1", port: 9999);

        let acceptedEndpoint: SocketEndpoint = try explicitPaths.validateConfiguredBindAddress(customEndpoint);

        #expect(acceptedEndpoint == customEndpoint);
    }
}
