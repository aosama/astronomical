import Foundation

import Testing;

import AstronomicalCli;
import AstronomicalConfig;
import JourneyCategories;

@testable import AstronomicalCli;

/// Instance-resolution journeys, porting instance.rs.
@Suite(.tags(.hermeticJourney))
final class InstanceResolutionTests {

    @Test
    func should_treat_the_stable_app_bundle_as_stable_loopback() {
        #expect(CliInstanceResolution.runtimeInstanceFromExecutablePath(
            "/Applications/Astronomical.app/Contents/MacOS/astronomical"
        ) == .stable);
    }

    @Test
    func should_treat_the_development_app_bundle_as_development_loopback() {
        #expect(CliInstanceResolution.runtimeInstanceFromExecutablePath(
            "/Applications/Astronomical Development.app/Contents/MacOS/astronomical"
        ) == .development);
    }

    @Test
    func should_treat_unpackaged_binaries_as_development() {
        #expect(CliInstanceResolution.runtimeInstanceFromExecutablePath(
            "/fictional/build/astronomical"
        ) == .development);
    }

    @Test
    func should_resolve_a_symlink_to_the_bundle_it_points_into() throws {
        let linkDirectory: String = CliJourneySupport.freshTestDirectory("instance-symlink");
        defer { try? FileManager.default.removeItem(atPath: linkDirectory) }
        let stableBundleTarget: String = linkDirectory + "/Astronomical.app/Contents/MacOS/astronomical";
        try FileManager.default.createDirectory(
            atPath: (stableBundleTarget as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true
        );
        try Data().write(to: URL(fileURLWithPath: stableBundleTarget));
        let linkPath: String = linkDirectory + "/astronomical";
        try FileManager.default.createSymbolicLink(
            atPath: linkPath,
            withDestinationPath: stableBundleTarget
        );
        #expect(CliInstanceResolution.runtimeInstanceFromExecutablePath(linkPath) == .stable);
    }

    @Test
    func should_keep_falling_back_to_the_given_path_when_it_cannot_be_canonicalized() {
        #expect(CliInstanceResolution.runtimeInstanceFromExecutablePath(
            "/fictional/does-not-exist/astronomical"
        ) == .development);
    }

    @Test
    func should_try_the_binarys_own_instance_before_the_other_channel() {
        #expect(CliInstanceResolution.candidateInstances(.development) == [.development, .stable]);
        #expect(CliInstanceResolution.candidateInstances(.stable) == [.stable, .development]);
    }

    @Test
    func should_expose_stable_and_development_loopback_ports() {
        #expect(AstronomicalRuntimeInstance.stable.loopbackEndpoint.port == 6732);
        #expect(AstronomicalRuntimeInstance.development.loopbackEndpoint.port == 6733);
    }
}
