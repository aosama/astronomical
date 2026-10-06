import Testing;

import AstronomicalConfig;
import JourneyCategories;

@Suite(.tags(.hermeticJourney))
final class AstronomicalRuntimeInstanceTests {

    @Test
    func should_parse_stable_and_development_raw_instance_values() throws -> Void {
        let stableInstance: AstronomicalRuntimeInstance = try AstronomicalRuntimeInstance(rawInstance: "stable");
        let developmentInstance: AstronomicalRuntimeInstance = try AstronomicalRuntimeInstance(rawInstance: "development");

        #expect(stableInstance.rawValue == "stable");
        #expect(developmentInstance.rawValue == "development");
        #expect(stableInstance.displayName == "Stable");
        #expect(developmentInstance.displayName == "Development");
    }

    @Test
    func should_reject_an_unknown_runtime_instance_raw_value() throws -> Void {
        do {
            _ = try AstronomicalRuntimeInstance(rawInstance: "nightly");
            Issue.record("unknown raw instance must be rejected");
        } catch let configError as AstronomicalConfigError {
            guard case let .invalidRuntimeInstance(rawInstance) = configError else {
                Issue.record(Comment(stringLiteral: "expected invalidRuntimeInstance, got \(configError)"));
                return;
            }
            #expect(rawInstance == "nightly");
        }
    }
}
