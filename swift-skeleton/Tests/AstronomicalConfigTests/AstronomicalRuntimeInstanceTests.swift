import XCTest;
import AstronomicalConfig;

final class AstronomicalRuntimeInstanceTests: XCTestCase {
    func testShouldParseStableAndDevelopmentRawInstanceValues() throws -> Void {
        let stableInstance: AstronomicalRuntimeInstance = try AstronomicalRuntimeInstance(rawInstance: "stable");
        let developmentInstance: AstronomicalRuntimeInstance = try AstronomicalRuntimeInstance(rawInstance: "development");

        XCTAssertEqual(stableInstance.rawValue, "stable");
        XCTAssertEqual(developmentInstance.rawValue, "development");
        XCTAssertEqual(stableInstance.displayName, "Stable");
        XCTAssertEqual(developmentInstance.displayName, "Development");
    }

    func testShouldRejectAnUnknownRuntimeInstanceRawValue() throws -> Void {
        XCTAssertThrowsError(
            try AstronomicalRuntimeInstance(rawInstance: "nightly"),
            "unknown raw instance must be rejected",
            { (caughtError: any Error) in
                guard let configError: AstronomicalConfigError = caughtError as? AstronomicalConfigError else {
                    XCTFail("expected AstronomicalConfigError, got \(caughtError)");
                    return;
                }
                guard case let AstronomicalConfigError.invalidRuntimeInstance(rawInstance) = configError else {
                    XCTFail("expected invalidRuntimeInstance, got \(configError)");
                    return;
                }
                XCTAssertEqual(rawInstance, "nightly");
            }
        );
    }
}
