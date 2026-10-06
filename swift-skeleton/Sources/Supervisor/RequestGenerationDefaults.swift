import Foundation;

import IpcProtocol;

/// Tracks which public settings were supplied so model policy changes only
/// omissions, mirroring apps/supervisor/src/request_generation_defaults.rs.
struct RequestGenerationSettingsPresence {

    let maximumOutputTokens: Bool;
    let temperature: Bool;
    let topP: Bool;

    init(generationSettings: ChatGenerationSettings) {
        self.maximumOutputTokens = generationSettings.maxOutputTokens != 0;
        self.temperature = generationSettings.temperatureThousandths != nil;
        self.topP = generationSettings.topPThousandths != nil;
    }

    /// The REST surface knows client-supplied settings before translation,
    /// so it records presence directly from the public request parts.
    init(
        maximumOutputTokensRequested: Bool,
        temperatureRequested: Bool,
        topPRequested: Bool
    ) {
        self.maximumOutputTokens = maximumOutputTokensRequested;
        self.temperature = temperatureRequested;
        self.topP = topPRequested;
    }
}

/// Applies one canonical model's live request defaults without overriding
/// client values.
enum RequestGenerationDefaults {

    static func apply(
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        modelId: String,
        settingsPresence: RequestGenerationSettingsPresence,
        generationSettings: ChatGenerationSettings
    ) -> ChatGenerationSettings {
        guard let modelPolicy: RuntimeModelPolicy = resolvedRuntimeConfig.modelPolicyCatalog[modelId] else {
            return generationSettings;
        }
        var maximumOutputTokens: UInt16 = generationSettings.maxOutputTokens;
        var temperatureThousandths: UInt16? = generationSettings.temperatureThousandths;
        var topPThousandths: UInt16? = generationSettings.topPThousandths;
        if !settingsPresence.maximumOutputTokens {
            maximumOutputTokens = modelPolicy.generationDefaults.maximumOutputTokens;
        }
        if !settingsPresence.temperature {
            temperatureThousandths = modelPolicy.generationDefaults.temperatureThousandths;
        }
        if !settingsPresence.topP {
            topPThousandths = modelPolicy.generationDefaults.topPThousandths;
        }
        return ChatGenerationSettings(
            maxOutputTokens: maximumOutputTokens,
            temperatureThousandths: temperatureThousandths,
            topPThousandths: topPThousandths,
            seed: generationSettings.seed,
            thinkingBudget: generationSettings.thinkingBudget);
    }
}
