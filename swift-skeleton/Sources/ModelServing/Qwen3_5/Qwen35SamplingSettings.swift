import Foundation;

import IpcProtocol;
import MLXLMCommon;

/// Maps bounded chat sampling settings onto the upstream logit samplers.
///
/// Mirrors the Rust gpu_token_sampling + sampling_seed pairing: one sampler
/// per request, seeded for reproducibility, built from the validated
/// thousandths-scale wire settings.
///
/// Temperature follows the repository sampling directive: an omitted setting
/// defers to the model's default (the upstream toolkit default 0.6); a zero
/// never enters a construction path we author.
public struct Qwen35SamplingSettings {

    public let temperature: Float;
    public let topP: Float;
    public let seed: UInt64?;

    public init(chatGenerationSettings: ChatGenerationSettings) {
        self.temperature = chatGenerationSettings.temperatureThousandths.map {
            temperatureThousandths in Float(temperatureThousandths) / 1000.0
        } ?? Qwen35SamplingSettings.defaultTemperature;
        self.topP = chatGenerationSettings.topPThousandths.map {
            topPThousandths in Float(topPThousandths) / 1000.0
        } ?? Qwen35SamplingSettings.defaultTopP;
        self.seed = chatGenerationSettings.seed;
    }

    /// The upstream toolkit's provider default for temperature.
    public static let defaultTemperature: Float = 0.6;
    public static let defaultTopP: Float = 1.0;

    /// Builds the one sampler this request samples every decode step with.
    public func makeSampler() -> LogitSampler {
        let usesTopP: Bool = self.topP > 0 && self.topP < 1;
        if self.temperature == 0 {
            return ArgMaxSampler();
        }
        if usesTopP {
            return TopPSampler(
                temperature: self.temperature, topP: self.topP, topK: 0, minP: 0,
                seed: self.seed);
        }
        return CategoricalSampler(temperature: self.temperature, seed: self.seed);
    }
}
