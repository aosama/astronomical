import Foundation;

/// Synthesizes a complete tiny tokenizer on disk: a minimal but valid BPE
/// tokenizer whose vocabulary chains the Romeo and Juliet fixture words from
/// their characters, plus the Qwen control markers, and a chat template in
/// the tokenizer config. Every id stays inside the tiny dense model's
/// 512-entry vocabulary so the same ids run through the real engine
/// forward pass.
public enum TinyTokenizerFixture {

    public static let endOfSequenceTokenId: Int = 3;
    public static let thinkOpenTokenId: Int = 12;
    public static let thinkCloseTokenId: Int = 13;

    /// Control markers exposed as added special tokens.
    private static let addedTokens: Array<(Int, String)> = [
        (3, "<|im_end|>"), (4, "<|im_start|>"), (12, "<think>"), (13, "</think>"),
    ];

    /// The fixture words the journeys generate and prompt with.
    private static let fixtureWords: Array<String> = [
        "What", "is", "the", "play", "about", "Two", "households", "both", "alike",
        "in", "dignity",
    ];

    private static let unknownToken: String = "[UNK]";
    private static let unknownTokenId: Int = 99;
    private static let characterTokenIdBase: Int = 100;

    /// Vocabulary: control markers at their fixed ids, fixture words at
    /// 20+, the unknown fallback, and one id per distinct character; the
    /// computed fillers keep every id from zero through the character block
    /// mapped, because the upstream grammar-vocab extraction walks ids from
    /// zero and stops at the first unmapped id.
    public static func vocabulary() -> [String: Int] {
        var vocabulary: [String: Int] = [:];
        for (tokenId, tokenText) in addedTokens {
            vocabulary[tokenText] = tokenId;
        }
        vocabulary[unknownToken] = unknownTokenId;
        var nextWordTokenId: Int = 20;
        for fixtureWord in fixtureWords {
            vocabulary[fixtureWord] = nextWordTokenId;
            nextWordTokenId += 1;
        }
        var nextCharacterTokenId: Int = characterTokenIdBase;
        for characterText: String in distinctCharacters() {
            vocabulary[characterText] = nextCharacterTokenId;
            nextCharacterTokenId += 1;
        }
        let highestMappedTokenId: Int = max(unknownTokenId, nextCharacterTokenId - 1);
        var mappedTokenIds: Set<Int> = Set(vocabulary.values);
        var fillerTokenId: Int = 0;
        for candidateTokenId: Int in 0...highestMappedTokenId {
            if mappedTokenIds.contains(candidateTokenId) {
                continue;
            }
            while mappedTokenIds.contains(fillerTokenId) {
                fillerTokenId += 1;
            }
            vocabulary["<|filler_\(fillerTokenId)|>"] = candidateTokenId;
            mappedTokenIds.insert(candidateTokenId);
        }
        return vocabulary;
    }

    /// One merge pair per word step, chaining characters into the word so
    /// BPE reduction reaches the whole-word token.
    public static func merges() -> Array<Array<String>> {
        var merges: Array<Array<String>> = [];
        for fixtureWord in fixtureWords {
            let characters: Array<String> = fixtureWord.map { String($0) };
            if characters.count < 2 {
                continue;
            }
            var accumulated: String = characters[0];
            for nextCharacter: String in characters.dropFirst() {
                let merged: String = accumulated + nextCharacter;
                merges.append([accumulated, nextCharacter]);
                accumulated = merged;
            }
        }
        return merges;
    }

    private static func distinctCharacters() -> Array<String> {
        var seenCharacters: Set<String> = [];
        var distinctCharacters: Array<String> = [];
        for fixtureWord in fixtureWords {
            for characterText: String in fixtureWord.map({ String($0) }) {
                if seenCharacters.contains(characterText) == false {
                    seenCharacters.insert(characterText);
                    distinctCharacters.append(characterText);
                }
            }
        }
        return distinctCharacters;
    }

    /// Writes `tokenizer.json` and `tokenizer_config.json` into the model
    /// directory with a Qwen-style chat template.
    public static func writeFiles(modelDirectoryUrl: URL) throws -> Void {
        let addedTokensJson: String = addedTokens.map { (addedToken: (Int, String)) -> String in
            return """
                    {"id": \(addedToken.0), "content": "\(addedToken.1)", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
                """;
        }.joined(separator: ",\n");
        let vocabularyEntriesJson: String = vocabulary()
            .sorted(by: { $0.value < $1.value }).map { (entry: (String, Int)) -> String in
                return "\"\(entry.0)\": \(entry.1)";
            }.joined(separator: ", ");
        let mergesJson: String = merges().map { (mergePair: Array<String>) -> String in
            return "[\"\(mergePair[0])\", \"\(mergePair[1])\"]";
        }.joined(separator: ", ");
        let tokenizerJson: String = """
            {
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [
            \(addedTokensJson)
                ],
                "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "BPE",
                    "vocab": {\(vocabularyEntriesJson)},
                    "merges": [\(mergesJson)],
                    "unk_token": "[UNK]"
                }
            }
            """;
        let tokenizerConfigJson: String = """
            {
                "model_max_length": 4096,
                "tokenizer_class": "PreTrainedTokenizer",
                "bos_token": null,
                "eos_token": "<|im_end|>",
                "pad_token": "<|im_end|>",
                "unk_token": "[UNK]",
                "additional_special_tokens": ["<think>", "</think>"],
                "chat_template": "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
            }
            """;
        try Data(tokenizerJson.utf8).write(
            to: modelDirectoryUrl.appendingPathComponent("tokenizer.json"));
        try Data(tokenizerConfigJson.utf8).write(
            to: modelDirectoryUrl.appendingPathComponent("tokenizer_config.json"));
    }
}
