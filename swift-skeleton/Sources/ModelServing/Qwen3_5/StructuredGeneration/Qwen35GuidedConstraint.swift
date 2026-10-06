import Foundation;

import IpcProtocol;
import MLX;
import MLXGuidedGeneration;
import MLXLMCommon;

/// One compile failure for an enforced structured-generation constraint.
/// Request-scoped: the loaded worker stays reusable for later requests.
public enum Qwen35GuidedConstraintError: Error, CustomStringConvertible {

    /// The guided-regex constraint has no upstream compile path yet; it is
    /// rejected with this bounded reason instead of being masked unsoundly.
    case regexUnsupported;
    /// The grammar compiler rejected the schema or choice grammar.
    case constraintCompilationFailed(String);
    /// The conversation's tokenizer vocabulary could not feed xgrammar.
    case vocabularyExtractionFailed(String);

    public var description: String {
        switch (self) {
        case .regexUnsupported:
            return "the guided-regex constraint is not yet enforceable by the local worker";
        case let .constraintCompilationFailed(detail):
            return "the structured-generation constraint failed to compile: \(detail)";
        case let .vocabularyExtractionFailed(detail):
            return "the tokenizer vocabulary could not feed the grammar engine: \(detail)";
        }
    }
}

/// One enforced structured-generation constraint compiled over the pinned
/// upstream grammar engine (MLXGuidedGeneration/MLXCXGrammar).
///
/// Adapter, not a port: the Rust structured_generation/ package hand-rolled
/// per-token-piece logit biases; upstream compiles a real grammar once and
/// masks every decode step from its matcher state, so the Rust engine is
/// never reproduced. The mask applies only to visible (non-thinking) tokens
/// — the caller owns the thinking-visibility state, mirroring the Rust
/// generated-token emission rule.
public final class Qwen35GuidedConstraint: @unchecked Sendable {

    private let grammarConstraint: GrammarConstraint;
    /// Set when the matcher reports its stop state, either at commit time or
    /// on a mask query; sticky so a terminated grammar never masks again.
    private var hasReachedStopState: Bool;
    private let stateLock: NSLock;

    /// Testable construction over an already-compiled constraint; production
    /// callers go through `compile`.
    init(grammarConstraint: GrammarConstraint) {
        self.grammarConstraint = grammarConstraint;
        self.hasReachedStopState = false;
        self.stateLock = NSLock();
    }

    /** Compiles one supervisor-validated constraint or explains why this
    worker cannot mask it.

    - Parameters:
      - constraint: The IPC structured-generation constraint.
      - tokenizer: The conversation tokenizer whose vocabulary binds the
        grammar.
      - endOfSequenceTokenId: The end-of-sequence id registered as a grammar
        stop token.
    - Throws: Qwen35GuidedConstraintError for unsupported kinds or compile
      failures.
     */
    public static func compile(
        constraint: StructuredGenerationConstraint,
        tokenizer: any Tokenizer,
        endOfSequenceTokenId: Int32
    ) throws -> Qwen35GuidedConstraint {
        let extractedVocab: TokenizerVocabExtractor.GrammarVocab =
            TokenizerVocabExtractor.extractForGrammar(from: tokenizer);
        let grammarTokenizer: GrammarTokenizer;
        do {
            grammarTokenizer = try GrammarTokenizer(
                vocab: extractedVocab.vocab,
                vocabType: extractedVocab.vocabType,
                eosTokenId: endOfSequenceTokenId);
        } catch {
            throw Qwen35GuidedConstraintError.vocabularyExtractionFailed(String(describing: error));
        }
        let compiledConstraint: GrammarConstraint;
        do {
            switch (constraint) {
            case .jsonObject:
                compiledConstraint = try GrammarConstraint(
                    tokenizer: grammarTokenizer, jsonSchema: "{\"type\":\"object\"}");
            case let .jsonSchema(schemaJson):
                compiledConstraint = try GrammarConstraint(
                    tokenizer: grammarTokenizer, jsonSchema: schemaJson);
            case let .choice(choices):
                compiledConstraint = try GrammarConstraint(
                    tokenizer: grammarTokenizer,
                    grammar: Qwen35GuidedConstraint.choiceGrammar(choices),
                    rootRule: nil);
            case .regex:
                throw Qwen35GuidedConstraintError.regexUnsupported;
            }
        } catch let guidedError as Qwen35GuidedConstraintError {
            throw guidedError;
        } catch {
            throw Qwen35GuidedConstraintError.constraintCompilationFailed(
                String(describing: error));
        }
        return Qwen35GuidedConstraint(grammarConstraint: compiledConstraint);
    }

    /** Builds the GBNF alternation matching exactly one of the choice
    literals, so the grammar enforces the same surface the Rust encoded
    choice sequences did. */
    static func choiceGrammar(_ choices: Array<String>) -> String {
        var alternation: String = "root ::= ";
        for (choiceIndex, choiceText) in choices.enumerated() {
            if choiceIndex > 0 {
                alternation += " | ";
            }
            alternation += "\"\(Qwen35GuidedConstraint.escapedGbnfLiteral(choiceText))\"";
        }
        return alternation;
    }

    private static func escapedGbnfLiteral(_ literalText: String) -> String {
        var escapedLiteral: String = String();
        for literalByte: UInt8 in literalText.utf8 {
            switch (literalByte) {
            case UInt8(ascii: "\""): escapedLiteral += "\\\"";
            case UInt8(ascii: "\\"): escapedLiteral += "\\\\";
            case UInt8(ascii: "\n"): escapedLiteral += "\\n";
            case UInt8(ascii: "\r"): escapedLiteral += "\\r";
            case UInt8(ascii: "\t"): escapedLiteral += "\\t";
            default:
                if literalByte < 0x20 || literalByte >= 0x7F {
                    escapedLiteral += String(format: "\\x%02X", literalByte);
                } else {
                    escapedLiteral.append(Character(UnicodeScalar(literalByte)));
                }
            }
        }
        return escapedLiteral;
    }

    /** Whether the grammar reached its stop state; once terminated the mask
    no longer applies and the model's natural end-of-sequence ends the
    request, mirroring the upstream loop's terminal handling. */
    public func isTerminated() -> Bool {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        if self.hasReachedStopState {
            return true;
        }
        do {
            let maskState: MaskResult = try self.grammarConstraint.computeMask();
            self.hasReachedStopState = maskState.isTerminated;
            return maskState.isTerminated;
        } catch {
            return false;
        }
    }

    /** Applies the grammar mask to one logit row: allowed tokens keep their
    logits, disallowed tokens drop to negative infinity. Returns the input
    row untouched while the grammar is in an unconditional splice
    (`needsApply == false`) or has already terminated, mirroring the
    upstream mask application. */
    public func maskLogits(_ logitsRow: MLXArray) throws -> MLXArray {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        if self.hasReachedStopState {
            return logitsRow;
        }
        let maskState: MaskResult = try self.grammarConstraint.computeMask();
        if maskState.isTerminated {
            self.hasReachedStopState = true;
            return logitsRow;
        }
        if maskState.needsApply == false {
            return logitsRow;
        }
        var biasValues: Array<Float> = Array(repeating: 0.0, count: logitsRow.count);
        let vocabularySize: Int = min(biasValues.count, maskState.mask.count * 32);
        for tokenIndex in 0..<vocabularySize {
            let maskWord: Int32 = maskState.mask[tokenIndex / 32];
            let maskBit: UInt32 = (UInt32(bitPattern: maskWord) >> UInt32(tokenIndex % 32)) & 1;
            if maskBit == 0 {
                biasValues[tokenIndex] = -Float.infinity;
            }
        }
        return logitsRow + MLXArray(biasValues);
    }

    /** Commits one sampled visible token to advance the grammar state.

    - Throws: Qwen35GuidedConstraintError when the sampled token is outside
      the most recent mask.
     */
    public func commitToken(_ sampledTokenId: Int32) throws -> Void {
        do {
            let commitResult: CommitResult = try self.grammarConstraint.commitToken(sampledTokenId);
            if commitResult.isTerminated {
                self.stateLock.lock();
                self.hasReachedStopState = true;
                self.stateLock.unlock();
            }
        } catch {
            throw Qwen35GuidedConstraintError.constraintCompilationFailed(
                String(describing: error));
        }
    }
}
