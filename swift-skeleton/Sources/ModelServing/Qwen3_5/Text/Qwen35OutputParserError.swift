import Foundation;

/// Typed failures raised while constructing a parser from declared tools.
///
/// Mirrors crates/model-serving/src/qwen3_5/text/output_parser_error.rs. The
/// messages name the offending tool and property so a rejected declaration is
/// actionable; the closed-path parse failures never surface because the parser
/// fails open and forwards the body instead.
public enum Qwen35OutputParserError: Error, CustomStringConvertible, Sendable {
    case duplicateDeclaredTool(functionName: String);
    case invalidDeclaredToolSchema(functionName: String, problem: String);
    case declaredToolSchemaMustBeObject(functionName: String);
    case invalidToolPropertySchema(functionName: String, parameterName: String);
    case invalidToolParameterTypeDeclaration(functionName: String, parameterName: String);
    case toolCallMissingFunction;

    public var description: String {
        switch self {
        case let .duplicateDeclaredTool(functionName):
            return "declared tool '\(functionName)' is declared more than once";
        case let .invalidDeclaredToolSchema(functionName, problem):
            return "declared tool '\(functionName)' has an invalid schema: \(problem)";
        case let .declaredToolSchemaMustBeObject(functionName):
            return "declared tool '\(functionName)' schema must be a JSON object";
        case let .invalidToolPropertySchema(functionName, parameterName):
            return
                "declared tool '\(functionName)' property '\(parameterName)' must be a JSON object schema";
        case let .invalidToolParameterTypeDeclaration(functionName, parameterName):
            return
                "declared tool '\(functionName)' property '\(parameterName)' has an unsupported type declaration";
        case .toolCallMissingFunction:
            return "the tool-call body did not name a function";
        }
    }
}
