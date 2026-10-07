import Foundation

import Testing;

import AstronomicalCli;
import JourneyCategories;

@testable import AstronomicalCli;

/// Parser journeys for the whole `astronomical` command tree, porting
/// arguments.rs.
@Suite(.tags(.hermeticJourney))
final class CliArgumentsTests {

    @Test
    func should_print_help_for_help_flag() {
        #expect(CliJourneySupport.parse(["--help"]) == .success(.help));
        #expect(CliJourneySupport.parse(["-h"]) == .success(.help));
        #expect(CliJourneySupport.parse(["respond", "hi", "-h"]) == .success(.help));
    }

    @Test
    func should_ignore_verbose_as_a_global_flag() {
        #expect(CliJourneySupport.parse(["status", "--verbose"]) == .success(.status));
        #expect(CliJourneySupport.parse(["-v", "status"]) == .success(.status));
    }

    @Test
    func should_print_version_for_version_flag() {
        #expect(CliJourneySupport.parse(["--version"]) == .success(.version));
    }

    @Test
    func should_require_a_command_when_invoked_bare() {
        guard case let .failure(usageError) = CliJourneySupport.parse([]) else {
            Issue.record("a bare invocation must be a usage error");
            return;
        }
        #expect(usageError == .missingCommand);
    }

    @Test
    func should_parse_launch_without_a_tool_name() {
        #expect(CliJourneySupport.parse(["launch"]) == .success(.launch(LaunchArguments(toolSlug: nil, modelId: nil))));
    }

    @Test
    func should_parse_launch_opencode_and_model_flag_in_either_order() {
        #expect(CliJourneySupport.parse(["launch", "opencode", "--model", "m1"])
            == .success(.launch(LaunchArguments(toolSlug: "opencode", modelId: "m1"))));
        #expect(CliJourneySupport.parse(["launch", "--model", "m1", "opencode"])
            == .success(.launch(LaunchArguments(toolSlug: "opencode", modelId: "m1"))));
    }

    @Test
    func should_reject_unknown_tools_as_launch_arguments_not_usage() {
        #expect(CliJourneySupport.parse(["launch", "some-tool"]).isSuccess);
    }

    @Test
    func should_reject_two_tool_names() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["launch", "opencode", "other"]) else {
            Issue.record("two tool names must be a usage error");
            return;
        }
        #expect(usageError == .multipleTools);
    }

    @Test
    func should_reject_a_missing_model_value() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["launch", "--model"]) else {
            Issue.record("a missing --model value must be a usage error");
            return;
        }
        #expect(usageError == .missingValue("--model"));
    }

    @Test
    func should_parse_respond_with_a_bare_prompt() {
        #expect(CliJourneySupport.parse(["respond", "hello"])
            == .success(.respond(RespondArguments(
                prompt: "hello",
                imagePaths: [],
                modelId: nil,
                instructions: nil,
                thinkingBudget: nil,
                schemaPath: nil,
                noStream: false
            ))));
    }

    @Test
    func should_parse_respond_prompt_with_model_and_no_stream() {
        #expect(CliJourneySupport.parse(["respond", "hi", "--model", "m1", "--no-stream"])
            == .success(.respond(RespondArguments(
                prompt: "hi",
                imagePaths: [],
                modelId: "m1",
                instructions: nil,
                thinkingBudget: nil,
                schemaPath: nil,
                noStream: true
            ))));
    }

    @Test
    func should_reject_respond_without_a_prompt_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["respond"]) else {
            Issue.record("respond without a prompt must be a usage error");
            return;
        }
        #expect(usageError == .respondPromptRequired);
    }

    @Test
    func should_reject_respond_with_an_unknown_argument_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["respond", "hi", "--unknown"]) else {
            Issue.record("an unknown respond argument must be a usage error");
            return;
        }
        #expect(usageError == .unknownArgument("--unknown"));
    }

    @Test
    func should_parse_repeatable_image_arguments_after_the_prompt() {
        #expect(CliJourneySupport.parse(["respond", "hi", "--image", "a.png", "--image", "b.jpg"])
            == .success(.respond(RespondArguments(
                prompt: "hi",
                imagePaths: ["a.png", "b.jpg"],
                modelId: nil,
                instructions: nil,
                thinkingBudget: nil,
                schemaPath: nil,
                noStream: false
            ))));
    }

    @Test
    func should_parse_image_arguments_before_the_prompt() {
        #expect(CliJourneySupport.parse(["respond", "--image", "a.png", "hi"])
            == .success(.respond(RespondArguments(
                prompt: "hi",
                imagePaths: ["a.png"],
                modelId: nil,
                instructions: nil,
                thinkingBudget: nil,
                schemaPath: nil,
                noStream: false
            ))));
    }

    @Test
    func should_reject_an_image_flag_missing_a_value_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["respond", "hi", "--image"]) else {
            Issue.record("a missing --image value must be a usage error");
            return;
        }
        #expect(usageError == .missingValue("--image"));
    }

    @Test
    func should_reject_an_image_flag_with_a_flag_shaped_value_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["respond", "hi", "--image", "--model"]) else {
            Issue.record("a flag-shaped --image value must be a usage error");
            return;
        }
        #expect(usageError == .missingValue("--image"));
    }

    @Test
    func should_parse_respond_prompt_from_the_text_flag() {
        #expect(CliJourneySupport.parse(["respond", "--text", "hello"])
            == .success(.respond(RespondArguments(
                prompt: "hello",
                imagePaths: [],
                modelId: nil,
                instructions: nil,
                thinkingBudget: nil,
                schemaPath: nil,
                noStream: false
            ))));
    }

    @Test
    func should_parse_respond_instructions_flag() {
        #expect(CliJourneySupport.parse(["respond", "hi", "--instructions", "be terse"])
            == .success(.respond(RespondArguments(
                prompt: "hi",
                imagePaths: [],
                modelId: nil,
                instructions: "be terse",
                thinkingBudget: nil,
                schemaPath: nil,
                noStream: false
            ))));
    }

    @Test
    func should_parse_respond_thinking_budget_flag() {
        #expect(CliJourneySupport.parse(["respond", "hi", "--thinking-budget", "512"])
            == .success(.respond(RespondArguments(
                prompt: "hi",
                imagePaths: [],
                modelId: nil,
                instructions: nil,
                thinkingBudget: 512,
                schemaPath: nil,
                noStream: false
            ))));
    }

    @Test
    func should_parse_the_schema_flag_value_as_a_path() {
        #expect(CliJourneySupport.parse(["respond", "hi", "--schema", "schema.json"])
            == .success(.respond(RespondArguments(
                prompt: "hi",
                imagePaths: [],
                modelId: nil,
                instructions: nil,
                thinkingBudget: nil,
                schemaPath: "schema.json",
                noStream: false
            ))));
    }

    @Test
    func should_reject_a_repeated_schema_flag_as_a_repeated_argument() {
        guard case let .failure(usageError) = CliJourneySupport.parse([
            "respond", "hi", "--schema", "a.json", "--schema", "b.json",
        ]) else {
            Issue.record("a repeated --schema flag must be a usage error");
            return;
        }
        #expect(usageError == .repeatedArgument("--schema"));
    }

    @Test
    func should_parse_a_zero_thinking_budget_as_a_valid_disable() {
        #expect(CliJourneySupport.parse(["respond", "hi", "--thinking-budget", "0"])
            == .success(.respond(RespondArguments(
                prompt: "hi",
                imagePaths: [],
                modelId: nil,
                instructions: nil,
                thinkingBudget: 0,
                schemaPath: nil,
                noStream: false
            ))));
    }

    @Test
    func should_reject_respond_with_both_a_positional_prompt_and_text_flag() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["respond", "hi", "--text", "other"]) else {
            Issue.record("a positional prompt plus --text must be a usage error");
            return;
        }
        #expect(usageError == .respondPromptConflict);
    }

    @Test
    func should_reject_a_repeated_text_flag_as_a_repeated_argument() {
        guard case let .failure(usageError) = CliJourneySupport.parse([
            "respond", "--text", "one", "--text", "two",
        ]) else {
            Issue.record("a repeated --text flag must be a usage error");
            return;
        }
        #expect(usageError == .repeatedArgument("--text"));
    }

    @Test
    func should_reject_respond_with_two_positional_prompts() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["respond", "one", "two"]) else {
            Issue.record("two positional prompts must be a usage error");
            return;
        }
        #expect(usageError == .respondPromptConflict);
    }

    @Test
    func should_reject_a_non_numeric_thinking_budget() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["respond", "hi", "--thinking-budget", "many"]) else {
            Issue.record("a non-numeric thinking budget must be a usage error");
            return;
        }
        #expect(usageError == .invalidThinkingBudget("many"));
    }

    @Test
    func should_reject_a_thinking_budget_above_the_u16_maximum() {
        guard case let .failure(usageError) = CliJourneySupport.parse([
            "respond", "hi", "--thinking-budget", "65536",
        ]) else {
            Issue.record("a thinking budget above the u16 maximum must be a usage error");
            return;
        }
        #expect(usageError == .invalidThinkingBudget("65536"));
    }

    @Test
    func should_parse_the_maximum_thinking_budget() {
        #expect(CliJourneySupport.parse(["respond", "hi", "--thinking-budget", "65535"])
            == .success(.respond(RespondArguments(
                prompt: "hi",
                imagePaths: [],
                modelId: nil,
                instructions: nil,
                thinkingBudget: 65535,
                schemaPath: nil,
                noStream: false
            ))));
    }

    @Test
    func should_parse_all_new_respond_flags_together() {
        #expect(CliJourneySupport.parse([
            "respond", "hi", "--model", "m1", "--instructions", "be terse",
            "--thinking-budget", "32", "--schema", "s.json", "--no-stream", "--image", "i.png",
        ]) == .success(.respond(RespondArguments(
            prompt: "hi",
            imagePaths: ["i.png"],
            modelId: "m1",
            instructions: "be terse",
            thinkingBudget: 32,
            schemaPath: "s.json",
            noStream: true
        ))));
    }
}

extension Result {

    /// Whether the parse succeeded, for the one journey that asserts
    /// unknown tool names parse and fail later at resolution.
    var isSuccess: Bool {
        if case .success = self {
            return true;
        }
        return false;
    }
}
