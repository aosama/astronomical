import Foundation

/// Where the chat window appears when the app launches.
///
/// A person who opens the app wants it in front. A developer's rebuild in the
/// middle of other work does not: `cargo xtask app dev --background` opens the
/// app with `open -g`, which launches it inactive, and passes this argument
/// so the app also leaves its window behind the others instead of raising
/// itself. The argument's text is repeated in xtask
/// (`LAUNCH_IN_BACKGROUND_ARGUMENT` in app_bundle/running_instance.rs).
public enum ChatLaunchPlacement: Equatable, Sendable {
  case inFront
  case behindOtherWindows

  public static let launchInBackgroundArgument = "--launch-in-background"

  public static func fromLaunchArguments(_ launchArguments: [String] = CommandLine.arguments) -> ChatLaunchPlacement {
    launchArguments.dropFirst().contains(launchInBackgroundArgument) ? .behindOtherWindows : .inFront
  }
}
