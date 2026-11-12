use std::ffi::OsString;
use std::path::PathBuf;

use super::parse_speak_arguments;
use crate::arguments::UsageError;

fn arguments(words: &[&str]) -> Vec<OsString> {
    words.iter().map(OsString::from).collect()
}

// Positive: text alone takes every default.
#[test]
fn should_take_text_alone_with_defaults() {
    let speak_arguments = parse_speak_arguments(&arguments(&[
        "But soft, what light through yonder window breaks?",
    ]))
    .expect("text alone is a valid invocation");
    assert_eq!(
        speak_arguments.text.as_deref(),
        Some("But soft, what light through yonder window breaks?")
    );
    assert_eq!(speak_arguments.speed, 1.0);
    assert_eq!(speak_arguments.voice_name, None);
    assert_eq!(speak_arguments.output_path, None);
}

// Positive: every flag, with the text from a file.
#[test]
fn should_read_every_flag() {
    let speak_arguments = parse_speak_arguments(&arguments(&[
        "--file",
        "romeo.txt",
        "--voice",
        "am_michael",
        "--speed",
        "1.25",
        "--model",
        "Kokoro-82M-bf16",
        "--seed",
        "7",
        "--output",
        "romeo.wav",
    ]))
    .expect("every flag is valid");
    assert_eq!(speak_arguments.text_file, Some(PathBuf::from("romeo.txt")));
    assert_eq!(speak_arguments.voice_name.as_deref(), Some("am_michael"));
    assert_eq!(speak_arguments.speed, 1.25);
    assert_eq!(speak_arguments.model_id.as_deref(), Some("Kokoro-82M-bf16"));
    assert_eq!(speak_arguments.seed, Some(7));
    assert_eq!(
        speak_arguments.output_path,
        Some(PathBuf::from("romeo.wav"))
    );
}

// Negative: text twice, a pace out of range, an unknown flag.
#[test]
fn should_refuse_text_twice_a_bad_pace_and_an_unknown_flag() {
    assert_eq!(
        parse_speak_arguments(&arguments(&["Arise", "--file", "sun.txt"])),
        Err(UsageError::SpeakTextGivenTwice)
    );
    assert_eq!(
        parse_speak_arguments(&arguments(&["Arise", "--speed", "9"])),
        Err(UsageError::InvalidSpeakSpeed("9".to_owned()))
    );
    assert!(matches!(
        parse_speak_arguments(&arguments(&["Arise", "--loud"])),
        Err(UsageError::UnknownArgument(_))
    ));
}

// Edge: the slowest and fastest paces are accepted; a repeated flag is not;
// no text at all is valid here because it is read from standard input.
#[test]
fn should_accept_the_pace_edges_and_no_text() {
    for edge_speed in ["0.25", "4"] {
        assert!(parse_speak_arguments(&arguments(&["Arise", "--speed", edge_speed])).is_ok());
    }
    assert_eq!(
        parse_speak_arguments(&arguments(&["Arise", "--speed", "1", "--speed", "2"])),
        Err(UsageError::RepeatedArgument("--speed"))
    );
    assert_eq!(
        parse_speak_arguments(&arguments(&[])).expect("stdin").text,
        None
    );
}
