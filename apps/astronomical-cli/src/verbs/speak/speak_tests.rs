//! Unit contracts for the `speak` journey's own steps: where the text comes
//! from, where the audio goes and in what format, and how reply frames end
//! the exchange.

use std::path::PathBuf;

use astronomical_ipc_protocol::{DaemonResponse, SpeechResultMetadata};
use astronomical_speech_contract::{AudioFormat, SpeechFailure};

use super::speak_exchange::take_frame;
use super::speak_output::{reserve_audio_output, write_audio};
use super::{SpeakArguments, SpeakError, resolve_spoken_text};

const ROMEO_LINE: &str = "But soft, what light through yonder window breaks?";

fn speak_arguments(text: Option<&str>, text_file: Option<PathBuf>) -> SpeakArguments {
    SpeakArguments {
        text: text.map(str::to_owned),
        text_file,
        model_id: None,
        voice_name: None,
        speed: 1.0,
        seed: None,
        output_path: None,
    }
}

fn spoken_metadata() -> SpeechResultMetadata {
    SpeechResultMetadata {
        sample_rate_hertz: 24_000,
        sample_count: 2,
        segment_count: 1,
        cut_short_segment_count: 0,
        voice_name: "af_heart".to_owned(),
        seed: 42,
    }
}

// Positive: the text comes from the argument, else the file, else stdin.
#[test]
fn should_read_the_text_from_the_argument_the_file_or_stdin() {
    let mut unused_stdin: &[u8] = b"";
    assert_eq!(
        resolve_spoken_text(&speak_arguments(Some(ROMEO_LINE), None), &mut unused_stdin)
            .expect("the argument"),
        ROMEO_LINE
    );
    let text_directory = tempfile::tempdir().expect("a temporary directory");
    let text_file = text_directory.path().join("romeo.txt");
    std::fs::write(&text_file, "It is the east, and Juliet is the sun.").expect("a text file");
    assert_eq!(
        resolve_spoken_text(&speak_arguments(None, Some(text_file)), &mut unused_stdin)
            .expect("the file"),
        "It is the east, and Juliet is the sun."
    );
    let mut stdin_text: &[u8] = b"Arise, fair sun.";
    assert_eq!(
        resolve_spoken_text(&speak_arguments(None, None), &mut stdin_text).expect("stdin"),
        "Arise, fair sun."
    );
}

// Negative: blank text and a missing file are the user's to fix.
#[test]
fn should_refuse_blank_text_and_a_missing_file() {
    let mut blank_stdin: &[u8] = b"   \n";
    let blank_error = resolve_spoken_text(&speak_arguments(None, None), &mut blank_stdin)
        .expect_err("blank text is refused");
    assert!(matches!(blank_error, SpeakError::TextEmpty));
    assert!(blank_error.is_usage_error());
    let mut unused_stdin: &[u8] = b"";
    assert!(matches!(
        resolve_spoken_text(
            &speak_arguments(None, Some(PathBuf::from("no/such/romeo.txt"))),
            &mut unused_stdin
        ),
        Err(SpeakError::TextUnreadable { .. })
    ));
}

// Positive and negative: the extension picks the format; anything else is
// refused, and an existing file is never overwritten.
#[test]
fn should_pick_the_format_from_the_extension_and_never_overwrite() {
    let output_directory = tempfile::tempdir().expect("a temporary directory");
    let default_output =
        reserve_audio_output(None, output_directory.path()).expect("a timestamped WAV");
    assert_eq!(default_output.audio_format, AudioFormat::Wav);
    default_output.reserved_file.discard();
    let pcm_path = output_directory.path().join("romeo.pcm");
    let pcm_output =
        reserve_audio_output(Some(&pcm_path), output_directory.path()).expect("a .pcm output");
    assert_eq!(pcm_output.audio_format, AudioFormat::Pcm);
    assert!(matches!(
        reserve_audio_output(Some(&pcm_path), output_directory.path()),
        Err(SpeakError::OutputExists { .. })
    ));
    pcm_output.reserved_file.discard();
    assert!(matches!(
        reserve_audio_output(
            Some(&output_directory.path().join("romeo.mp3")),
            output_directory.path()
        ),
        Err(SpeakError::OutputMustBeAudio { .. })
    ));
}

// Positive and edge: WAV gains its header; half a sample is refused and
// leaves no file behind.
#[test]
fn should_write_wav_with_its_header_and_refuse_half_a_sample() {
    let output_directory = tempfile::tempdir().expect("a temporary directory");
    let wav_path = output_directory.path().join("juliet.wav");
    let wav_output =
        reserve_audio_output(Some(&wav_path), output_directory.path()).expect("a .wav output");
    let saved_path = write_audio(wav_output, &[0, 0, 0, 64], 24_000).expect("whole samples");
    let saved_bytes = std::fs::read(saved_path).expect("the WAV is readable");
    assert_eq!(&saved_bytes[..4], b"RIFF");
    assert_eq!(saved_bytes.len(), 48);
    let pcm_path = output_directory.path().join("juliet.pcm");
    let pcm_output =
        reserve_audio_output(Some(&pcm_path), output_directory.path()).expect("a .pcm output");
    assert!(matches!(
        write_audio(pcm_output, &[0, 0, 7], 24_000),
        Err(SpeakError::ReturnedAudioInvalid)
    ));
    assert!(!pcm_path.exists());
}

// Positive: audio frames accumulate until the completed frame ends the
// exchange with all of them.
#[test]
fn should_join_audio_frames_until_the_completed_frame() {
    let mut pcm16_bytes = Vec::new();
    for audio_frame in [vec![0, 0], vec![0, 64]] {
        assert!(
            take_frame(
                DaemonResponse::SpeechAudio {
                    pcm16_bytes: audio_frame
                },
                &mut pcm16_bytes
            )
            .expect("an audio frame")
            .is_none()
        );
    }
    let spoken_reply = take_frame(
        DaemonResponse::SpeechCompleted {
            model: "Kokoro-82M-bf16".to_owned(),
            metadata: spoken_metadata(),
        },
        &mut pcm16_bytes,
    )
    .expect("the terminal frame")
    .expect("the exchange ends");
    assert_eq!(spoken_reply.pcm16_bytes, [0, 0, 0, 64]);
    assert_eq!(spoken_reply.model, "Kokoro-82M-bf16");
}

// Negative: a failure, a refusal, and an unrelated frame each end it.
#[test]
fn should_end_on_a_failure_a_refusal_or_an_unrelated_frame() {
    let mut pcm16_bytes = Vec::new();
    let failure = take_frame(
        DaemonResponse::SpeechFailed {
            reason: SpeechFailure::invalid_request("there is no voice \"alloy\""),
        },
        &mut pcm16_bytes,
    )
    .expect_err("a failed request");
    assert!(failure.is_usage_error(), "a bad voice is the user's to fix");
    assert!(matches!(
        take_frame(
            DaemonResponse::GenerationRejected {
                reason: "busy".to_owned()
            },
            &mut pcm16_bytes
        ),
        Err(SpeakError::SpeechRejected { .. })
    ));
    assert!(matches!(
        take_frame(
            DaemonResponse::Catalog { entries: vec![] },
            &mut pcm16_bytes
        ),
        Err(SpeakError::DaemonStoppedResponding)
    ));
}
