use super::AudioFormat;

#[test]
fn should_read_the_openai_format_names() {
    assert_eq!(AudioFormat::from_name("wav"), Some(AudioFormat::Wav));
    assert_eq!(AudioFormat::from_name("pcm"), Some(AudioFormat::Pcm));
    assert_eq!(AudioFormat::Wav.mime_type(), "audio/wav");
}

// Negative: formats the engine does not encode.
#[test]
fn should_refuse_formats_it_cannot_encode() {
    assert_eq!(AudioFormat::from_name("mp3"), None);
    assert_eq!(AudioFormat::from_name("WAV"), None);
}
