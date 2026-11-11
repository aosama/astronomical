//! The Development app's background links: the commands that open them and
//! the links themselves.

use super::channel_profile::{Channel, ChannelProfile};
use super::command::{AppCommand, parse_app_command};
use super::running_instance::{draft_link, export_folder_link};

fn arguments(argument_words: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_word in argument_words {
        owned_arguments.push((*argument_word).to_owned());
    }
    owned_arguments
}

#[test]
fn should_parse_a_draft_and_refuse_an_empty_or_missing_one() {
    assert_eq!(
        parse_app_command(&arguments(&["draft", "What is 15% of 80?"])),
        Ok(AppCommand::Draft("What is 15% of 80?".to_owned()))
    );
    assert!(parse_app_command(&arguments(&["draft"])).is_err());
    assert!(parse_app_command(&arguments(&["draft", "   "])).is_err());
    assert!(parse_app_command(&arguments(&["draft", "one", "two"])).is_err());
}

/// The draft is data in a link: a character that means something in a URL
/// must never be able to end the text or add a parameter.
#[test]
fn should_encode_every_character_of_a_draft_that_a_link_could_misread() {
    let development_profile = ChannelProfile::for_channel(Channel::Development);

    assert_eq!(
        draft_link(&development_profile, "abc-XYZ_09.~"),
        "astronomical-dev://draft?text=abc-XYZ_09.~"
    );
    assert_eq!(
        draft_link(&development_profile, "a b&c=d#e?f/g%"),
        "astronomical-dev://draft?text=a%20b%26c%3Dd%23e%3Ff%2Fg%25"
    );
    assert_eq!(
        draft_link(&development_profile, "é\n"),
        "astronomical-dev://draft?text=%C3%A9%0A"
    );
    assert_eq!(
        draft_link(&development_profile, ""),
        "astronomical-dev://draft?text="
    );
}

#[test]
fn should_parse_an_export_folder_and_refuse_a_relative_missing_or_extra_one() {
    assert_eq!(
        parse_app_command(&arguments(&["export-to", "/work/exports"])),
        Ok(AppCommand::ExportTo("/work/exports".to_owned()))
    );
    assert!(parse_app_command(&arguments(&["export-to"])).is_err());
    assert!(parse_app_command(&arguments(&["export-to", "/"])).is_err());
    assert!(parse_app_command(&arguments(&["export-to", "exports"])).is_err());
    assert!(parse_app_command(&arguments(&["export-to", "/one", "/two"])).is_err());
}

/// The folder is data in a link, as a draft is: a space or an ampersand in
/// its name must not end the path or add a parameter.
#[test]
fn should_encode_an_export_folder_so_its_name_cannot_change_the_link() {
    let development_profile = ChannelProfile::for_channel(Channel::Development);

    assert_eq!(
        export_folder_link(&development_profile, "/work/my exports&path=x"),
        "astronomical-dev://export-folder?path=%2Fwork%2Fmy%20exports%26path%3Dx"
    );
}
