use crate::commit_identity::{
    email_in_angle_brackets, identity_emails_in_raw_commit, is_publishable_identity_email,
    rev_list_arguments_for_pre_push_line, unpublishable_identity_emails,
};

const PUSHED_OBJECT_ID: &str = "1111111111111111111111111111111111111111";
const REMOTE_OBJECT_ID: &str = "2222222222222222222222222222222222222222";
const ABSENT_OBJECT_ID: &str = "0000000000000000000000000000000000000000";

fn raw_commit(author_email: &str, committer_email: &str, message_text: &str) -> String {
    format!(
        "tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n\
         author Example Person <{author_email}> 1700000000 +0000\n\
         committer Example Person <{committer_email}> 1700000000 +0000\n\
         \n\
         {message_text}\n"
    )
}

#[test]
fn should_accept_the_numeric_github_noreply_address() {
    assert!(is_publishable_identity_email("583231+octocat@users.noreply.github.com", &[]));
}

#[test]
fn should_accept_github_owned_and_service_noreply_addresses() {
    assert!(is_publishable_identity_email("noreply@github.com", &[]));
    assert!(is_publishable_identity_email("noreply@anthropic.com", &[]));
    assert!(is_publishable_identity_email("49699333+dependabot[bot]@users.noreply.github.com", &[]));
}

#[test]
fn should_accept_reserved_fixture_domains() {
    assert!(is_publishable_identity_email("fixture@astronomical.invalid", &[]));
    assert!(is_publishable_identity_email("fixture@build.test", &[]));
    assert!(is_publishable_identity_email("fixture@host.local", &[]));
}

#[test]
fn should_reject_a_personal_mailbox() {
    assert!(!is_publishable_identity_email("person@mail.example.org", &[]));
}

#[test]
fn should_reject_a_login_only_noreply_address_that_github_credits_to_a_stranger() {
    assert!(!is_publishable_identity_email("octocat@users.noreply.github.com", &[]));
}

#[test]
fn should_reject_a_noreply_address_with_a_non_numeric_account_id() {
    assert!(!is_publishable_identity_email("12ab+octocat@users.noreply.github.com", &[]));
    assert!(!is_publishable_identity_email("+octocat@users.noreply.github.com", &[]));
    assert!(!is_publishable_identity_email("583231+@users.noreply.github.com", &[]));
}

#[test]
fn should_reject_empty_and_malformed_addresses() {
    assert!(!is_publishable_identity_email("", &[]));
    assert!(!is_publishable_identity_email("no-at-sign", &[]));
    assert!(!is_publishable_identity_email("@github.com", &[]));
    assert!(!is_publishable_identity_email("person@", &[]));
}

#[test]
fn should_compare_addresses_case_insensitively() {
    assert!(is_publishable_identity_email("NoReply@GitHub.com", &[]));
    assert!(!is_publishable_identity_email("OctoCat@Users.NoReply.GitHub.com", &[]));
}

#[test]
fn should_extract_the_email_between_angle_brackets() {
    assert_eq!(
        email_in_angle_brackets("Example Person <person@mail.example.org> 1700000000 +0000"),
        Some("person@mail.example.org")
    );
    assert_eq!(email_in_angle_brackets("no brackets here"), None);
    assert_eq!(email_in_angle_brackets("unterminated <bracket"), None);
}

#[test]
fn should_collect_author_committer_and_co_author_trailer_emails() {
    let commit_text = raw_commit(
        "583231+octocat@users.noreply.github.com",
        "noreply@github.com",
        "Subject\n\nBody\n\nCo-authored-by: Helper <person@mail.example.org>\nco-authored-by: Tool <noreply@anthropic.com>",
    );
    assert_eq!(
        identity_emails_in_raw_commit(&commit_text),
        vec![
            "583231+octocat@users.noreply.github.com".to_owned(),
            "noreply@github.com".to_owned(),
            "person@mail.example.org".to_owned(),
            "noreply@anthropic.com".to_owned(),
        ]
    );
}

#[test]
fn should_ignore_angle_brackets_in_ordinary_message_lines() {
    let commit_text = raw_commit(
        "583231+octocat@users.noreply.github.com",
        "583231+octocat@users.noreply.github.com",
        "Subject mentions <person@mail.example.org> in prose",
    );
    assert!(unpublishable_identity_emails(&commit_text, &[]).is_empty());
}

#[test]
fn should_report_only_the_unpublishable_emails_of_a_commit() {
    let commit_text = raw_commit(
        "person@mail.example.org",
        "octocat@users.noreply.github.com",
        "Subject\n\nCo-authored-by: Tool <noreply@anthropic.com>",
    );
    assert_eq!(
        unpublishable_identity_emails(&commit_text, &[]),
        vec![
            "person@mail.example.org".to_owned(),
            "octocat@users.noreply.github.com".to_owned(),
        ]
    );
}

#[test]
fn should_select_the_new_range_for_an_updated_remote_branch() {
    let pre_push_line =
        format!("refs/heads/topic {PUSHED_OBJECT_ID} refs/heads/topic {REMOTE_OBJECT_ID}");
    assert_eq!(
        rev_list_arguments_for_pre_push_line(&pre_push_line),
        Some(vec![format!("{REMOTE_OBJECT_ID}..{PUSHED_OBJECT_ID}")])
    );
}

#[test]
fn should_select_only_unpublished_commits_for_a_new_remote_branch() {
    let pre_push_line =
        format!("refs/heads/topic {PUSHED_OBJECT_ID} refs/heads/topic {ABSENT_OBJECT_ID}");
    assert_eq!(
        rev_list_arguments_for_pre_push_line(&pre_push_line),
        Some(vec![
            PUSHED_OBJECT_ID.to_owned(),
            "--not".to_owned(),
            "--remotes".to_owned(),
        ])
    );
}

#[test]
fn should_select_nothing_for_a_branch_deletion() {
    let pre_push_line =
        format!("(delete) {ABSENT_OBJECT_ID} refs/heads/topic {REMOTE_OBJECT_ID}");
    assert_eq!(rev_list_arguments_for_pre_push_line(&pre_push_line), None);
}

#[test]
fn should_select_nothing_for_a_malformed_pre_push_line() {
    assert_eq!(rev_list_arguments_for_pre_push_line(""), None);
    assert_eq!(rev_list_arguments_for_pre_push_line("refs/heads/topic only-two"), None);
}

#[test]
fn should_accept_an_address_the_developer_declared_publishable() {
    let declared_emails = vec!["Relay.Address@mail.example.org".to_owned()];
    assert!(is_publishable_identity_email("relay.address@mail.example.org", &declared_emails));
}

#[test]
fn should_still_reject_undeclared_addresses_when_others_are_declared() {
    let declared_emails = vec!["relay.address@mail.example.org".to_owned()];
    assert!(!is_publishable_identity_email("person@mail.example.org", &declared_emails));
}

#[test]
fn should_ignore_blank_declared_entries() {
    let declared_emails = vec!["   ".to_owned()];
    assert!(!is_publishable_identity_email("person@mail.example.org", &declared_emails));
}

#[test]
fn should_pass_a_commit_whose_only_personal_address_is_declared() {
    let commit_text = raw_commit(
        "relay.address@mail.example.org",
        "noreply@github.com",
        "Subject",
    );
    let declared_emails = vec!["relay.address@mail.example.org".to_owned()];
    assert!(unpublishable_identity_emails(&commit_text, &declared_emails).is_empty());
}
