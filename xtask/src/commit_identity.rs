//! Keeps personal email addresses out of the public history.
//!
//! Every author, committer, and `Co-authored-by` email that reaches GitHub is
//! published forever: rewriting history afterwards cannot reach forks, pull
//! request refs, or scrapers. GitHub also credits a commit to whichever
//! account owns a bare `login@users.noreply.github.com` address, so a guessed
//! login hands the commit to an unrelated person. The check therefore allows
//! only addresses that can never be a private inbox:
//!
//! - the numeric GitHub no-reply form `<id>+<login>@users.noreply.github.com`,
//!   which GitHub resolves by account id rather than by guessable login;
//! - GitHub-owned addresses (`noreply@github.com` for web merges, bots);
//! - any `noreply@` service address such as tool co-author trailers;
//! - reserved, non-deliverable top-level domains used by test fixtures;
//! - addresses the developer has already made public and declares locally with
//!   `git config --add astronomical.publishableEmail <address>`. The list
//!   lives in the developer's own git config, never in this repository, so no
//!   personal address is written into the public sources.
//!
//! `--staged` checks the identity the next commit will carry (pre-commit).
//! `--pre-push` reads git's pre-push lines on stdin and checks every commit
//! that the push would publish, including co-author trailers that only exist
//! once the message is written.

use std::io::Read;
use std::process::Command;
use std::time::Instant;

const DECLARED_PUBLISHABLE_EMAIL_CONFIG_KEY: &str = "astronomical.publishableEmail";
const GITHUB_NOREPLY_DOMAIN: &str = "users.noreply.github.com";
const GITHUB_OWNED_DOMAIN: &str = "github.com";
const SERVICE_NOREPLY_LOCAL_PART: &str = "noreply";
const ABSENT_OBJECT_ID_DIGIT: char = '0';
const PROGRESS_INTERVAL_SECONDS: u64 = 5;
// RFC 2606 / RFC 6761 reserve these; mail to them is never delivered, so a
// fixture identity under them cannot identify a person.
const RESERVED_TOP_LEVEL_DOMAINS: [&str; 5] = ["invalid", "test", "example", "localhost", "local"];

/// True when `email` is safe to publish in commit metadata.
///
/// `declared_publishable_emails` are addresses the developer has already
/// made public and listed in their local git config.
#[must_use]
pub fn is_publishable_identity_email(email: &str, declared_publishable_emails: &[String]) -> bool {
    let normalized_email = email.trim().to_ascii_lowercase();
    if is_declared_publishable_email(&normalized_email, declared_publishable_emails) {
        return true;
    }
    let Some((local_part, domain)) = normalized_email.rsplit_once('@') else {
        return false;
    };
    if local_part.is_empty() || domain.is_empty() {
        return false;
    }
    if domain == GITHUB_NOREPLY_DOMAIN {
        return is_numeric_github_noreply_local_part(local_part);
    }
    if domain == GITHUB_OWNED_DOMAIN || local_part == SERVICE_NOREPLY_LOCAL_PART {
        return true;
    }
    is_reserved_top_level_domain(domain)
}

/// `<digits>+<login>`: GitHub resolves this by account id, so it cannot be
/// credited to a stranger who happens to own the login.
fn is_numeric_github_noreply_local_part(local_part: &str) -> bool {
    let Some((account_id, login)) = local_part.split_once('+') else {
        return false;
    };
    !account_id.is_empty() && is_all_ascii_digits(account_id) && !login.is_empty()
}

fn is_all_ascii_digits(text: &str) -> bool {
    for text_byte in text.bytes() {
        if !text_byte.is_ascii_digit() {
            return false;
        }
    }
    true
}

fn is_declared_publishable_email(
    normalized_email: &str,
    declared_publishable_emails: &[String],
) -> bool {
    for declared_email in declared_publishable_emails {
        if declared_email.trim().eq_ignore_ascii_case(normalized_email) {
            return true;
        }
    }
    false
}

fn is_reserved_top_level_domain(domain: &str) -> bool {
    let top_level_domain = domain.rsplit('.').next().unwrap_or(domain);
    RESERVED_TOP_LEVEL_DOMAINS.contains(&top_level_domain)
}

/// The email inside the first `<...>` of an identity line, if any.
#[must_use]
pub fn email_in_angle_brackets(identity_text: &str) -> Option<&str> {
    let (_, after_open) = identity_text.split_once('<')?;
    let (email, _) = after_open.split_once('>')?;
    Some(email.trim())
}

/// Every identity email a raw commit object would publish: the author and
/// committer headers plus each `Co-authored-by` trailer in the message.
#[must_use]
pub fn identity_emails_in_raw_commit(raw_commit_text: &str) -> Vec<String> {
    let mut identity_emails = Vec::new();
    let (header_text, message_text) = raw_commit_text
        .split_once("\n\n")
        .unwrap_or((raw_commit_text, ""));
    for header_line in header_text.lines() {
        let is_identity_header =
            header_line.starts_with("author ") || header_line.starts_with("committer ");
        if is_identity_header && let Some(email) = email_in_angle_brackets(header_line) {
            identity_emails.push(email.to_owned());
        }
    }
    for message_line in message_text.lines() {
        let is_co_author_trailer = message_line
            .trim_start()
            .to_ascii_lowercase()
            .starts_with("co-authored-by:");
        if is_co_author_trailer && let Some(email) = email_in_angle_brackets(message_line) {
            identity_emails.push(email.to_owned());
        }
    }
    identity_emails
}

/// The identity emails in `raw_commit_text` that must not be published.
#[must_use]
pub fn unpublishable_identity_emails(
    raw_commit_text: &str,
    declared_publishable_emails: &[String],
) -> Vec<String> {
    let mut unpublishable_emails = Vec::new();
    for identity_email in identity_emails_in_raw_commit(raw_commit_text) {
        if !is_publishable_identity_email(&identity_email, declared_publishable_emails) {
            unpublishable_emails.push(identity_email);
        }
    }
    unpublishable_emails
}

/// The `git rev-list` arguments selecting what one pre-push line publishes.
///
/// `None` for a branch deletion, which publishes nothing. A new remote ref
/// lists only commits no remote-tracking ref already has, so pushing a fresh
/// branch does not re-scan the whole history that is already public.
#[must_use]
pub fn rev_list_arguments_for_pre_push_line(pre_push_line: &str) -> Option<Vec<String>> {
    let pre_push_fields: Vec<&str> = pre_push_line.split_whitespace().collect();
    let [_, local_object_id, _, remote_object_id] = pre_push_fields.as_slice() else {
        return None;
    };
    if is_absent_object_id(local_object_id) {
        return None;
    }
    if is_absent_object_id(remote_object_id) {
        return Some(vec![
            (*local_object_id).to_owned(),
            "--not".to_owned(),
            "--remotes".to_owned(),
        ]);
    }
    Some(vec![format!("{remote_object_id}..{local_object_id}")])
}

fn is_absent_object_id(object_id: &str) -> bool {
    if object_id.is_empty() {
        return false;
    }
    for object_id_digit in object_id.chars() {
        if object_id_digit != ABSENT_OBJECT_ID_DIGIT {
            return false;
        }
    }
    true
}

/// The pre-commit hint, shared by both modes so the remedy reads the same.
#[must_use]
pub fn publishable_identity_hint() -> &'static str {
    "[commit-identity] hint=use your numeric GitHub no-reply address: git config user.email '<id>+<login>@users.noreply.github.com' (GitHub Settings -> Emails shows it), or declare an address you have already made public: git config --add astronomical.publishableEmail <address>"
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    match arguments.first().map(String::as_str) {
        Some("--staged") => run_staged_mode(),
        Some("--pre-push") => run_pre_push_mode(),
        _ => Err("usage: cargo xtask commit-identity --staged | --pre-push".to_owned()),
    }
}

/// The developer's locally declared publishable addresses; none when unset.
fn declared_publishable_emails() -> Vec<String> {
    let mut declared_emails = Vec::new();
    let config_output = Command::new("git")
        .args(["config", "--get-all", DECLARED_PUBLISHABLE_EMAIL_CONFIG_KEY])
        .output();
    // git exits 1 when the key is unset, which simply means nothing declared.
    if let Ok(config_output) = config_output {
        for declared_line in String::from_utf8_lossy(&config_output.stdout).lines() {
            if !declared_line.trim().is_empty() {
                declared_emails.push(declared_line.trim().to_owned());
            }
        }
    }
    declared_emails
}

fn run_staged_mode() -> Result<(), String> {
    let declared_emails = declared_publishable_emails();
    let mut unpublishable_emails = Vec::new();
    for identity_variable in ["GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"] {
        let identity_line = git_stdout(&["var", identity_variable])?;
        let identity_email = email_in_angle_brackets(&identity_line).unwrap_or("");
        if !is_publishable_identity_email(identity_email, &declared_emails) {
            unpublishable_emails.push(format!("{identity_variable}={identity_email}"));
        }
    }
    if unpublishable_emails.is_empty() {
        println!("[commit-identity] status=success mode=staged");
        return Ok(());
    }
    Err(format!(
        "[commit-identity] status=failed mode=staged identities={}\n{}",
        unpublishable_emails.join(","),
        publishable_identity_hint()
    ))
}

fn run_pre_push_mode() -> Result<(), String> {
    let mut pre_push_input = String::new();
    if let Err(read_error) = std::io::stdin().read_to_string(&mut pre_push_input) {
        return Err(format!("[commit-identity] status=failed reason=stdin {read_error}"));
    }
    let mut commit_object_ids = Vec::new();
    for pre_push_line in pre_push_input.lines() {
        if let Some(rev_list_arguments) = rev_list_arguments_for_pre_push_line(pre_push_line) {
            let mut git_arguments = vec!["rev-list".to_owned()];
            git_arguments.extend(rev_list_arguments);
            let listed_commits = git_stdout_owned(&git_arguments)?;
            for commit_object_id in listed_commits.lines() {
                commit_object_ids.push(commit_object_id.to_owned());
            }
        }
    }
    commit_object_ids.sort_unstable();
    commit_object_ids.dedup();
    check_pushed_commits(&commit_object_ids, &declared_publishable_emails())
}

fn check_pushed_commits(
    commit_object_ids: &[String],
    declared_emails: &[String],
) -> Result<(), String> {
    let started_at = Instant::now();
    let mut last_report_seconds = 0;
    let mut rejected_commit_lines = Vec::new();
    for (commit_position, commit_object_id) in commit_object_ids.iter().enumerate() {
        let elapsed_seconds = started_at.elapsed().as_secs();
        if elapsed_seconds >= last_report_seconds + PROGRESS_INTERVAL_SECONDS {
            last_report_seconds = elapsed_seconds;
            report_progress(commit_position, commit_object_ids.len(), elapsed_seconds);
        }
        let raw_commit_text = git_stdout(&["cat-file", "commit", commit_object_id])?;
        let unpublishable_emails = unpublishable_identity_emails(&raw_commit_text, declared_emails);
        if !unpublishable_emails.is_empty() {
            rejected_commit_lines.push(format!(
                "[commit-identity] status=rejected commit={commit_object_id} emails={}",
                unpublishable_emails.join(",")
            ));
        }
    }
    if rejected_commit_lines.is_empty() {
        println!(
            "[commit-identity] status=success mode=pre-push commits={}",
            commit_object_ids.len()
        );
        return Ok(());
    }
    Err(format!(
        "{}\n[commit-identity] status=failed mode=pre-push rejected_commits={}\n{}\n[commit-identity] hint=rewrite those commits (git rebase with --exec 'git commit --amend --no-edit --reset-author') before pushing",
        rejected_commit_lines.join("\n"),
        rejected_commit_lines.len(),
        publishable_identity_hint()
    ))
}

fn report_progress(checked_commits: usize, total_commits: usize, elapsed_seconds: u64) {
    let percentage = checked_commits * 100 / total_commits.max(1);
    let eta_seconds = if checked_commits == 0 {
        0
    } else {
        (elapsed_seconds * total_commits as u64).div_ceil(checked_commits as u64) - elapsed_seconds
    };
    println!(
        "[commit-identity] status=progress commits={checked_commits}/{total_commits} percentage={percentage} ETA_seconds={eta_seconds} elapsed_seconds={elapsed_seconds}"
    );
}

fn git_stdout(git_arguments: &[&str]) -> Result<String, String> {
    let mut owned_arguments = Vec::with_capacity(git_arguments.len());
    for git_argument in git_arguments {
        owned_arguments.push((*git_argument).to_owned());
    }
    git_stdout_owned(&owned_arguments)
}

fn git_stdout_owned(git_arguments: &[String]) -> Result<String, String> {
    let git_output = match Command::new("git").args(git_arguments).output() {
        Ok(git_output) => git_output,
        Err(spawn_error) => {
            return Err(format!("[commit-identity] status=failed reason=git {spawn_error}"));
        }
    };
    if !git_output.status.success() {
        return Err(format!(
            "[commit-identity] status=failed reason=git-exit arguments={}",
            git_arguments.join(" ")
        ));
    }
    Ok(String::from_utf8_lossy(&git_output.stdout).into_owned())
}
