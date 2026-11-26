use super::cargo_citations::{
    CitedCommand, alias_names, citations_in_code, cited_commands, inline_code_spans, is_citable,
    run, violation_line,
};

const CONFIG_WITH_ALIASES: &str = "[build]\nwarnings = \"deny\"\n\n[alias]\n# a comment = \"no\"\nverify = \"check --workspace\"\nmeasure-speed = \"run -p xtask\"\nmeasure-memory = \"run -p xtask\"\n\n[env]\nnot-an-alias = \"1\"\n";

fn exact(command_word: &str) -> CitedCommand {
    CitedCommand::Exact(command_word.to_owned())
}

fn alias_family(alias_prefix: &str) -> CitedCommand {
    CitedCommand::AliasFamily(alias_prefix.to_owned())
}

fn repository_aliases() -> Vec<String> {
    alias_names(CONFIG_WITH_ALIASES)
}

#[test]
fn should_read_only_the_keys_of_the_alias_table() {
    assert_eq!(
        repository_aliases(),
        vec!["verify", "measure-speed", "measure-memory"]
    );
}

#[test]
fn should_read_no_aliases_from_a_config_without_an_alias_table() {
    assert!(alias_names("[build]\nwarnings = \"deny\"\n").is_empty());
}

#[test]
fn should_accept_a_cited_alias_and_a_stock_subcommand() {
    assert!(is_citable(&exact("verify"), &repository_aliases()));
    assert!(is_citable(&exact("build"), &repository_aliases()));
    assert!(is_citable(&exact("clippy"), &repository_aliases()));
}

#[test]
fn should_refuse_a_retired_alias() {
    assert!(!is_citable(
        &exact("retired-fixture-alias"),
        &repository_aliases()
    ));
}

#[test]
fn should_accept_an_alias_family_that_names_at_least_one_alias() {
    assert!(is_citable(&alias_family("measure-"), &repository_aliases()));
    assert!(is_citable(&alias_family("measure"), &repository_aliases()));
}

#[test]
fn should_refuse_an_alias_family_no_alias_belongs_to() {
    assert!(!is_citable(
        &alias_family("retired-"),
        &repository_aliases()
    ));
}

#[test]
fn should_read_every_placeholder_form_as_an_alias_family() {
    for placeholder_citation in ["cargo measure-...", "cargo measure-*", "cargo measure-…"] {
        assert_eq!(
            citations_in_code(placeholder_citation),
            vec![alias_family("measure-")],
            "{placeholder_citation}"
        );
    }
}

#[test]
fn should_cite_commands_in_fences_and_inline_spans_but_not_in_text() {
    let markdown_text = "Text mentions cargo textword without a span.\n\n```sh\ncargo build --release\ncargo retired-fixture-alias --flag\n```\n\nRun `cargo verify` before pushing, then `cargo measure-...`.\n";
    assert_eq!(
        cited_commands(markdown_text),
        vec![
            exact("build"),
            exact("retired-fixture-alias"),
            exact("verify"),
            alias_family("measure-"),
        ]
    );
}

#[test]
fn should_read_lines_after_an_unclosed_fence_as_code() {
    assert_eq!(cited_commands("```\ncargo build\n"), vec![exact("build")]);
}

#[test]
fn should_cite_nothing_where_cargo_does_not_start_a_word() {
    for not_a_citation in [
        "xcargo build",
        "~/.cargo build",
        "my-cargo build",
        "path/cargo run",
    ] {
        assert!(
            citations_in_code(not_a_citation).is_empty(),
            "{not_a_citation}"
        );
    }
}

#[test]
fn should_skip_flags_single_letters_and_trailing_hyphens() {
    assert!(citations_in_code("cargo --list").is_empty());
    assert!(citations_in_code("cargo +nightly fmt").is_empty());
    assert!(citations_in_code("cargo x").is_empty());
    assert_eq!(
        citations_in_code("cargo verify- now"),
        vec![exact("verify")]
    );
}

#[test]
fn should_find_several_citations_in_one_span() {
    assert_eq!(
        citations_in_code("cargo fmt && cargo verify"),
        vec![exact("fmt"), exact("verify")]
    );
}

#[test]
fn should_find_each_closed_inline_span_and_ignore_an_unclosed_one() {
    assert_eq!(
        inline_code_spans("a `one` b `` c `two` d `open"),
        vec!["one", "two"]
    );
    assert!(inline_code_spans("no spans here").is_empty());
}

#[test]
fn should_name_the_file_and_the_unresolved_citation_in_each_violation() {
    assert_eq!(
        violation_line("docs/a.md", &exact("retired-thing")),
        "docs/a.md cites unknown cargo command \"retired-thing\""
    );
    assert_eq!(
        violation_line("docs/a.md", &alias_family("nosuch-")),
        "docs/a.md cites cargo alias family \"nosuch-...\" but no alias starts with \"nosuch-\""
    );
}

#[test]
fn should_refuse_arguments() {
    let sandbox = tempfile::tempdir().expect("a sandbox should be creatable");
    assert_eq!(
        run(sandbox.path(), &["--fix".to_owned()]),
        Err("usage: cargo xtask check-cargo-citations".to_owned())
    );
}
