use super::{MetalKernelCheckInvocation, parse_arguments};

fn arguments(words: &[&str]) -> Vec<String> {
    let mut arguments = Vec::new();
    for word in words {
        arguments.push((*word).to_owned());
    }
    arguments
}

#[test]
fn should_compile_only_what_changed_when_given_no_arguments() {
    assert_eq!(
        parse_arguments(&[]),
        Ok(MetalKernelCheckInvocation::default())
    );
}

#[test]
fn should_read_fresh_and_skip_compile_together() {
    assert_eq!(
        parse_arguments(&arguments(&["--skip-compile", "--fresh"])),
        Ok(MetalKernelCheckInvocation {
            is_fresh_run: true,
            skips_compile: true,
        })
    );
}

#[test]
fn should_refuse_an_unknown_argument_and_name_it() {
    let refusal = parse_arguments(&arguments(&["--fast"])).expect_err("an unknown flag is refused");
    assert!(refusal.contains("--fast"), "{refusal}");
    assert!(refusal.contains("usage"), "{refusal}");
}
