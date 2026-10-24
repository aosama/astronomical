use std::path::PathBuf;

pub(super) const USAGE: &str = "cargo xtask test-run --package NAME (--test TARGET | --lib | --tests) [--features FEATURES] [--release | --profile NAME] [--configuration PATH] [--filter NAME] [--exact] [--ignored]";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum TestTarget {
    Library,
    Integration(String),
    AllTests,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TestRunRequest {
    pub package: String,
    pub target: TestTarget,
    pub features: Option<String>,
    pub profile: Option<String>,
    pub configuration_path: Option<PathBuf>,
    pub filter: Option<String>,
    pub exact: bool,
    pub ignored: bool,
}

impl TestRunRequest {
    pub fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut package = None;
        let mut target = None;
        let mut features = None;
        let mut profile = None;
        let mut configuration_path = None;
        let mut filter = None;
        let mut exact = false;
        let mut ignored = false;
        let mut remaining_arguments = arguments.iter();
        while let Some(argument) = remaining_arguments.next() {
            match argument.as_str() {
                "--package" => set_once(
                    &mut package,
                    next_text(&mut remaining_arguments, argument)?,
                    argument,
                )?,
                "--test" => set_once(
                    &mut target,
                    TestTarget::Integration(next_text(&mut remaining_arguments, argument)?),
                    argument,
                )?,
                "--lib" => set_once(&mut target, TestTarget::Library, argument)?,
                "--tests" => set_once(&mut target, TestTarget::AllTests, argument)?,
                "--features" => set_once(
                    &mut features,
                    next_text(&mut remaining_arguments, argument)?,
                    argument,
                )?,
                "--release" => set_once(&mut profile, "release".to_owned(), argument)?,
                "--profile" => set_once(
                    &mut profile,
                    next_text(&mut remaining_arguments, argument)?,
                    argument,
                )?,
                "--configuration" => set_once(
                    &mut configuration_path,
                    PathBuf::from(next_text(&mut remaining_arguments, argument)?),
                    argument,
                )?,
                "--filter" => set_once(
                    &mut filter,
                    next_text(&mut remaining_arguments, argument)?,
                    argument,
                )?,
                "--exact" if !exact => exact = true,
                "--ignored" if !ignored => ignored = true,
                _ => return Err(format!("unknown or repeated test-run argument: {argument}")),
            }
        }
        let package = package.ok_or_else(|| "--package is required".to_owned())?;
        let target =
            target.ok_or_else(|| "select exactly one of --test, --lib, or --tests".to_owned())?;
        if exact && filter.is_none() {
            return Err("--exact requires --filter".to_owned());
        }
        if ignored
            && matches!(&target, TestTarget::Integration(name) if name.starts_with("performance"))
            && profile.as_deref() != Some("release")
        {
            return Err("performance measurements require --release".to_owned());
        }
        Ok(Self {
            package,
            target,
            features,
            profile,
            configuration_path,
            filter,
            exact,
            ignored,
        })
    }

    pub fn append_profile_arguments(&self, cargo_arguments: &mut Vec<String>) {
        if let Some(profile) = &self.profile {
            cargo_arguments.extend(["--profile".to_owned(), profile.clone()]);
        }
    }

    pub fn listing_arguments(&self, ignored_only: bool) -> Vec<String> {
        let mut test_arguments = vec![
            "--list".to_owned(),
            "--format".to_owned(),
            "terse".to_owned(),
        ];
        if let Some(filter) = &self.filter {
            test_arguments.push(filter.clone());
        }
        if self.exact {
            test_arguments.push("--exact".to_owned());
        }
        if ignored_only {
            test_arguments.push("--ignored".to_owned());
        }
        test_arguments
    }
}

fn next_text<'a>(
    remaining_arguments: &mut impl Iterator<Item = &'a String>,
    option_name: &str,
) -> Result<String, String> {
    remaining_arguments
        .next()
        .filter(|argument| !argument.is_empty() && !argument.starts_with('-'))
        .cloned()
        .ok_or_else(|| format!("{option_name} requires a nonempty argument"))
}

fn set_once<T>(destination: &mut Option<T>, selected: T, option_name: &str) -> Result<(), String> {
    if destination.is_some() {
        return Err(format!("{option_name} conflicts with an earlier selection"));
    }
    *destination = Some(selected);
    Ok(())
}

#[cfg(test)]
#[path = "arguments_tests.rs"]
mod arguments_tests;
