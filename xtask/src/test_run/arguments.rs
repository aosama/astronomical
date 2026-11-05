use std::path::PathBuf;
use std::time::Duration;

pub(super) const USAGE: &str = "cargo xtask test-run --package NAME (--test TARGET | --lib | --tests) [--features FEATURES] [--release | --profile NAME] [--configuration PATH] [--filter NAME] [--exact] [--ignored] [--case-budget-seconds SECONDS]";

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
    /// Replaces the default per-case budget for journeys whose measured work is
    /// inherently long, such as loading every program of a 64-layer model.
    pub case_budget: Option<Duration>,
}

/// Selections collected before the request is validated as a whole, so parsing
/// one option and checking how options combine stay separate jobs.
#[derive(Default)]
struct TestRunSelections {
    package: Option<String>,
    target: Option<TestTarget>,
    features: Option<String>,
    profile: Option<String>,
    configuration_path: Option<PathBuf>,
    filter: Option<String>,
    exact: bool,
    ignored: bool,
    case_budget: Option<Duration>,
}

impl TestRunSelections {
    fn apply_option<'a>(
        &mut self,
        argument: &str,
        remaining_arguments: &mut impl Iterator<Item = &'a String>,
    ) -> Result<(), String> {
        match argument {
            "--package" => set_once(
                &mut self.package,
                next_text(remaining_arguments, argument)?,
                argument,
            ),
            "--features" => set_once(
                &mut self.features,
                next_text(remaining_arguments, argument)?,
                argument,
            ),
            "--configuration" => set_once(
                &mut self.configuration_path,
                PathBuf::from(next_text(remaining_arguments, argument)?),
                argument,
            ),
            "--filter" => set_once(
                &mut self.filter,
                next_text(remaining_arguments, argument)?,
                argument,
            ),
            "--case-budget-seconds" => set_once(
                &mut self.case_budget,
                parse_case_budget(&next_text(remaining_arguments, argument)?)?,
                argument,
            ),
            _ => self.apply_target_or_profile(argument, remaining_arguments),
        }
    }

    fn apply_target_or_profile<'a>(
        &mut self,
        argument: &str,
        remaining_arguments: &mut impl Iterator<Item = &'a String>,
    ) -> Result<(), String> {
        match argument {
            "--test" => set_once(
                &mut self.target,
                TestTarget::Integration(next_text(remaining_arguments, argument)?),
                argument,
            ),
            "--lib" => set_once(&mut self.target, TestTarget::Library, argument),
            "--tests" => set_once(&mut self.target, TestTarget::AllTests, argument),
            "--release" => set_once(&mut self.profile, "release".to_owned(), argument),
            "--profile" => set_once(
                &mut self.profile,
                next_text(remaining_arguments, argument)?,
                argument,
            ),
            "--exact" if !self.exact => {
                self.exact = true;
                Ok(())
            }
            "--ignored" if !self.ignored => {
                self.ignored = true;
                Ok(())
            }
            _ => Err(format!("unknown or repeated test-run argument: {argument}")),
        }
    }

    fn into_request(self) -> Result<TestRunRequest, String> {
        let Some(package) = self.package else {
            return Err("--package is required".to_owned());
        };
        let Some(target) = self.target else {
            return Err("select exactly one of --test, --lib, or --tests".to_owned());
        };
        if self.exact && self.filter.is_none() {
            return Err("--exact requires --filter".to_owned());
        }
        if self.ignored
            && matches!(&target, TestTarget::Integration(name) if name.starts_with("performance"))
            && self.profile.as_deref() != Some("release")
        {
            return Err("performance measurements require --release".to_owned());
        }
        Ok(TestRunRequest {
            package,
            target,
            features: self.features,
            profile: self.profile,
            configuration_path: self.configuration_path,
            filter: self.filter,
            exact: self.exact,
            ignored: self.ignored,
            case_budget: self.case_budget,
        })
    }
}

impl TestRunRequest {
    pub fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut selections = TestRunSelections::default();
        let mut remaining_arguments = arguments.iter();
        while let Some(argument) = remaining_arguments.next() {
            selections.apply_option(argument, &mut remaining_arguments)?;
        }
        selections.into_request()
    }

    pub fn effective_case_budget(&self, default_budget: Duration) -> Duration {
        self.case_budget.unwrap_or(default_budget)
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

fn parse_case_budget(seconds_text: &str) -> Result<Duration, String> {
    match seconds_text.parse::<u64>() {
        Ok(budget_seconds) if budget_seconds > 0 => Ok(Duration::from_secs(budget_seconds)),
        _ => Err(format!(
            "--case-budget-seconds requires a positive whole number, got {seconds_text}"
        )),
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
