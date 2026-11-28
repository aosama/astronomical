//! The command line of `cargo xtask measure-speed`.

use std::process::Command;

use crate::release_speed::speed_ledger::MeasuringMachine;
use crate::small_mac_matrix::matrix_cells::DEVELOPMENT_BASE_URL;

/// The widest temperature a chat request accepts.
const MAXIMUM_MEASUREMENT_TEMPERATURE: f64 = 2.0;
const MEASURE_SPEED_USAGE: &str = "usage: cargo xtask measure-speed [--base-url URL] [--temperature T] [--reference-set] [--write-ledger] [MODEL_ID...]";

#[derive(Debug, PartialEq)]
pub struct MeasureSpeedOptions {
    pub base_url: String,
    /// One sampling temperature for every reply, when given; each task's own
    /// otherwise (`SpeedTask::default_temperature`). A release ledger is only
    /// written at the tasks' own temperatures, so releases stay comparable.
    pub temperature_override: Option<f64>,
    /// Measure the cases of `registry/release_speed/reference_set.json`,
    /// narrowed to the named models when any are named.
    pub should_use_reference_set: bool,
    pub should_write_ledger: bool,
    pub model_ids: Vec<String>,
}

impl MeasureSpeedOptions {
    pub fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut parsed_options = Self {
            base_url: DEVELOPMENT_BASE_URL.to_owned(),
            temperature_override: None,
            should_use_reference_set: false,
            should_write_ledger: false,
            model_ids: Vec::new(),
        };
        let mut remaining_arguments = arguments.iter();
        while let Some(argument) = remaining_arguments.next() {
            match argument.as_str() {
                "--base-url" | "--temperature" => {
                    let value = option_value(&mut remaining_arguments, argument)?;
                    parsed_options.apply_valued_option(argument, value)?;
                }
                "--reference-set" => parsed_options.should_use_reference_set = true,
                "--write-ledger" => parsed_options.should_write_ledger = true,
                flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
                model_id => parsed_options.model_ids.push(model_id.to_owned()),
            }
        }
        parsed_options.check_combination()?;
        Ok(parsed_options)
    }

    /// A ledger holds only reference cases at the tasks' own temperatures:
    /// anything else could not be compared with the next release.
    fn check_combination(&self) -> Result<(), String> {
        if self.model_ids.is_empty() && !self.should_use_reference_set {
            return Err(MEASURE_SPEED_USAGE.to_owned());
        }
        if self.should_write_ledger && !self.should_use_reference_set {
            return Err(
                "--write-ledger needs --reference-set: a release ledger holds only reference cases"
                    .to_owned(),
            );
        }
        if self.should_write_ledger && self.temperature_override.is_some() {
            return Err(
                "--write-ledger measures at each task's own temperature; leave out --temperature"
                    .to_owned(),
            );
        }
        Ok(())
    }

    fn apply_valued_option(&mut self, option_name: &str, value: String) -> Result<(), String> {
        match option_name {
            "--base-url" => self.base_url = value,
            "--temperature" => self.temperature_override = Some(parse_temperature(&value)?),
            _ => return Err(format!("unknown option {option_name}")),
        }
        Ok(())
    }
}

/// A temperature above zero: greedy decoding is a diagnostic, not a speed
/// people see, and the chat API takes nothing above 2.
fn parse_temperature(temperature_text: &str) -> Result<f64, String> {
    match temperature_text.parse::<f64>() {
        Ok(temperature) if temperature > 0.0 && temperature <= MAXIMUM_MEASUREMENT_TEMPERATURE => {
            Ok(temperature)
        }
        _ => Err(format!(
            "--temperature needs a number above 0 and at most {MAXIMUM_MEASUREMENT_TEMPERATURE}, got {temperature_text}"
        )),
    }
}

fn option_value(
    remaining_arguments: &mut std::slice::Iter<'_, String>,
    option_name: &str,
) -> Result<String, String> {
    remaining_arguments
        .next()
        .cloned()
        .ok_or_else(|| format!("{option_name} needs a value"))
}

/// This Mac's chip ("M5 Pro") and memory, the only facts about the machine a
/// measurement records.
pub(crate) fn this_mac() -> MeasuringMachine {
    MeasuringMachine {
        chip: sysctl_value("machdep.cpu.brand_string").map_or_else(
            || "Mac".to_owned(),
            |brand| brand.trim_start_matches("Apple ").to_owned(),
        ),
        memory_bytes: sysctl_value("hw.memsize")
            .and_then(|memory_text| memory_text.parse::<u64>().ok())
            .unwrap_or(0),
    }
}

pub(crate) fn describe_mac(chip_name: &str, memory_gigabytes: u64) -> String {
    let article = if starts_with_vowel_sound(chip_name) {
        "an"
    } else {
        "a"
    };
    format!("Mac with {article} {chip_name} and {memory_gigabytes} GB")
}

/// "an M5 Pro" but "a Mac": a chip name read letter by letter ("M5") takes
/// "an" when its first letter's name starts with a vowel sound, as does a
/// word starting with a vowel.
fn starts_with_vowel_sound(chip_name: &str) -> bool {
    let mut name_characters = chip_name.chars();
    let Some(first_character) = name_characters.next() else {
        return false;
    };
    let is_spelled_out = name_characters
        .next()
        .is_some_and(|second_character| second_character.is_ascii_digit());
    matches!(first_character, 'A' | 'E' | 'I' | 'O')
        || (is_spelled_out
            && matches!(
                first_character,
                'F' | 'H' | 'L' | 'M' | 'N' | 'R' | 'S' | 'X'
            ))
}

fn sysctl_value(sysctl_name: &str) -> Option<String> {
    let sysctl_output = Command::new("sysctl")
        .args(["-n", sysctl_name])
        .output()
        .ok()?;
    sysctl_output.status.success().then(|| {
        String::from_utf8_lossy(&sysctl_output.stdout)
            .trim()
            .to_owned()
    })
}

#[cfg(test)]
#[path = "measure_speed_options_tests.rs"]
mod measure_speed_options_tests;
