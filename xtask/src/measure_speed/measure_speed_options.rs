//! The command line of `cargo xtask measure-speed`.

use std::process::Command;

use crate::release_speed::speed_ledger::MeasuringMachine;
use crate::small_mac_matrix::matrix_cells::DEVELOPMENT_BASE_URL;

use super::case_time_budget::MOST_ROUNDS_A_CASE;

/// The widest temperature a chat request accepts.
const MAXIMUM_MEASUREMENT_TEMPERATURE: f64 = 2.0;
const BYTES_PER_GIGABYTE: u64 = 1_073_741_824;
const MEASURE_SPEED_USAGE: &str = "usage: cargo xtask measure-speed [--base-url URL] [--rounds N] [--temperature T] [--measured-on TEXT] [--reference-set] [--write-ledger] [--write-catalog] [MODEL_ID...]";

#[derive(Debug, PartialEq)]
pub struct MeasureSpeedOptions {
    pub base_url: String,
    /// The most rounds a case runs; `case_time_budget` may stop it sooner.
    pub round_count: usize,
    /// One sampling temperature for every reply, when given; each task's own
    /// otherwise (`SpeedTask::default_temperature`). A release ledger is only
    /// written at the tasks' own temperatures, so releases stay comparable.
    pub temperature_override: Option<f64>,
    /// Where the figures were measured, when given; `run` reads this Mac
    /// otherwise, so parsing never touches the hardware.
    pub measured_on: Option<String>,
    pub should_write_catalog: bool,
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
            round_count: MOST_ROUNDS_A_CASE,
            temperature_override: None,
            measured_on: None,
            should_write_catalog: false,
            should_use_reference_set: false,
            should_write_ledger: false,
            model_ids: Vec::new(),
        };
        let mut remaining_arguments = arguments.iter();
        while let Some(argument) = remaining_arguments.next() {
            match argument.as_str() {
                "--base-url" | "--rounds" | "--measured-on" | "--temperature" => {
                    let value = option_value(&mut remaining_arguments, argument)?;
                    parsed_options.apply_valued_option(argument, value)?;
                }
                "--write-catalog" => parsed_options.should_write_catalog = true,
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
            "--rounds" => self.round_count = parse_round_count(&value)?,
            "--temperature" => self.temperature_override = Some(parse_temperature(&value)?),
            _ => self.measured_on = Some(value),
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

/// The most rounds a case may run; its time budget may stop it sooner, and
/// one round always runs.
fn parse_round_count(round_text: &str) -> Result<usize, String> {
    match round_text.parse::<usize>() {
        Ok(round_count) if round_count >= 1 => Ok(round_count),
        _ => Err(format!(
            "--rounds needs a whole number of at least 1, got {round_text}"
        )),
    }
}

/// "Mac with an M5 Pro and 48 GB", read from this Mac, so the figures always
/// say where they were measured without anyone typing it.
pub(crate) fn this_mac_description() -> String {
    let this_mac = this_mac();
    describe_mac(&this_mac.chip, this_mac.memory_bytes / BYTES_PER_GIGABYTE)
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
