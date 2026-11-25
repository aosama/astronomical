//! The command line of `cargo xtask measure-speed`.

use std::process::Command;

use crate::small_mac_matrix::matrix_cells::DEVELOPMENT_BASE_URL;

const DEFAULT_ROUND_COUNT: usize = 3;
/// Temperature people chat at; never greedy for figures people read.
const DEFAULT_MEASUREMENT_TEMPERATURE: f64 = 0.6;
/// The widest temperature a chat request accepts.
const MAXIMUM_MEASUREMENT_TEMPERATURE: f64 = 2.0;
const BYTES_PER_GIGABYTE: u64 = 1_073_741_824;

#[derive(Debug, PartialEq)]
pub struct MeasureSpeedOptions {
    pub base_url: String,
    pub round_count: usize,
    /// Sampling temperature of every measured reply: 0.6 by default, 0.7
    /// for coding work and 1.0 for general work when a measurement asks.
    pub temperature: f64,
    /// Where the figures were measured, when given; `run` reads this Mac
    /// otherwise, so parsing never touches the hardware.
    pub measured_on: Option<String>,
    pub should_write_catalog: bool,
    pub model_ids: Vec<String>,
}

impl MeasureSpeedOptions {
    pub fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut parsed_options = Self {
            base_url: DEVELOPMENT_BASE_URL.to_owned(),
            round_count: DEFAULT_ROUND_COUNT,
            temperature: DEFAULT_MEASUREMENT_TEMPERATURE,
            measured_on: None,
            should_write_catalog: false,
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
                flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
                model_id => parsed_options.model_ids.push(model_id.to_owned()),
            }
        }
        if parsed_options.model_ids.is_empty() {
            return Err("usage: cargo xtask measure-speed [--base-url URL] [--rounds N] [--temperature T] [--measured-on TEXT] [--write-catalog] MODEL_ID...".to_owned());
        }
        Ok(parsed_options)
    }

    fn apply_valued_option(&mut self, option_name: &str, value: String) -> Result<(), String> {
        match option_name {
            "--base-url" => self.base_url = value,
            "--rounds" => self.round_count = parse_round_count(&value)?,
            "--temperature" => self.temperature = parse_temperature(&value)?,
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

/// Three rounds give a median a single slow round cannot move; fewer are
/// refused because the catalog would keep a figure one round set.
fn parse_round_count(round_text: &str) -> Result<usize, String> {
    match round_text.parse::<usize>() {
        Ok(round_count) if round_count >= DEFAULT_ROUND_COUNT => Ok(round_count),
        _ => Err(format!(
            "--rounds needs a whole number of at least {DEFAULT_ROUND_COUNT}, got {round_text}"
        )),
    }
}

/// "Mac with an M5 Pro and 48 GB", read from this Mac, so the figures always
/// say where they were measured without anyone typing it.
pub(crate) fn this_mac_description() -> String {
    let chip_name = sysctl_value("machdep.cpu.brand_string").map_or_else(
        || "Mac".to_owned(),
        |brand| brand.trim_start_matches("Apple ").to_owned(),
    );
    let memory_gigabytes = sysctl_value("hw.memsize")
        .and_then(|memory_text| memory_text.parse::<u64>().ok())
        .map_or(0, |memory_bytes| memory_bytes / BYTES_PER_GIGABYTE);
    describe_mac(&chip_name, memory_gigabytes)
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
