//! The repeated-block measure of issue #2473: a code line counts as repeated
//! when it sits in a run of at least [`REPEATED_BLOCK_MINIMUM_LINES`]
//! consecutive normalized code lines that also appears in another file.
//!
//! A run is found by its first [`REPEATED_BLOCK_MINIMUM_LINES`] lines: every
//! longer run is covered by the overlapping fixed-length blocks inside it, so
//! one fixed length marks exactly the lines of every run of that length or
//! more. Copies inside a single file are not counted; the measure is about
//! code that two places have to keep in step.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::source_normalization::{is_substantial_code_line, normalize_code_lines};

/// The shortest run of matching code lines that counts as a repeated block.
pub(crate) const REPEATED_BLOCK_MINIMUM_LINES: usize = 6;

/// A block needs this many substantial lines (see
/// `source_normalization::is_substantial_code_line`) to count, so runs of
/// closing brackets and one-word lines that every Rust file shares never do.
const MINIMUM_SUBSTANTIAL_LINES_PER_BLOCK: usize = REPEATED_BLOCK_MINIMUM_LINES - 2;

/// One source file, normalized and placed in its area.
pub(crate) struct NormalizedSourceFile {
    pub(crate) relative_path: String,
    pub(crate) area: String,
    pub(crate) code_lines: Vec<String>,
    /// Whether each code line, by index, is substantial.
    pub(crate) substantial_lines: Vec<bool>,
}

impl NormalizedSourceFile {
    pub(crate) fn from_source_text(relative_path: String, area: String, source_text: &str) -> Self {
        let code_lines = normalize_code_lines(source_text);
        let mut substantial_lines = Vec::with_capacity(code_lines.len());
        for code_line in &code_lines {
            substantial_lines.push(is_substantial_code_line(code_line));
        }
        Self {
            relative_path,
            area,
            code_lines,
            substantial_lines,
        }
    }
}

/// Two places that share repeated lines, with the count of lines on the side
/// that holds more of them. The two may be the same area, for copies between
/// files of one area.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RepeatedLinePair {
    pub(crate) first_place: String,
    pub(crate) second_place: String,
    pub(crate) repeated_line_count: usize,
}

/// What the measure found in one corpus. The pair lists are sorted with the
/// largest first.
#[derive(Debug, PartialEq)]
pub(crate) struct DuplicationMeasurement {
    pub(crate) file_count: usize,
    pub(crate) code_line_count: usize,
    pub(crate) repeated_line_count: usize,
    pub(crate) area_pairs: Vec<RepeatedLinePair>,
    pub(crate) file_pairs: Vec<RepeatedLinePair>,
}

impl DuplicationMeasurement {
    /// The share of code lines that are repeated, as a percentage; zero for
    /// an empty corpus.
    pub(crate) fn repeated_percentage(&self) -> f64 {
        if self.code_line_count == 0 {
            return 0.0;
        }
        self.repeated_line_count as f64 * 100.0 / self.code_line_count as f64
    }
}

/// Where one block starts: a file index and a code-line index in that file.
#[derive(Clone, Copy)]
struct BlockOccurrence {
    file_index: u32,
    first_line_index: u32,
}

/// For each ordered pair of files (A, B), the code-line indices in A that sit
/// in a block also found in B.
type LinesByFilePair = HashMap<(u32, u32), HashSet<u32>>;

/// Measures one corpus. Files in different corpora are never compared.
pub(crate) fn measure_repeated_blocks(
    source_files: &[NormalizedSourceFile],
) -> DuplicationMeasurement {
    let interned_files = intern_code_lines(source_files);
    let block_occurrences = index_substantial_blocks(source_files, &interned_files);
    let lines_by_file_pair = attribute_shared_lines(block_occurrences.values());
    let repeated_line_count = count_repeated_lines(&lines_by_file_pair);
    let mut file_paths: Vec<&str> = Vec::with_capacity(source_files.len());
    let mut code_line_count = 0;
    for source_file in source_files {
        file_paths.push(source_file.relative_path.as_str());
        code_line_count += source_file.code_lines.len();
    }
    let file_pair_counts = count_lines_by_file_pair(&lines_by_file_pair);
    DuplicationMeasurement {
        file_count: source_files.len(),
        code_line_count,
        repeated_line_count,
        area_pairs: summarize_area_pairs(source_files, &lines_by_file_pair),
        file_pairs: larger_direction_pairs(&file_pair_counts, &file_paths),
    }
}

/// Replaces each distinct code line with a small number, so a block becomes
/// a slice of six integers. Comparing and hashing those is far cheaper than
/// hashing six strings at every position, and unlike a digest of the text it
/// can never report two different blocks as one.
fn intern_code_lines(source_files: &[NormalizedSourceFile]) -> Vec<Vec<u32>> {
    let mut line_number_by_text: HashMap<&str, u32> = HashMap::new();
    let mut interned_files = Vec::with_capacity(source_files.len());
    for source_file in source_files {
        let mut interned_lines = Vec::with_capacity(source_file.code_lines.len());
        for code_line in &source_file.code_lines {
            let next_line_number = line_number_by_text.len() as u32;
            let line_number = *line_number_by_text
                .entry(code_line.as_str())
                .or_insert(next_line_number);
            interned_lines.push(line_number);
        }
        interned_files.push(interned_lines);
    }
    interned_files
}

/// Every block that has enough substantial lines, keyed by its interned
/// lines, with every place it starts.
fn index_substantial_blocks<'interned>(
    source_files: &[NormalizedSourceFile],
    interned_files: &'interned [Vec<u32>],
) -> HashMap<&'interned [u32], Vec<BlockOccurrence>> {
    let mut block_occurrences: HashMap<&[u32], Vec<BlockOccurrence>> = HashMap::new();
    for (file_index, interned_lines) in interned_files.iter().enumerate() {
        let substantial_lines = &source_files[file_index].substantial_lines;
        for first_line_index in substantial_block_starts(substantial_lines) {
            let block_lines =
                &interned_lines[first_line_index..first_line_index + REPEATED_BLOCK_MINIMUM_LINES];
            block_occurrences
                .entry(block_lines)
                .or_default()
                .push(BlockOccurrence {
                    file_index: file_index as u32,
                    first_line_index: first_line_index as u32,
                });
        }
    }
    block_occurrences
}

/// The start indices of every block of [`REPEATED_BLOCK_MINIMUM_LINES`]
/// lines that holds enough substantial lines. A running count slides along
/// the file, so each start costs two additions instead of six checks.
pub(crate) fn substantial_block_starts(substantial_lines: &[bool]) -> Vec<usize> {
    let mut block_starts = Vec::new();
    if substantial_lines.len() < REPEATED_BLOCK_MINIMUM_LINES {
        return block_starts;
    }
    let mut substantial_in_block = 0;
    for &is_substantial in &substantial_lines[..REPEATED_BLOCK_MINIMUM_LINES] {
        substantial_in_block += usize::from(is_substantial);
    }
    for first_line_index in 0..=substantial_lines.len() - REPEATED_BLOCK_MINIMUM_LINES {
        if first_line_index > 0 {
            let leaving_line = substantial_lines[first_line_index - 1];
            let entering_line =
                substantial_lines[first_line_index + REPEATED_BLOCK_MINIMUM_LINES - 1];
            substantial_in_block =
                substantial_in_block + usize::from(entering_line) - usize::from(leaving_line);
        }
        if substantial_in_block >= MINIMUM_SUBSTANTIAL_LINES_PER_BLOCK {
            block_starts.push(first_line_index);
        }
    }
    block_starts
}

/// For every block found in two or more files, records each of its lines
/// against every other file that holds the same block.
fn attribute_shared_lines<'block>(
    all_block_occurrences: impl Iterator<Item = &'block Vec<BlockOccurrence>>,
) -> LinesByFilePair {
    let mut lines_by_file_pair = LinesByFilePair::new();
    for block_occurrences in all_block_occurrences {
        let mut sharing_files: Vec<u32> = Vec::with_capacity(block_occurrences.len());
        for occurrence in block_occurrences {
            sharing_files.push(occurrence.file_index);
        }
        sharing_files.sort_unstable();
        sharing_files.dedup();
        if sharing_files.len() < 2 {
            continue;
        }
        for occurrence in block_occurrences {
            record_block_lines(occurrence, &sharing_files, &mut lines_by_file_pair);
        }
    }
    lines_by_file_pair
}

fn record_block_lines(
    occurrence: &BlockOccurrence,
    sharing_files: &[u32],
    lines_by_file_pair: &mut LinesByFilePair,
) {
    let block_line_range = occurrence.first_line_index
        ..occurrence.first_line_index + REPEATED_BLOCK_MINIMUM_LINES as u32;
    for &partner_file_index in sharing_files {
        if partner_file_index == occurrence.file_index {
            continue;
        }
        lines_by_file_pair
            .entry((occurrence.file_index, partner_file_index))
            .or_default()
            .extend(block_line_range.clone());
    }
}

/// Distinct repeated lines across the corpus: a line shared with several
/// files is counted once.
fn count_repeated_lines(lines_by_file_pair: &LinesByFilePair) -> usize {
    let mut repeated_lines: HashSet<(u32, u32)> = HashSet::new();
    for (&(file_index, _partner_file_index), shared_lines) in lines_by_file_pair {
        for &line_index in shared_lines {
            repeated_lines.insert((file_index, line_index));
        }
    }
    repeated_lines.len()
}

fn count_lines_by_file_pair(lines_by_file_pair: &LinesByFilePair) -> HashMap<(u32, u32), usize> {
    let mut file_pair_counts = HashMap::with_capacity(lines_by_file_pair.len());
    for (&file_pair, shared_lines) in lines_by_file_pair {
        file_pair_counts.insert(file_pair, shared_lines.len());
    }
    file_pair_counts
}

/// Lines in area A that repeat in area B, for every ordered pair of areas. A
/// line of A shared with two files of B is counted once for that pair.
fn summarize_area_pairs(
    source_files: &[NormalizedSourceFile],
    lines_by_file_pair: &LinesByFilePair,
) -> Vec<RepeatedLinePair> {
    let mut area_names: Vec<&str> = Vec::new();
    let mut area_index_by_name: HashMap<&str, u32> = HashMap::new();
    let mut area_index_by_file: Vec<u32> = Vec::with_capacity(source_files.len());
    for source_file in source_files {
        let next_area_index = area_names.len() as u32;
        let area_index = *area_index_by_name
            .entry(source_file.area.as_str())
            .or_insert(next_area_index);
        if area_index == next_area_index {
            area_names.push(source_file.area.as_str());
        }
        area_index_by_file.push(area_index);
    }
    let mut lines_by_area_pair: HashMap<(u32, u32), HashSet<(u32, u32)>> = HashMap::new();
    for (&(file_index, partner_file_index), shared_lines) in lines_by_file_pair {
        let area_pair = (
            area_index_by_file[file_index as usize],
            area_index_by_file[partner_file_index as usize],
        );
        let area_pair_lines = lines_by_area_pair.entry(area_pair).or_default();
        for &line_index in shared_lines {
            area_pair_lines.insert((file_index, line_index));
        }
    }
    let mut area_pair_counts = HashMap::with_capacity(lines_by_area_pair.len());
    for (area_pair, area_pair_lines) in lines_by_area_pair {
        area_pair_counts.insert(area_pair, area_pair_lines.len());
    }
    larger_direction_pairs(&area_pair_counts, &area_names)
}

/// Folds the two directions of each pair into one entry that keeps the
/// larger count and names that direction's side first, then sorts largest
/// first, breaking ties by name so the report reads the same on every run.
pub(crate) fn larger_direction_pairs(
    directional_counts: &HashMap<(u32, u32), usize>,
    place_names: &[&str],
) -> Vec<RepeatedLinePair> {
    let mut larger_direction_by_pair: HashMap<(u32, u32), (u32, u32, usize)> = HashMap::new();
    for (&(first_index, second_index), &repeated_line_count) in directional_counts {
        let unordered_pair = (first_index.min(second_index), first_index.max(second_index));
        let kept_direction = larger_direction_by_pair.entry(unordered_pair).or_insert((
            first_index,
            second_index,
            repeated_line_count,
        ));
        if repeated_line_count > kept_direction.2
            || (repeated_line_count == kept_direction.2 && first_index < kept_direction.0)
        {
            *kept_direction = (first_index, second_index, repeated_line_count);
        }
    }
    let mut repeated_line_pairs: Vec<RepeatedLinePair> = Vec::new();
    for (first_index, second_index, repeated_line_count) in larger_direction_by_pair.into_values() {
        repeated_line_pairs.push(RepeatedLinePair {
            first_place: place_names[first_index as usize].to_owned(),
            second_place: place_names[second_index as usize].to_owned(),
            repeated_line_count,
        });
    }
    repeated_line_pairs.sort_by(compare_largest_first);
    repeated_line_pairs
}

fn compare_largest_first(
    first_pair: &RepeatedLinePair,
    second_pair: &RepeatedLinePair,
) -> Ordering {
    second_pair
        .repeated_line_count
        .cmp(&first_pair.repeated_line_count)
        .then(first_pair.first_place.cmp(&second_pair.first_place))
        .then(first_pair.second_place.cmp(&second_pair.second_place))
}
