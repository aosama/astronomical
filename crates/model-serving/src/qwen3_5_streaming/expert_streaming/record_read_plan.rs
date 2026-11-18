//! Turns the slices a batch of experts needs into the fewest reads.
//!
//! Each expert is nine slices of a file (a weight, scales and biases for each
//! of three projections), and each slice belongs in a different table. Where
//! the file stores an expert's slices back to back, as an Expert Atlas layer
//! file does, nine slices are one read that scatters into nine destinations;
//! where it stores them gigabytes apart, as a stock checkpoint does, they
//! stay nine reads. This module only decides the grouping, from offsets and
//! lengths, so it can be tested without a file or a graphics processor.

/// One slice of a file that one destination wants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageSlice {
    /// Which open file the slice is in; slices of different files never join.
    pub(crate) file_index: usize,
    pub(crate) file_offset: u64,
    pub(crate) byte_count: usize,
}

/// Reads longer than this are cut in two, so two requests reach storage at
/// once. Measured on a five-expert fetch read from storage, 1.7 MB records
/// took 0.88 ms whole and 0.81 ms as halves
/// (`cargo measure-fetching-layer-stages`).
const HALVED_READ_MINIMUM_BYTES: usize = 1 << 20;

/// Groups `slices` into reads: each group is a run of slice indexes whose
/// slices follow one another in the same file with no gap, in file order.
/// A run of two or more slices longer than `HALVED_READ_MINIMUM_BYTES` is cut
/// where its second half begins.
pub(crate) fn plan_record_reads(slices: &[StorageSlice]) -> Vec<Vec<usize>> {
    let mut slice_indexes_in_file_order: Vec<usize> = (0..slices.len()).collect();
    slice_indexes_in_file_order.sort_by_key(|&slice_index| {
        (
            slices[slice_index].file_index,
            slices[slice_index].file_offset,
            slice_index,
        )
    });
    let mut contiguous_runs: Vec<Vec<usize>> = Vec::new();
    for slice_index in slice_indexes_in_file_order {
        let slice = slices[slice_index];
        if slice.byte_count == 0 {
            continue;
        }
        let continues_last_run = contiguous_runs.last().is_some_and(|last_run| {
            let last_slice = slices[last_run[last_run.len() - 1]];
            last_slice.file_index == slice.file_index
                && last_slice.file_offset + last_slice.byte_count as u64 == slice.file_offset
        });
        if continues_last_run {
            let last_run_index = contiguous_runs.len() - 1;
            contiguous_runs[last_run_index].push(slice_index);
        } else {
            contiguous_runs.push(vec![slice_index]);
        }
    }
    let mut planned_reads = Vec::with_capacity(contiguous_runs.len() * 2);
    for contiguous_run in contiguous_runs {
        let (first_half, second_half) = halves_of_a_long_run(slices, contiguous_run);
        planned_reads.push(first_half);
        if !second_half.is_empty() {
            planned_reads.push(second_half);
        }
    }
    planned_reads
}

/// Cuts a long run at the first slice that starts at or past its middle; a
/// short run, or one slice, comes back whole with an empty second half.
fn halves_of_a_long_run(slices: &[StorageSlice], run: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut run_byte_count = 0_usize;
    for &slice_index in &run {
        run_byte_count += slices[slice_index].byte_count;
    }
    if run.len() < 2 || run_byte_count < HALVED_READ_MINIMUM_BYTES {
        return (run, Vec::new());
    }
    let mut bytes_before_slice = 0_usize;
    for (run_position, &slice_index) in run.iter().enumerate() {
        if run_position > 0 && bytes_before_slice * 2 >= run_byte_count {
            let second_half = run[run_position..].to_vec();
            let mut first_half = run;
            first_half.truncate(run_position);
            return (first_half, second_half);
        }
        bytes_before_slice += slices[slice_index].byte_count;
    }
    (run, Vec::new())
}

#[cfg(test)]
#[path = "record_read_plan_tests.rs"]
mod tests;
