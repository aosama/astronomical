use std::io::Write;
use std::os::fd::AsFd;

use super::{ExpertRecordReader, RecordRead, RecordReadError, open_for_direct_reads};

const BLOCK_BYTES: usize = 4096;

/// A file whose every byte names its 4 KiB block, so a misplaced read shows
/// up as the wrong block number.
fn numbered_block_file(block_count: usize) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().expect("a scratch file");
    for block_index in 0..block_count {
        file.write_all(&vec![block_index as u8; BLOCK_BYTES])
            .expect("the block writes");
    }
    file.flush().expect("the file flushes");
    file
}

#[test]
fn should_read_each_record_into_its_own_destination() {
    let file = numbered_block_file(8);
    let reader = ExpertRecordReader::new(3).expect("the reader starts");
    let mut destinations = vec![vec![0xFF_u8; BLOCK_BYTES]; 4];
    // Records in reverse file order, so a read that ignored its offset shows.
    let mut record_reads = Vec::new();
    for (destination_index, destination) in destinations.iter_mut().enumerate() {
        record_reads.push(RecordRead {
            file: file.as_file().as_fd(),
            file_offset: ((7 - destination_index) * BLOCK_BYTES) as u64,
            destinations: vec![destination.as_mut_slice()],
        });
    }
    reader.read(&mut record_reads).expect("the batch reads");
    drop(record_reads);
    for (destination_index, destination) in destinations.iter().enumerate() {
        assert!(
            destination
                .iter()
                .all(|byte| *byte == (7 - destination_index) as u8),
            "destination {destination_index} holds block {}",
            7 - destination_index
        );
    }
}

#[test]
fn should_scatter_one_record_over_its_destinations_in_order() {
    let file = numbered_block_file(4);
    let reader = ExpertRecordReader::new(0).expect("the reader starts");
    let mut first_destination = vec![0xFF_u8; BLOCK_BYTES];
    let mut second_destination = vec![0xFF_u8; 2 * BLOCK_BYTES];
    let mut record_reads = vec![RecordRead {
        file: file.as_file().as_fd(),
        file_offset: BLOCK_BYTES as u64,
        destinations: vec![
            first_destination.as_mut_slice(),
            second_destination.as_mut_slice(),
        ],
    }];
    reader.read(&mut record_reads).expect("the record reads");
    drop(record_reads);
    assert!(first_destination.iter().all(|byte| *byte == 1));
    assert!(
        second_destination[..BLOCK_BYTES]
            .iter()
            .all(|byte| *byte == 2)
    );
    assert!(
        second_destination[BLOCK_BYTES..]
            .iter()
            .all(|byte| *byte == 3)
    );
}

#[test]
fn should_read_far_more_records_than_it_has_threads() {
    let block_count = 200;
    let file = numbered_block_file(block_count);
    let reader = ExpertRecordReader::new(2).expect("the reader starts");
    let mut destinations = vec![vec![0xFF_u8; BLOCK_BYTES]; block_count];
    let mut record_reads = Vec::new();
    for (block_index, destination) in destinations.iter_mut().enumerate() {
        record_reads.push(RecordRead {
            file: file.as_file().as_fd(),
            file_offset: (block_index * BLOCK_BYTES) as u64,
            destinations: vec![destination.as_mut_slice()],
        });
    }
    reader.read(&mut record_reads).expect("the batch reads");
    drop(record_reads);
    for (block_index, destination) in destinations.iter().enumerate() {
        assert!(destination.iter().all(|byte| *byte == block_index as u8));
    }
}

#[test]
fn should_read_every_record_of_many_batches_in_a_row() {
    // A helper that wakes late for a finished batch must not take a read of
    // the next one: every batch here would then leave a destination unread.
    let file = numbered_block_file(4);
    let reader = ExpertRecordReader::new(6).expect("the reader starts");
    for _ in 0..2_000 {
        let mut destinations = vec![vec![0xFF_u8; BLOCK_BYTES]; 4];
        let mut record_reads = Vec::new();
        for (block_index, destination) in destinations.iter_mut().enumerate() {
            record_reads.push(RecordRead {
                file: file.as_file().as_fd(),
                file_offset: (block_index * BLOCK_BYTES) as u64,
                destinations: vec![destination.as_mut_slice()],
            });
        }
        reader.read(&mut record_reads).expect("the batch reads");
        drop(record_reads);
        for (block_index, destination) in destinations.iter().enumerate() {
            assert_eq!(destination[0], block_index as u8);
            assert_eq!(destination[BLOCK_BYTES - 1], block_index as u8);
        }
    }
}

#[test]
fn should_serve_batches_from_several_threads() {
    let file = numbered_block_file(16);
    let reader = ExpertRecordReader::new(4).expect("the reader starts");
    std::thread::scope(|scope| {
        for caller_index in 0..4_usize {
            let reader = &reader;
            let file = &file;
            scope.spawn(move || {
                for _ in 0..200 {
                    let mut destination = vec![0xFF_u8; BLOCK_BYTES];
                    let mut record_reads = vec![RecordRead {
                        file: file.as_file().as_fd(),
                        file_offset: (caller_index * BLOCK_BYTES) as u64,
                        destinations: vec![destination.as_mut_slice()],
                    }];
                    reader.read(&mut record_reads).expect("the batch reads");
                    drop(record_reads);
                    assert!(destination.iter().all(|byte| *byte == caller_index as u8));
                }
            });
        }
    });
}

#[test]
fn should_do_nothing_for_an_empty_batch() {
    let reader = ExpertRecordReader::new(2).expect("the reader starts");
    assert_eq!(reader.read(&mut []), Ok(()));
}

#[test]
fn should_fill_nothing_for_a_record_with_no_destinations() {
    let file = numbered_block_file(1);
    let reader = ExpertRecordReader::new(0).expect("the reader starts");
    let mut record_reads = vec![RecordRead {
        file: file.as_file().as_fd(),
        file_offset: 0,
        destinations: Vec::new(),
    }];
    assert_eq!(reader.read(&mut record_reads), Ok(()));
}

#[test]
fn should_report_a_record_the_file_does_not_hold() {
    let file = numbered_block_file(2);
    let reader = ExpertRecordReader::new(1).expect("the reader starts");
    let mut destination = vec![0_u8; 2 * BLOCK_BYTES];
    let mut record_reads = vec![RecordRead {
        file: file.as_file().as_fd(),
        // One block from the end, asking for two.
        file_offset: BLOCK_BYTES as u64,
        destinations: vec![destination.as_mut_slice()],
    }];
    assert_eq!(
        reader.read(&mut record_reads),
        Err(RecordReadError::RecordTruncated)
    );
}

#[test]
fn should_report_the_failure_of_one_record_among_good_ones() {
    let file = numbered_block_file(4);
    let write_only_file = std::fs::OpenOptions::new()
        .write(true)
        .open(file.path())
        .expect("the file opens for writing only");
    let reader = ExpertRecordReader::new(2).expect("the reader starts");
    let mut good_destination = vec![0_u8; BLOCK_BYTES];
    let mut refused_destination = vec![0_u8; BLOCK_BYTES];
    let mut record_reads = vec![
        RecordRead {
            file: file.as_file().as_fd(),
            file_offset: 0,
            destinations: vec![good_destination.as_mut_slice()],
        },
        RecordRead {
            // A descriptor opened without read access refuses the read.
            file: write_only_file.as_fd(),
            file_offset: 0,
            destinations: vec![refused_destination.as_mut_slice()],
        },
    ];
    let read_error = reader
        .read(&mut record_reads)
        .expect_err("a refused read fails the batch");
    assert!(
        matches!(read_error, RecordReadError::ReadFailed { errno, .. } if errno == libc::EBADF),
        "{read_error:?}"
    );
}

#[test]
fn should_refuse_an_offset_no_read_can_address() {
    let file = numbered_block_file(1);
    let reader = ExpertRecordReader::new(0).expect("the reader starts");
    let mut destination = vec![0_u8; BLOCK_BYTES];
    let mut record_reads = vec![RecordRead {
        file: file.as_file().as_fd(),
        file_offset: u64::MAX,
        destinations: vec![destination.as_mut_slice()],
    }];
    assert_eq!(
        reader.read(&mut record_reads),
        Err(RecordReadError::ReadOutOfRange {
            file_offset: u64::MAX,
            destination_count: 1
        })
    );
}

#[test]
fn should_read_the_same_bytes_through_a_descriptor_opened_for_direct_reads() {
    let file = numbered_block_file(8);
    let direct_file = open_for_direct_reads(file.path()).expect("the file opens for direct reads");
    let reader = ExpertRecordReader::new(1).expect("the reader starts");
    let mut destination = vec![0xFF_u8; BLOCK_BYTES];
    let mut record_reads = vec![RecordRead {
        file: direct_file.as_fd(),
        file_offset: (5 * BLOCK_BYTES) as u64,
        destinations: vec![destination.as_mut_slice()],
    }];
    reader.read(&mut record_reads).expect("the record reads");
    drop(record_reads);
    assert!(destination.iter().all(|byte| *byte == 5));
}

#[test]
fn should_report_a_file_that_does_not_exist() {
    let missing_path = std::path::Path::new("/nonexistent/astronomical/expert-records");
    assert!(open_for_direct_reads(missing_path).is_err());
}

#[test]
fn should_read_every_batch_while_keeping_storage_awake_without_a_pause() {
    // With no pause the awake thread reads back to back. It must never hold
    // up a batch on its way in: 400 batches then take milliseconds, where a
    // thread that held its lock while reading made each one wait.
    let file = numbered_block_file(8);
    let reader = ExpertRecordReader::new(2).expect("the reader starts");
    reader.keep_storage_awake(
        std::time::Duration::ZERO,
        std::time::Duration::from_millis(20),
    );
    let batches_started_at = std::time::Instant::now();
    for batch_index in 0..400_usize {
        let block_index = batch_index % 8;
        let mut destination = vec![0xFF_u8; BLOCK_BYTES];
        let mut record_reads = vec![RecordRead {
            file: file.as_file().as_fd(),
            file_offset: (block_index * BLOCK_BYTES) as u64,
            destinations: vec![destination.as_mut_slice()],
        }];
        reader.read(&mut record_reads).expect("the batch reads");
        drop(record_reads);
        assert!(destination.iter().all(|byte| *byte == block_index as u8));
    }
    assert!(
        batches_started_at.elapsed() < std::time::Duration::from_secs(2),
        "batches waited on the awake thread: {:?}",
        batches_started_at.elapsed()
    );
}

#[test]
fn should_follow_batches_from_one_file_to_another_while_keeping_storage_awake() {
    let first_file = numbered_block_file(4);
    let second_file = numbered_block_file(4);
    let reader = ExpertRecordReader::new(1).expect("the reader starts");
    reader.keep_storage_awake(
        std::time::Duration::from_micros(50),
        std::time::Duration::from_millis(5),
    );
    for batch_index in 0..200_usize {
        let file = if batch_index % 2 == 0 {
            &first_file
        } else {
            &second_file
        };
        let mut destination = vec![0xFF_u8; BLOCK_BYTES];
        let mut record_reads = vec![RecordRead {
            file: file.as_file().as_fd(),
            file_offset: (2 * BLOCK_BYTES) as u64,
            destinations: vec![destination.as_mut_slice()],
        }];
        reader.read(&mut record_reads).expect("the batch reads");
        drop(record_reads);
        assert!(destination.iter().all(|byte| *byte == 2));
    }
}

#[test]
fn should_stop_keeping_storage_awake_when_the_linger_is_zero() {
    let file = numbered_block_file(2);
    let reader = ExpertRecordReader::new(0).expect("the reader starts");
    reader.keep_storage_awake(
        std::time::Duration::from_micros(100),
        std::time::Duration::from_millis(10),
    );
    reader.keep_storage_awake(
        std::time::Duration::from_micros(100),
        std::time::Duration::ZERO,
    );
    let mut destination = vec![0xFF_u8; BLOCK_BYTES];
    let mut record_reads = vec![RecordRead {
        file: file.as_file().as_fd(),
        file_offset: 0,
        destinations: vec![destination.as_mut_slice()],
    }];
    reader.read(&mut record_reads).expect("the batch reads");
    drop(record_reads);
    assert!(destination.iter().all(|byte| *byte == 0));
}

/// One batch of one block, which is what starts the awake reads.
fn read_one_block(reader: &ExpertRecordReader, file: &tempfile::NamedTempFile) {
    let mut destination = vec![0xFF_u8; BLOCK_BYTES];
    let mut record_reads = vec![RecordRead {
        file: file.as_file().as_fd(),
        file_offset: 0,
        destinations: vec![destination.as_mut_slice()],
    }];
    reader.read(&mut record_reads).expect("the batch reads");
}

#[test]
fn should_read_to_keep_storage_awake_only_until_the_linger_runs_out() {
    let file = numbered_block_file(8);
    let reader = ExpertRecordReader::new(0).expect("the reader starts");
    reader.keep_storage_awake(
        std::time::Duration::from_micros(200),
        std::time::Duration::from_millis(10),
    );
    assert_eq!(
        reader.awake_read_count(),
        0,
        "nothing is read before a batch"
    );
    read_one_block(&reader, &file);
    std::thread::sleep(std::time::Duration::from_millis(60));
    let count_after_the_linger = reader.awake_read_count();
    assert!(count_after_the_linger > 0, "a batch starts the awake reads");
    std::thread::sleep(std::time::Duration::from_millis(60));
    assert_eq!(
        reader.awake_read_count(),
        count_after_the_linger,
        "the awake reads stop by themselves once the linger has passed"
    );
}

#[test]
fn should_never_read_to_keep_storage_awake_unless_asked_to() {
    let file = numbered_block_file(8);
    let reader = ExpertRecordReader::new(1).expect("the reader starts");
    read_one_block(&reader, &file);
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(reader.awake_read_count(), 0);
}
