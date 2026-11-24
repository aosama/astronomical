// The C interface of the expert record reader.
//
// A record is one contiguous span of a file. One read fills a list of
// destinations from it, in order, with a single preadv: the span's first
// bytes go to the first destination, and so on. A batch of reads is issued
// from the calling thread and the reader's long-lived helper threads at once
// and returns when every read has finished.

#ifndef ASTRONOMICAL_EXPERT_RECORD_READER_H
#define ASTRONOMICAL_EXPERT_RECORD_READER_H

#include <stddef.h>
#include <stdint.h>
#include <sys/uio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct astro_expert_record_reader astro_expert_record_reader;

typedef struct astro_expert_record_read {
    int32_t file_descriptor;
    int32_t destination_count;
    int64_t file_offset;
    // `destination_count` destinations, filled in order. Each must stay
    // valid, and written by nobody else, until the batch returns.
    const struct iovec* destinations;
} astro_expert_record_read;

// Starts `helper_thread_count` helper threads; zero makes every batch run on
// the calling thread alone. Returns null when a thread cannot be started.
astro_expert_record_reader* astro_expert_record_reader_create(uint32_t helper_thread_count);

// Stops and joins the helper threads. No batch may be running.
void astro_expert_record_reader_destroy(astro_expert_record_reader* reader);

// Keeps storage awake between batches. A drive left alone for a millisecond
// answers its next request more slowly: a fetch of four experts measured 0.81
// ms after a quiet millisecond against 0.68 ms straight after another read,
// and a reply token makes dozens of fetches about a millisecond apart. While
// batches keep coming, a thread of the reader's own reads one 16 KiB page
// every `pause_microseconds`, from the file the last batch read; it stops by
// itself `linger_milliseconds` after the last batch. Off until this is
// called; a linger of zero turns it off again.
void astro_expert_record_reader_keep_storage_awake(astro_expert_record_reader* reader,
                                                   uint32_t pause_microseconds,
                                                   uint32_t linger_milliseconds);

// How many pages the reader has read to keep storage awake since it was
// created: what that upkeep costs, for whoever measures or reports it.
uint64_t astro_expert_record_reader_awake_read_count(const astro_expert_record_reader* reader);

// What a read returns when the file ended before its destinations were full.
// Negative so no errno can take it: a drive's own EIO must not read as a file
// too short to hold the record.
#define ASTRO_EXPERT_RECORD_TRUNCATED (-1)

// Runs every read of the batch and returns when all have finished: 0 when
// every read filled every destination, otherwise the status of the first read
// to fail (ASTRO_EXPERT_RECORD_TRUNCATED, or the errno storage gave).
// Batches from different threads run one after another.
int32_t astro_expert_record_reader_read(astro_expert_record_reader* reader,
                                        const astro_expert_record_read* reads,
                                        size_t read_count);

#ifdef __cplusplus
}
#endif

#endif
