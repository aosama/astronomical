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

// Runs every read of the batch and returns when all have finished: 0 when
// every read filled every destination, otherwise the errno of the first read
// to fail (EIO for a read that ended before its destinations were full).
// Batches from different threads run one after another.
int32_t astro_expert_record_reader_read(astro_expert_record_reader* reader,
                                        const astro_expert_record_read* reads,
                                        size_t read_count);

#ifdef __cplusplus
}
#endif

#endif
