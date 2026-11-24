// The expert record reader: long-lived threads that issue one preadv per
// record, straight into memory the caller owns.
//
// Why long-lived threads: a reply token fetches a handful of records on each
// of dozens of layers, and each fetch is under a millisecond. Starting a
// thread per record costs a measurable share of that; a thread already
// waiting costs a wake-up. Why the caller reads too: a batch of one record,
// or the first record of any batch, then starts with no wake-up at all.
//
// Why one preadv per record: a record holds an expert's tensors back to
// back, and each belongs in a different place in memory. One scatter read
// puts every tensor where it is used with a single system call and a single
// trip to storage, where separate reads would queue one request per tensor.

#include "expert_record_reader.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <new>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// What the reader reads to keep storage awake: one page of the size macOS
// reads directly, so the request goes to the drive and not to memory.
constexpr size_t kAwakeReadBytes = 16384;

int64_t monotonic_nanoseconds() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// The total bytes a read's destinations hold, which a complete preadv returns.
ssize_t destination_byte_count(const astro_expert_record_read& read) {
    ssize_t byte_count = 0;
    for (int32_t index = 0; index < read.destination_count; ++index) {
        byte_count += static_cast<ssize_t>(read.destinations[index].iov_len);
    }
    return byte_count;
}

// One record: 0, or the errno that stopped it. A regular file returns fewer
// bytes than asked only at its end, so a short read means the file does not
// hold the record and is reported as ASTRO_EXPERT_RECORD_TRUNCATED instead of
// being retried.
int32_t read_one_record(const astro_expert_record_read& read) {
    if (read.destination_count == 0) {
        // preadv rejects an empty destination list; a record with nothing to
        // fill is simply complete.
        return 0;
    }
    for (;;) {
        const ssize_t read_byte_count =
            preadv(read.file_descriptor, read.destinations, read.destination_count,
                   static_cast<off_t>(read.file_offset));
        if (read_byte_count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return errno;
        }
        return read_byte_count == destination_byte_count(read) ? 0 : ASTRO_EXPERT_RECORD_TRUNCATED;
    }
}

}  // namespace

struct astro_expert_record_reader {
    std::mutex batch_mutex;
    std::condition_variable batch_arrived;
    std::condition_variable batch_finished;
    // Serializes whole batches, so two callers never share the fields below.
    std::mutex caller_mutex;

    const astro_expert_record_read* reads = nullptr;
    size_t read_count = 0;
    // The next read nobody has claimed; claimed with one atomic step so the
    // caller and the helpers never take the lock to pick work.
    std::atomic<size_t> next_read_index{0};
    std::atomic<int32_t> first_error{0};
    // Helpers still inside the current batch; guarded by batch_mutex.
    size_t working_helper_count = 0;
    // Bumped for every batch so a helper can tell a new batch from the one
    // it already worked; guarded by batch_mutex.
    uint64_t batch_generation = 0;
    bool is_stopping = false;

    std::vector<std::thread> helper_threads;

    // Keeping storage awake between batches. `awake_*` settings and the
    // descriptor are written under awake_mutex; the thread sleeps on
    // awake_wanted while no batch has come for longer than the linger.
    std::mutex awake_mutex;
    std::condition_variable awake_wanted;
    std::thread awake_thread;
    bool awake_thread_is_stopping = false;
    uint32_t awake_pause_microseconds = 0;
    int64_t awake_linger_nanoseconds = 0;
    // A duplicate of the descriptor the last batch read from, so the thread
    // never reads a descriptor its owner has since closed. Only the awake
    // thread reads or closes it; a batch that reads another file leaves a
    // new duplicate in `awake_next_file_descriptor` for the thread to adopt.
    // The thread can then read without holding the lock, which a batch needs
    // for a moment on its way in. Files are told apart by descriptor number:
    // a caller that closes a file and opens another under the same number
    // leaves the thread reading the first, which keeps storage just as awake
    // (the drive is what matters, not the file) until a batch brings a
    // different number.
    int awake_file_descriptor = -1;
    int awake_next_file_descriptor = -1;
    int awake_source_file_descriptor = -1;
    int64_t awake_file_offset = 0;
    std::atomic<int64_t> last_batch_nanoseconds{0};
    std::atomic<uint64_t> awake_read_count{0};
    void* awake_page = nullptr;

    // Remembers where the batch read and wakes the thread that keeps storage
    // awake. Called once for every batch.
    void note_batch_for_keeping_awake(const astro_expert_record_read& first_read) {
        last_batch_nanoseconds.store(monotonic_nanoseconds(), std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(awake_mutex);
        if (awake_linger_nanoseconds == 0) {
            return;
        }
        if (first_read.file_descriptor != awake_source_file_descriptor) {
            if (awake_next_file_descriptor >= 0) {
                close(awake_next_file_descriptor);
            }
            awake_next_file_descriptor = dup(first_read.file_descriptor);
            awake_source_file_descriptor = first_read.file_descriptor;
        }
        awake_file_offset =
            first_read.file_offset - (first_read.file_offset % static_cast<int64_t>(kAwakeReadBytes));
        awake_wanted.notify_one();
    }

    void awake_loop() {
        std::unique_lock<std::mutex> lock(awake_mutex);
        for (;;) {
            if (awake_next_file_descriptor >= 0) {
                if (awake_file_descriptor >= 0) {
                    close(awake_file_descriptor);
                }
                awake_file_descriptor = awake_next_file_descriptor;
                awake_next_file_descriptor = -1;
            }
            const bool has_work =
                awake_linger_nanoseconds != 0 && awake_file_descriptor >= 0 &&
                monotonic_nanoseconds() - last_batch_nanoseconds.load(std::memory_order_relaxed) <=
                    awake_linger_nanoseconds;
            if (awake_thread_is_stopping) {
                return;
            }
            if (!has_work) {
                awake_wanted.wait(lock);
                continue;
            }
            const int file_descriptor = awake_file_descriptor;
            const int64_t file_offset = awake_file_offset;
            const uint32_t pause_microseconds = awake_pause_microseconds;
            lock.unlock();
            (void)pread(file_descriptor, awake_page, kAwakeReadBytes, static_cast<off_t>(file_offset));
            awake_read_count.fetch_add(1, std::memory_order_relaxed);
            if (pause_microseconds > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(pause_microseconds));
            }
            lock.lock();
        }
    }

    // Claims and runs reads until none is left.
    void run_claimed_reads(const astro_expert_record_read* batch_reads, size_t batch_read_count) {
        for (;;) {
            const size_t read_index = next_read_index.fetch_add(1, std::memory_order_relaxed);
            if (read_index >= batch_read_count) {
                return;
            }
            const int32_t read_error = read_one_record(batch_reads[read_index]);
            if (read_error != 0) {
                int32_t no_error = 0;
                first_error.compare_exchange_strong(no_error, read_error);
            }
        }
    }

    void helper_loop() {
        uint64_t worked_generation = 0;
        std::unique_lock<std::mutex> lock(batch_mutex);
        for (;;) {
            while (!is_stopping && batch_generation == worked_generation) {
                batch_arrived.wait(lock);
            }
            if (is_stopping) {
                return;
            }
            worked_generation = batch_generation;
            const astro_expert_record_read* batch_reads = reads;
            const size_t batch_read_count = read_count;
            if (batch_reads == nullptr) {
                // This helper woke after its batch had already finished. It
                // must not touch the claim counter: the next batch may have
                // reset it, and a claim made here would be a read nobody runs.
                continue;
            }
            working_helper_count += 1;
            lock.unlock();
            run_claimed_reads(batch_reads, batch_read_count);
            lock.lock();
            working_helper_count -= 1;
            if (working_helper_count == 0) {
                batch_finished.notify_one();
            }
        }
    }
};

astro_expert_record_reader* astro_expert_record_reader_create(uint32_t helper_thread_count) {
    astro_expert_record_reader* reader = new (std::nothrow) astro_expert_record_reader();
    if (reader == nullptr) {
        return nullptr;
    }
    try {
        reader->helper_threads.reserve(helper_thread_count);
        for (uint32_t index = 0; index < helper_thread_count; ++index) {
            reader->helper_threads.emplace_back(&astro_expert_record_reader::helper_loop, reader);
        }
    } catch (...) {
        astro_expert_record_reader_destroy(reader);
        return nullptr;
    }
    return reader;
}

void astro_expert_record_reader_destroy(astro_expert_record_reader* reader) {
    if (reader == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(reader->awake_mutex);
        reader->awake_thread_is_stopping = true;
    }
    reader->awake_wanted.notify_all();
    if (reader->awake_thread.joinable()) {
        reader->awake_thread.join();
    }
    if (reader->awake_file_descriptor >= 0) {
        close(reader->awake_file_descriptor);
    }
    if (reader->awake_next_file_descriptor >= 0) {
        close(reader->awake_next_file_descriptor);
    }
    std::free(reader->awake_page);
    {
        std::lock_guard<std::mutex> lock(reader->batch_mutex);
        reader->is_stopping = true;
    }
    reader->batch_arrived.notify_all();
    for (std::thread& helper_thread : reader->helper_threads) {
        helper_thread.join();
    }
    delete reader;
}

void astro_expert_record_reader_keep_storage_awake(astro_expert_record_reader* reader,
                                                   uint32_t pause_microseconds,
                                                   uint32_t linger_milliseconds) {
    if (reader == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(reader->awake_mutex);
    if (reader->awake_page == nullptr &&
        posix_memalign(&reader->awake_page, kAwakeReadBytes, kAwakeReadBytes) != 0) {
        reader->awake_page = nullptr;
        return;
    }
    reader->awake_pause_microseconds = pause_microseconds;
    reader->awake_linger_nanoseconds = static_cast<int64_t>(linger_milliseconds) * 1000000;
    if (linger_milliseconds > 0 && !reader->awake_thread.joinable()) {
        try {
            reader->awake_thread = std::thread(&astro_expert_record_reader::awake_loop, reader);
        } catch (...) {
            reader->awake_linger_nanoseconds = 0;
        }
    }
}

uint64_t astro_expert_record_reader_awake_read_count(const astro_expert_record_reader* reader) {
    if (reader == nullptr) {
        return 0;
    }
    return reader->awake_read_count.load(std::memory_order_relaxed);
}

int32_t astro_expert_record_reader_read(astro_expert_record_reader* reader,
                                        const astro_expert_record_read* reads,
                                        size_t read_count) {
    if (reader == nullptr || (reads == nullptr && read_count > 0)) {
        return EINVAL;
    }
    if (read_count == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> caller_lock(reader->caller_mutex);
    reader->note_batch_for_keeping_awake(reads[0]);
    {
        std::lock_guard<std::mutex> lock(reader->batch_mutex);
        reader->reads = reads;
        reader->read_count = read_count;
        reader->next_read_index.store(0, std::memory_order_relaxed);
        reader->first_error.store(0, std::memory_order_relaxed);
        reader->batch_generation += 1;
    }
    // A batch of one is read here with no helper woken: the wake-up would
    // cost more than it saves.
    if (read_count > 1) {
        reader->batch_arrived.notify_all();
    }
    reader->run_claimed_reads(reads, read_count);
    // A helper that woke for this batch holds the batch's pointer until it
    // leaves it, so the batch may not return (and its reads be freed) before
    // every such helper has left.
    std::unique_lock<std::mutex> lock(reader->batch_mutex);
    while (reader->working_helper_count != 0) {
        reader->batch_finished.wait(lock);
    }
    reader->reads = nullptr;
    reader->read_count = 0;
    return reader->first_error.load(std::memory_order_relaxed);
}
