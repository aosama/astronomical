// The K best of candidates [0, CANDIDATES), read through CANDIDATE_VALUE(index)
// and CANDIDATE_ID(index) and written through WRITE_BEST(round, value, id) by
// the first simdgroup; each top-k kernel defines the three ahead of it. Shared
// by every pass, so the chunk pass and the merge rank candidates the same
// way: every thread keeps the best K of its strided share in registers, each
// simdgroup merges its 32 lists by K rounds of simd_max, and the first
// simdgroup merges the 8 simdgroups' lists the same way. Equal logits go to
// the lower token id.

    constexpr uint THREADS = 256;
    constexpr uint SIMDGROUPS = THREADS / 32;
    const uint thread_index = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;

    float best_values[K];
    uint best_ids[K];
    for (int slot = 0; slot < K; ++slot) {
        best_values[slot] = -INFINITY;
        best_ids[slot] = 0xFFFFFFFFu;
    }
    for (uint index = thread_index; index < CANDIDATES; index += THREADS) {
        const float value = CANDIDATE_VALUE(index);
        const uint token = CANDIDATE_ID(index);
        if (value < best_values[K - 1] ||
            (value == best_values[K - 1] && token > best_ids[K - 1])) {
            continue;
        }
        int slot = K - 1;
        while (slot > 0 && (value > best_values[slot - 1] ||
               (value == best_values[slot - 1] && token < best_ids[slot - 1]))) {
            best_values[slot] = best_values[slot - 1];
            best_ids[slot] = best_ids[slot - 1];
            --slot;
        }
        best_values[slot] = value;
        best_ids[slot] = token;
    }

    threadgroup float simdgroup_values[THREADS / 32 * K];
    threadgroup uint simdgroup_ids[THREADS / 32 * K];
    int head = 0;
    for (int round = 0; round < K; ++round) {
        const float candidate = head < K ? best_values[head] : -INFINITY;
        const uint candidate_id = head < K ? best_ids[head] : 0xFFFFFFFFu;
        const float round_best = simd_max(candidate);
        const uint round_id = simd_min(candidate == round_best ? candidate_id : 0xFFFFFFFFu);
        if (candidate == round_best && candidate_id == round_id) {
            ++head;
        }
        if (lane == 0) {
            simdgroup_values[simdgroup * K + round] = round_best;
            simdgroup_ids[simdgroup * K + round] = round_id;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup != 0) {
        return;
    }
    head = 0;
    for (int round = 0; round < K; ++round) {
        const bool lane_has_list = lane < SIMDGROUPS && head < K;
        const float candidate = lane_has_list ? simdgroup_values[lane * K + head] : -INFINITY;
        const uint candidate_id = lane_has_list ? simdgroup_ids[lane * K + head] : 0xFFFFFFFFu;
        const float round_best = simd_max(candidate);
        const uint round_id = simd_min(candidate == round_best ? candidate_id : 0xFFFFFFFFu);
        if (candidate == round_best && candidate_id == round_id) {
            ++head;
        }
        if (lane == 0) {
            WRITE_BEST(round, round_best, round_id);
        }
    }
