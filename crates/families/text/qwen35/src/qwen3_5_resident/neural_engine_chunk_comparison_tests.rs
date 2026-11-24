use std::time::{Duration, Instant};

use super::{ChunkComparison, ChunkComparisonVerdict, ChunkEngines};

const CHUNK_TOKENS: usize = 2_048;

/// Real chunk milliseconds from the `neural_engine_chunk_comparison` journey, one prompt
/// per row, each starting on the GPU as a cold prompt does.
const SMALLEST_MODEL_PROMPTS: [[u64; 12]; 4] = [
    [154, 202, 173, 217, 200, 266, 229, 297, 257, 319, 281, 319],
    [140, 175, 169, 202, 198, 233, 228, 262, 254, 293, 285, 321],
    [143, 177, 169, 201, 196, 230, 226, 260, 257, 296, 286, 327],
    [164, 313, 167, 209, 199, 235, 226, 261, 258, 300, 284, 316],
];
const TWO_BILLION_MODEL_PROMPTS: [[u64; 12]; 4] = [
    [329, 377, 330, 369, 358, 408, 388, 437, 413, 462, 442, 455],
    [306, 311, 334, 337, 356, 372, 385, 372, 413, 404, 441, 423],
    [301, 286, 332, 319, 365, 346, 383, 368, 410, 401, 438, 422],
    [331, 482, 329, 368, 354, 403, 383, 404, 407, 398, 437, 419],
];
const NINE_BILLION_MODEL_PROMPTS: [[u64; 12]; 4] = [
    [
        1497, 1279, 1486, 1224, 1553, 1573, 1616, 1392, 1694, 1503, 1811, 1760,
    ],
    [
        1516, 1314, 1604, 1377, 1675, 1467, 1758, 1512, 1848, 1624, 1918, 1692,
    ],
    [
        1550, 1315, 1613, 1429, 1677, 1433, 1749, 1495, 1842, 1592, 1888, 1732,
    ],
    [
        1432, 1486, 1515, 1356, 1554, 1442, 1636, 1487, 1729, 1559, 1808, 1603,
    ],
];

/// Runs prompts through the comparison as serving does, a minute apart so each is cold,
/// and returns the verdict once one exists.
fn verdict_after(
    prompts: &[[u64; 12]],
    chunks_per_prompt: usize,
) -> Option<ChunkComparisonVerdict> {
    let mut comparison = ChunkComparison::new(CHUNK_TOKENS);
    let mut now = Instant::now();
    for chunk_millis in prompts {
        now += Duration::from_secs(60);
        for (chunk_index, milliseconds) in chunk_millis.iter().take(chunks_per_prompt).enumerate() {
            comparison.choose_engines(now);
            now += Duration::from_millis(*milliseconds);
            comparison.note_finished_chunk(
                chunk_index * CHUNK_TOKENS,
                CHUNK_TOKENS,
                Duration::from_millis(*milliseconds),
                now,
            );
        }
    }
    comparison.verdict()
}

#[test]
fn should_turn_off_the_programs_of_a_model_they_slow_down() {
    let verdict = verdict_after(&SMALLEST_MODEL_PROMPTS, 12).expect("enough chunks ran");
    assert!(matches!(
        verdict,
        ChunkComparisonVerdict::NotWorthwhile { .. }
    ));
    assert!((0.85..0.95).contains(&verdict.gain()), "{verdict:?}");
}

#[test]
fn should_keep_the_programs_of_a_model_they_speed_up() {
    let verdict = verdict_after(&NINE_BILLION_MODEL_PROMPTS, 12).expect("enough chunks ran");
    assert!(matches!(verdict, ChunkComparisonVerdict::Worthwhile { .. }));
    assert!(verdict.gain() > 1.1, "{verdict:?}");
}

#[test]
fn should_keep_the_programs_of_a_model_with_a_small_gain_on_warm_prompts() {
    let warm_prompts = [
        TWO_BILLION_MODEL_PROMPTS[1],
        TWO_BILLION_MODEL_PROMPTS[2],
        TWO_BILLION_MODEL_PROMPTS[1],
        TWO_BILLION_MODEL_PROMPTS[2],
    ];
    let verdict = verdict_after(&warm_prompts, 12).expect("enough chunks ran");
    assert!(matches!(verdict, ChunkComparisonVerdict::Worthwhile { .. }));
}

/// The gain is barely there for a model whose prompts all wake cold programs, so turning
/// it off is the right answer for someone who uses it that way.
#[test]
fn should_turn_off_a_small_gain_that_cold_starts_use_up() {
    let verdict = verdict_after(&TWO_BILLION_MODEL_PROMPTS, 12).expect("enough chunks ran");
    assert!(matches!(
        verdict,
        ChunkComparisonVerdict::NotWorthwhile { .. }
    ));
    assert!((0.95..1.02).contains(&verdict.gain()), "{verdict:?}");
}

#[test]
fn should_reach_a_verdict_on_ten_thousand_token_prompts() {
    let many_prompts = [SMALLEST_MODEL_PROMPTS, SMALLEST_MODEL_PROMPTS].concat();
    // 10,000 tokens is four full chunks, so two comparisons each.
    let verdict = verdict_after(&many_prompts, 4).expect("eight prompts give sixteen comparisons");
    assert!(matches!(
        verdict,
        ChunkComparisonVerdict::NotWorthwhile { .. }
    ));
}

#[test]
fn should_not_decide_from_a_single_long_prompt() {
    assert_eq!(verdict_after(&SMALLEST_MODEL_PROMPTS[..1], 12), None);
}

#[test]
fn should_not_decide_from_prompts_too_short_to_compare() {
    assert_eq!(verdict_after(&SMALLEST_MODEL_PROMPTS, 2), None);
}

#[test]
fn should_lead_with_the_gpu_after_programs_go_cold_then_use_both() {
    let mut comparison = ChunkComparison::already_decided(
        CHUNK_TOKENS,
        ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
    );
    let mut now = Instant::now();
    assert_eq!(comparison.choose_engines(now), ChunkEngines::GpuAlone);
    now += Duration::from_secs(1);
    comparison.note_finished_chunk(0, CHUNK_TOKENS, Duration::from_secs(1), now);
    assert_eq!(comparison.choose_engines(now), ChunkEngines::Both);
    now += Duration::from_secs(1);
    comparison.note_finished_chunk(CHUNK_TOKENS, CHUNK_TOKENS, Duration::from_secs(1), now);
    now += Duration::from_secs(9);
    assert_eq!(comparison.choose_engines(now), ChunkEngines::Both);
    now += Duration::from_secs(1);
    comparison.note_finished_chunk(0, CHUNK_TOKENS, Duration::from_secs(1), now);
    now += Duration::from_secs(10);
    assert_eq!(comparison.choose_engines(now), ChunkEngines::GpuAlone);
}

#[test]
fn should_use_the_gpu_alone_for_a_model_found_slower_even_when_warm() {
    let mut comparison = ChunkComparison::already_decided(
        CHUNK_TOKENS,
        ChunkComparisonVerdict::NotWorthwhile { gain: 0.9 },
    );
    let now = Instant::now();
    assert_eq!(comparison.choose_engines(now), ChunkEngines::GpuAlone);
    assert_eq!(comparison.choose_engines(now), ChunkEngines::GpuAlone);
}

#[test]
fn should_not_compare_across_a_shorter_chunk() {
    let mut comparison = ChunkComparison::new(CHUNK_TOKENS);
    let mut now = Instant::now();
    let chunk_time = Duration::from_millis(100);
    for chunk_index in 0..40 {
        comparison.choose_engines(now);
        now += chunk_time;
        comparison.note_finished_chunk(chunk_index * CHUNK_TOKENS, CHUNK_TOKENS, chunk_time, now);
        comparison.note_finished_chunk(0, 512, chunk_time, now);
    }
    assert_eq!(comparison.verdict(), None);
}

#[test]
fn should_not_compare_across_a_gap_in_the_prompt() {
    let mut comparison = ChunkComparison::new(CHUNK_TOKENS);
    let mut now = Instant::now();
    let chunk_time = Duration::from_millis(100);
    for chunk_index in 0..40 {
        comparison.choose_engines(now);
        now += chunk_time;
        comparison.note_finished_chunk(
            chunk_index * 2 * CHUNK_TOKENS,
            CHUNK_TOKENS,
            chunk_time,
            now,
        );
    }
    assert_eq!(comparison.verdict(), None);
}

#[test]
fn should_not_compare_across_a_chunk_memory_kept_off_the_programs() {
    let mut comparison = ChunkComparison::new(CHUNK_TOKENS);
    let mut now = Instant::now();
    let chunk_time = Duration::from_millis(100);
    for chunk_index in 0..40 {
        comparison.choose_engines(now);
        now += chunk_time;
        comparison.note_finished_chunk(chunk_index * CHUNK_TOKENS, CHUNK_TOKENS, chunk_time, now);
        comparison.note_uncomparable_chunk(chunk_index * CHUNK_TOKENS, CHUNK_TOKENS);
    }
    assert_eq!(comparison.verdict(), None);
}

#[test]
fn should_ignore_zero_length_timings() {
    let mut comparison = ChunkComparison::new(CHUNK_TOKENS);
    let now = Instant::now();
    for chunk_index in 0..40 {
        comparison.choose_engines(now);
        comparison.note_finished_chunk(
            chunk_index * CHUNK_TOKENS,
            CHUNK_TOKENS,
            Duration::ZERO,
            now,
        );
    }
    assert_eq!(comparison.verdict(), None);
}

#[test]
fn should_treat_the_lowest_worthwhile_gain_as_worthwhile() {
    assert!(matches!(
        ChunkComparisonVerdict::from_gain(1.02),
        ChunkComparisonVerdict::Worthwhile { .. }
    ));
    assert!(matches!(
        ChunkComparisonVerdict::from_gain(1.019),
        ChunkComparisonVerdict::NotWorthwhile { .. }
    ));
}

#[test]
fn should_turn_off_a_model_that_is_only_just_slower_than_the_gpu() {
    assert!(matches!(
        ChunkComparisonVerdict::from_gain(0.98),
        ChunkComparisonVerdict::NotWorthwhile { .. }
    ));
}

#[test]
fn should_call_a_prompt_cold_from_its_lead_to_its_end() {
    let mut comparison = ChunkComparison::already_decided(
        CHUNK_TOKENS,
        ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
    );
    let mut now = Instant::now();
    let chunk_time = Duration::from_millis(100);
    comparison.choose_engines(now);
    assert!(comparison.did_lead_this_chunk());
    comparison.note_finished_chunk(0, CHUNK_TOKENS, chunk_time, now);
    assert!(!comparison.is_prompt_warm());

    now += chunk_time;
    comparison.choose_engines(now);
    assert!(!comparison.did_lead_this_chunk());
    comparison.note_finished_chunk(CHUNK_TOKENS, CHUNK_TOKENS, chunk_time, now);

    assert!(!comparison.is_prompt_warm());
}

#[test]
fn should_call_the_next_prompt_warm_when_it_follows_closely() {
    let mut comparison = ChunkComparison::already_decided(
        CHUNK_TOKENS,
        ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
    );
    let mut now = Instant::now();
    let chunk_time = Duration::from_millis(100);
    comparison.choose_engines(now);
    comparison.note_finished_chunk(0, CHUNK_TOKENS, chunk_time, now);
    comparison.choose_engines(now);
    comparison.note_finished_chunk(CHUNK_TOKENS, CHUNK_TOKENS, chunk_time, now);
    comparison.note_finished_chunk(2 * CHUNK_TOKENS, 512, chunk_time, now);

    now += Duration::from_secs(1);
    comparison.choose_engines(now);
    comparison.note_finished_chunk(0, CHUNK_TOKENS, chunk_time, now);

    assert!(comparison.is_prompt_warm());
}

#[test]
fn should_call_a_prompt_after_an_idle_spell_cold_again() {
    let mut comparison = ChunkComparison::already_decided(
        CHUNK_TOKENS,
        ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
    );
    let mut now = Instant::now();
    let chunk_time = Duration::from_millis(100);
    comparison.choose_engines(now);
    comparison.note_finished_chunk(0, CHUNK_TOKENS, chunk_time, now);
    comparison.choose_engines(now);
    comparison.note_finished_chunk(CHUNK_TOKENS, CHUNK_TOKENS, chunk_time, now);
    comparison.note_finished_chunk(2 * CHUNK_TOKENS, 512, chunk_time, now);
    comparison.choose_engines(now);
    comparison.note_finished_chunk(0, CHUNK_TOKENS, chunk_time, now);
    assert!(comparison.is_prompt_warm());

    now += Duration::from_secs(30);
    comparison.choose_engines(now);

    assert!(comparison.did_lead_this_chunk());
    assert!(!comparison.is_prompt_warm());
}

#[test]
fn should_use_both_engines_after_a_lead_longer_than_the_cold_limit() {
    for mut comparison in [
        ChunkComparison::new(CHUNK_TOKENS),
        ChunkComparison::already_decided(
            CHUNK_TOKENS,
            ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
        ),
    ] {
        let mut now = Instant::now();
        assert_eq!(comparison.choose_engines(now), ChunkEngines::GpuAlone);
        let slow_chunk = Duration::from_secs(12);
        now += slow_chunk;
        comparison.note_finished_chunk(0, CHUNK_TOKENS, slow_chunk, now);

        assert_eq!(comparison.choose_engines(now), ChunkEngines::Both);
    }
}

#[test]
fn should_still_reach_a_verdict_when_every_chunk_outlasts_the_cold_limit() {
    let mut comparison = ChunkComparison::new(CHUNK_TOKENS);
    let mut now = Instant::now();
    let slow_chunk = Duration::from_secs(12);
    for prompt_index in 0..6 {
        now += Duration::from_secs(60);
        for chunk_index in 0..12 {
            comparison.choose_engines(now);
            now += slow_chunk;
            comparison.note_finished_chunk(
                chunk_index * CHUNK_TOKENS,
                CHUNK_TOKENS,
                slow_chunk,
                now,
            );
        }
        assert!(prompt_index < 5 || comparison.verdict().is_some());
    }
}

#[test]
fn should_keep_a_cold_prompt_cold_through_a_chunk_memory_kept_off_the_programs() {
    let mut comparison = ChunkComparison::already_decided(
        CHUNK_TOKENS,
        ChunkComparisonVerdict::Worthwhile { gain: 1.2 },
    );
    let now = Instant::now();
    let chunk_time = Duration::from_millis(100);
    comparison.choose_engines(now);
    comparison.note_finished_chunk(0, CHUNK_TOKENS, chunk_time, now);
    comparison.note_uncomparable_chunk(CHUNK_TOKENS, CHUNK_TOKENS);
    comparison.choose_engines(now);
    comparison.note_finished_chunk(2 * CHUNK_TOKENS, CHUNK_TOKENS, chunk_time, now);

    assert!(!comparison.is_prompt_warm());
}

#[test]
fn should_count_one_broken_prompt_as_one_prompt() {
    let mut comparison = ChunkComparison::new(CHUNK_TOKENS);
    let now = Instant::now();
    let chunk_time = Duration::from_millis(100);
    let mut start_token = 0;
    for _ in 0..40 {
        for _ in 0..3 {
            comparison.choose_engines(now);
            comparison.note_finished_chunk(start_token, CHUNK_TOKENS, chunk_time, now);
            start_token += CHUNK_TOKENS;
        }
        comparison.note_finished_chunk(start_token, 512, chunk_time, now);
        start_token += 512;
    }
    assert_eq!(comparison.verdict(), None);
}

#[test]
fn should_name_the_line_a_refused_model_had_to_reach() {
    let reason = super::not_worthwhile_reason(1.015);
    assert!(
        reason.contains("1.01 times") || reason.contains("1.02 times"),
        "{reason}"
    );
    assert!(reason.contains("short of the 1.02"), "{reason}");
}

#[test]
fn should_round_the_speed_to_a_whole_percent() {
    assert_eq!(
        ChunkComparisonVerdict::Worthwhile { gain: 1.0649 }.speed_percent(),
        106
    );
    assert_eq!(
        ChunkComparisonVerdict::NotWorthwhile { gain: 0.905 }.speed_percent(),
        91
    );
}
