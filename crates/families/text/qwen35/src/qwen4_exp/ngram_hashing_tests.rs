use super::NgramHashTables;

const END_OF_SEQUENCE: u32 = 248_044;

/// The published checkpoint's hash constants.
fn published_tables() -> NgramHashTables {
    NgramHashTables::new(
        vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
        vec![
            20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
            20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
            20_000_161, 20_000_171,
        ],
        vec![
            0,
            20_000_003,
            40_000_026,
            60_000_059,
            80_000_106,
            100_000_165,
            120_000_228,
            140_000_297,
            160_000_374,
            180_000_455,
            200_000_548,
            220_000_655,
            240_000_802,
            260_000_955,
            280_001_114,
            300_001_275,
        ],
        8,
        END_OF_SEQUENCE,
    )
}

// Expected rows were produced by the reference `Qwen4ExpNGramEmbedding` in
// mlx-vlm 0.8.0 from the same constants: a stretch of Romeo and Juliet
// tokens with an end-of-sequence in the middle, then a continuation.
const FIRST_STRETCH: [u32; 5] = [1439, 1330, END_OF_SEQUENCE, 51941, 357];
const FIRST_STRETCH_ROWS: [u64; 80] = [
    19080894, 25453820, 52383136, 78276130, 90076913, 108142348, 125989035, 144515248, 160066698,
    198782743, 213512413, 223981269, 246273853, 269798303, 284580254, 300542626, 12338235,
    32833421, 43085834, 69444636, 97757222, 113862469, 128021274, 140234791, 167983839, 197570362,
    217236440, 238482184, 247045427, 275967875, 285688481, 314889770, 12656652, 37087449, 43722406,
    65961269, 86762350, 101305570, 124004314, 155919282, 179587874, 188288938, 210479265,
    220045455, 255554160, 272125384, 284551993, 308455826, 2343391, 39067014, 51978314, 75149560,
    99742096, 108910164, 123572160, 144819960, 163757884, 195770758, 218337840, 230926271,
    257771792, 265649448, 295171610, 304501773, 14454187, 37387359, 49630995, 71642388, 99888885,
    109470156, 133997506, 153657364, 164745846, 197604185, 214037994, 229492080, 258896570,
    268584263, 291876621, 308809866,
];
const SECOND_STRETCH: [u32; 2] = [6505, 314];
const SECOND_STRETCH_ROWS: [u64; 32] = [
    11669139, 25757920, 42922722, 75088320, 99926915, 114898821, 137380764, 147401571, 163112215,
    187671631, 216845881, 223295533, 256658700, 270124890, 294636707, 317366947, 12110801,
    32585035, 42919012, 65495079, 82089584, 114308499, 132656157, 157155869, 174083483, 187075883,
    202348693, 226667666, 253402891, 260160715, 282418538, 313744939,
];

#[test]
fn should_hash_a_first_stretch_with_an_end_of_sequence_exactly_as_the_reference() {
    let tables = published_tables();
    let row_ids = tables.row_ids(&tables.initial_context(), &FIRST_STRETCH);
    assert_eq!(row_ids, FIRST_STRETCH_ROWS);
}

#[test]
fn should_continue_from_the_last_two_tokens_of_the_previous_stretch() {
    let tables = published_tables();
    let context = tables.next_context(&tables.initial_context(), &FIRST_STRETCH);
    assert_eq!(context, vec![51941, 357]);
    assert_eq!(
        tables.row_ids(&context, &SECOND_STRETCH),
        SECOND_STRETCH_ROWS
    );
}

#[test]
fn should_hash_one_token_at_a_time_the_same_as_the_whole_stretch() {
    // Decode feeds one token per forward; prefill feeds a whole chunk.
    let tables = published_tables();
    let mut context = tables.initial_context();
    let mut token_by_token_rows = Vec::new();
    for token in FIRST_STRETCH {
        token_by_token_rows.extend(tables.row_ids(&context, &[token]));
        context = tables.next_context(&context, &[token]);
    }
    assert_eq!(token_by_token_rows, FIRST_STRETCH_ROWS);
}

#[test]
fn should_read_no_rows_for_an_empty_stretch() {
    let tables = published_tables();
    assert!(tables.row_ids(&tables.initial_context(), &[]).is_empty());
}

#[test]
fn should_start_a_request_with_only_end_of_sequence_history() {
    assert_eq!(
        published_tables().initial_context(),
        vec![END_OF_SEQUENCE, END_OF_SEQUENCE]
    );
}
