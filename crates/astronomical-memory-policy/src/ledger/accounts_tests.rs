use super::{AccountInputs, LedgerAccounts, compose_accounts};
use crate::ledger::mlx_working_buffers::MlxWorkingBufferReserve;
use crate::ledger::model_shape::ModelMemoryShape;
use crate::ledger::phase::LedgerPhase;

const GIBIBYTE: u64 = 1 << 30;
const READING: LedgerPhase = LedgerPhase::ReadingPrompt {
    span_token_count: 4_096,
};

/// A model of 3 GiB core and 20 GiB of streamable weights whose layers are
/// 1 GiB, read through two slots while a prompt is read.
const MODEL_SHAPE: ModelMemoryShape = ModelMemoryShape {
    core_weight_bytes: 3 * GIBIBYTE,
    streamable_weight_bytes: 20 * GIBIBYTE,
    reading_prompt_stream_slot_bytes: 2 * GIBIBYTE,
    writing_reply_stream_slot_bytes: GIBIBYTE / 8,
};

fn inputs_at(limit_bytes: u64) -> AccountInputs {
    AccountInputs {
        limit_bytes,
        model_shape: MODEL_SHAPE,
        context_bytes: GIBIBYTE / 2,
        phase: READING,
        mlx_working_buffer_reserve: MlxWorkingBufferReserve::Measured(GIBIBYTE),
        lease_bytes: 0,
        system_wired_reserve_bytes: 0,
    }
}

/// The identity every composition must satisfy.
fn assert_accounts_add_up(accounts: LedgerAccounts) {
    assert_eq!(
        accounts.needed_bytes() + accounts.kept_weight_capacity_bytes + accounts.unassigned_bytes,
        accounts.limit_bytes + accounts.shortfall_bytes,
        "the accounts must add up to the limit: {accounts:?}"
    );
}

#[test]
fn should_leave_the_remainder_for_kept_weights_when_the_model_must_stream() {
    let accounts = compose_accounts(inputs_at(16 * GIBIBYTE));
    let margin_bytes = 16 * GIBIBYTE / 50;
    assert_eq!(accounts.margin_bytes, margin_bytes);
    assert_eq!(accounts.stream_slot_bytes, 2 * GIBIBYTE);
    assert_eq!(
        accounts.kept_weight_capacity_bytes,
        16 * GIBIBYTE - 3 * GIBIBYTE - GIBIBYTE / 2 - GIBIBYTE - 2 * GIBIBYTE - margin_bytes
    );
    assert_eq!(accounts.unassigned_bytes, 0);
    assert_eq!(accounts.shortfall_bytes, 0);
    assert!(!accounts.keeps_every_weight(&MODEL_SHAPE));
    assert_accounts_add_up(accounts);
}

#[test]
fn should_charge_no_slots_when_every_weight_fits_without_them() {
    let accounts = compose_accounts(inputs_at(32 * GIBIBYTE));
    assert_eq!(accounts.stream_slot_bytes, 0);
    assert_eq!(accounts.kept_weight_capacity_bytes, 20 * GIBIBYTE);
    assert!(accounts.unassigned_bytes > 0);
    assert!(accounts.keeps_every_weight(&MODEL_SHAPE));
    assert_accounts_add_up(accounts);
}

#[test]
fn should_keep_slot_bytes_fewer_weights_when_one_byte_short_of_fitting() {
    let fits_without_slots = compose_accounts(inputs_at(32 * GIBIBYTE));
    let limit_that_exactly_fits = 32 * GIBIBYTE - fits_without_slots.unassigned_bytes;
    // The margin is a fraction of the limit, so search down for the first
    // limit that no longer fits everything.
    let mut limit_bytes = limit_that_exactly_fits;
    while compose_accounts(inputs_at(limit_bytes)).keeps_every_weight(&MODEL_SHAPE) {
        limit_bytes -= 1;
    }
    let just_short = compose_accounts(inputs_at(limit_bytes));
    assert_eq!(just_short.stream_slot_bytes, 2 * GIBIBYTE);
    assert!(
        just_short.kept_weight_capacity_bytes < 20 * GIBIBYTE - 2 * GIBIBYTE + GIBIBYTE / 50,
        "streaming must cost its slots: {just_short:?}"
    );
    assert_accounts_add_up(just_short);
}

#[test]
fn should_keep_nothing_and_hold_the_rest_while_mlx_working_buffers_are_unmeasured() {
    let mut account_inputs = inputs_at(16 * GIBIBYTE);
    account_inputs.mlx_working_buffer_reserve = MlxWorkingBufferReserve::Unmeasured;
    let accounts = compose_accounts(account_inputs);
    assert!(!accounts.mlx_working_buffers_are_measured);
    assert_eq!(accounts.kept_weight_capacity_bytes, 0);
    assert_eq!(accounts.stream_slot_bytes, 2 * GIBIBYTE);
    assert!(accounts.mlx_working_buffer_bytes > 0);
    assert_accounts_add_up(accounts);
}

#[test]
fn should_report_a_shortfall_instead_of_refusing_below_the_streaming_floor() {
    let accounts = compose_accounts(inputs_at(4 * GIBIBYTE));
    assert_eq!(accounts.kept_weight_capacity_bytes, 0);
    assert!(accounts.shortfall_bytes > 0);
    assert_accounts_add_up(accounts);
}

#[test]
fn should_report_a_shortfall_below_the_floor_even_when_unmeasured() {
    let mut account_inputs = inputs_at(4 * GIBIBYTE);
    account_inputs.mlx_working_buffer_reserve = MlxWorkingBufferReserve::Unmeasured;
    let accounts = compose_accounts(account_inputs);
    assert_eq!(accounts.mlx_working_buffer_bytes, 0);
    assert!(accounts.shortfall_bytes > 0);
    assert_accounts_add_up(accounts);
}

#[test]
fn should_take_a_lease_straight_out_of_the_kept_weight_capacity() {
    let without_lease = compose_accounts(inputs_at(16 * GIBIBYTE));
    let mut account_inputs = inputs_at(16 * GIBIBYTE);
    account_inputs.lease_bytes = GIBIBYTE;
    let with_lease = compose_accounts(account_inputs);
    assert_eq!(
        without_lease.kept_weight_capacity_bytes - with_lease.kept_weight_capacity_bytes,
        GIBIBYTE
    );
    assert_accounts_add_up(with_lease);
}

#[test]
fn should_need_no_mlx_working_buffers_or_slots_while_idle() {
    let mut account_inputs = inputs_at(16 * GIBIBYTE);
    account_inputs.phase = LedgerPhase::Idle;
    account_inputs.mlx_working_buffer_reserve = MlxWorkingBufferReserve::Measured(0);
    let accounts = compose_accounts(account_inputs);
    assert_eq!(accounts.mlx_working_buffer_bytes, 0);
    assert_eq!(accounts.stream_slot_bytes, 0);
    assert_accounts_add_up(accounts);
}

#[test]
fn should_use_the_small_reply_slot_while_writing_a_reply() {
    let mut account_inputs = inputs_at(16 * GIBIBYTE);
    account_inputs.phase = LedgerPhase::WritingReply;
    account_inputs.mlx_working_buffer_reserve = MlxWorkingBufferReserve::Measured(GIBIBYTE / 16);
    let accounts = compose_accounts(account_inputs);
    assert_eq!(accounts.stream_slot_bytes, GIBIBYTE / 8);
    assert!(
        accounts.kept_weight_capacity_bytes
            > compose_accounts(inputs_at(16 * GIBIBYTE)).kept_weight_capacity_bytes
    );
    assert_accounts_add_up(accounts);
}

#[test]
fn should_saturate_instead_of_wrapping_on_absurd_inputs() {
    let mut account_inputs = inputs_at(u64::MAX);
    account_inputs.context_bytes = u64::MAX;
    account_inputs.lease_bytes = u64::MAX;
    let accounts = compose_accounts(account_inputs);
    assert_eq!(accounts.kept_weight_capacity_bytes, 0);
}

fn writing_reply_inputs_at(limit_bytes: u64) -> AccountInputs {
    AccountInputs {
        phase: LedgerPhase::WritingReply,
        mlx_working_buffer_reserve: MlxWorkingBufferReserve::Measured(GIBIBYTE / 4),
        ..inputs_at(limit_bytes)
    }
}

/// What the needs come to before any room for the `mlx_buffer_cache`.
fn needed_without_mlx_buffer_cache_bytes(accounts: LedgerAccounts) -> u64 {
    accounts.needed_bytes() - accounts.mlx_buffer_cache_bytes
}

#[test]
fn should_set_aside_one_reply_steps_working_buffers_for_the_mlx_buffer_cache() {
    let accounts = compose_accounts(writing_reply_inputs_at(20 * GIBIBYTE));

    assert_eq!(accounts.mlx_buffer_cache_bytes, GIBIBYTE / 4);
    assert_eq!(
        accounts.kept_weight_capacity_bytes,
        accounts.limit_bytes - needed_without_mlx_buffer_cache_bytes(accounts) - GIBIBYTE / 4,
        "the room comes out of what would otherwise be kept"
    );
    assert_eq!(accounts.shortfall_bytes, 0);
    assert_accounts_add_up(accounts);
}

#[test]
fn should_set_nothing_aside_for_the_mlx_buffer_cache_while_reading_a_prompt_or_idle() {
    let reading_accounts = compose_accounts(inputs_at(20 * GIBIBYTE));
    let idle_accounts = compose_accounts(AccountInputs {
        phase: LedgerPhase::Idle,
        ..inputs_at(20 * GIBIBYTE)
    });

    assert_eq!(reading_accounts.mlx_buffer_cache_bytes, 0);
    assert_eq!(idle_accounts.mlx_buffer_cache_bytes, 0);
    assert_accounts_add_up(reading_accounts);
    assert_accounts_add_up(idle_accounts);
}

#[test]
fn should_shrink_the_mlx_buffer_cache_room_to_what_fits_beside_the_needs() {
    let roomy_accounts = compose_accounts(writing_reply_inputs_at(20 * GIBIBYTE));
    let needed_bytes = needed_without_mlx_buffer_cache_bytes(roomy_accounts);
    // A tenth of a GiB to spare, less than the quarter the room would take;
    // the margin is a fiftieth of the limit, so it is recomputed here.
    let tight_limit_bytes = (needed_bytes - roomy_accounts.margin_bytes) * 50 / 49 + GIBIBYTE / 10;

    let tight_accounts = compose_accounts(writing_reply_inputs_at(tight_limit_bytes));

    assert!(tight_accounts.mlx_buffer_cache_bytes > 0);
    assert!(tight_accounts.mlx_buffer_cache_bytes < GIBIBYTE / 4);
    assert_eq!(tight_accounts.kept_weight_capacity_bytes, 0);
    assert_eq!(tight_accounts.shortfall_bytes, 0);
    assert_accounts_add_up(tight_accounts);
}

#[test]
fn should_never_add_the_mlx_buffer_cache_room_to_a_shortfall() {
    let accounts = compose_accounts(writing_reply_inputs_at(2 * GIBIBYTE));

    let needs_that_serving_requires_bytes = MODEL_SHAPE.core_weight_bytes
        + GIBIBYTE / 2
        + GIBIBYTE / 4
        + MODEL_SHAPE.writing_reply_stream_slot_bytes
        + accounts.margin_bytes;
    assert_eq!(accounts.mlx_buffer_cache_bytes, 0);
    assert_eq!(
        accounts.shortfall_bytes,
        needs_that_serving_requires_bytes - accounts.limit_bytes,
        "a limit too small to serve at is exactly as short as its required needs make it"
    );
    assert_accounts_add_up(accounts);
}

#[test]
fn should_take_the_system_wired_reserve_from_what_would_be_kept() {
    let accounts_without_reserve = compose_accounts(writing_reply_inputs_at(20 * GIBIBYTE));
    let accounts = compose_accounts(AccountInputs {
        system_wired_reserve_bytes: GIBIBYTE / 2,
        ..writing_reply_inputs_at(20 * GIBIBYTE)
    });

    assert_eq!(accounts.system_wired_reserve_bytes, GIBIBYTE / 2);
    assert_eq!(
        accounts.kept_weight_capacity_bytes,
        accounts_without_reserve.kept_weight_capacity_bytes - GIBIBYTE / 2
    );
    assert_eq!(accounts.mlx_buffer_cache_bytes, GIBIBYTE / 4);
    assert_accounts_add_up(accounts);
}

#[test]
fn should_shrink_the_system_wired_reserve_to_what_fits_and_leave_no_shortfall() {
    let accounts_without_reserve = compose_accounts(writing_reply_inputs_at(20 * GIBIBYTE));
    let accounts = compose_accounts(AccountInputs {
        system_wired_reserve_bytes: 40 * GIBIBYTE,
        ..writing_reply_inputs_at(20 * GIBIBYTE)
    });

    assert_eq!(
        accounts.system_wired_reserve_bytes,
        accounts_without_reserve.kept_weight_capacity_bytes
            + accounts_without_reserve.mlx_buffer_cache_bytes,
        "the reserve takes all the optional room and no more"
    );
    assert_eq!(accounts.kept_weight_capacity_bytes, 0);
    assert_eq!(accounts.mlx_buffer_cache_bytes, 0);
    assert_eq!(accounts.shortfall_bytes, 0);
    assert_accounts_add_up(accounts);
}

#[test]
fn should_reserve_nothing_for_the_system_on_a_limit_too_small_to_serve_at() {
    let accounts = compose_accounts(AccountInputs {
        system_wired_reserve_bytes: GIBIBYTE,
        ..writing_reply_inputs_at(2 * GIBIBYTE)
    });

    assert_eq!(accounts.system_wired_reserve_bytes, 0);
    assert!(accounts.shortfall_bytes > 0);
    assert_accounts_add_up(accounts);
}
