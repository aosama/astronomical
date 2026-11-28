use super::{MtpHeadKind, resolved_mtp_draft_token_count};

#[test]
fn should_draft_nothing_without_a_hosted_head_whatever_is_configured() {
    assert_eq!(resolved_mtp_draft_token_count(Some(3), None), 0);
    assert_eq!(resolved_mtp_draft_token_count(None, None), 0);
}

#[test]
fn should_draft_one_token_a_round_on_flash_next_when_left_open() {
    assert_eq!(
        resolved_mtp_draft_token_count(None, Some(MtpHeadKind::FlashNext)),
        1
    );
}

#[test]
fn should_draft_nothing_on_a_dense_pack_when_left_open() {
    assert_eq!(
        resolved_mtp_draft_token_count(None, Some(MtpHeadKind::DenseQwen3_5)),
        0
    );
}

#[test]
fn should_draft_the_configured_count_on_either_kind() {
    assert_eq!(
        resolved_mtp_draft_token_count(Some(3), Some(MtpHeadKind::DenseQwen3_5)),
        3
    );
    assert_eq!(
        resolved_mtp_draft_token_count(Some(0), Some(MtpHeadKind::FlashNext)),
        0
    );
}
