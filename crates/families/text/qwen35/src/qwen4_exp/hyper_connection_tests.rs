use super::lane_split_shape;

#[test]
fn should_split_the_lane_axis_off_the_feature_axis() {
    assert_eq!(
        lane_split_shape(&[1, 7, 10240], 4, 2560),
        vec![1, 7, 4, 2560]
    );
}

#[test]
fn should_split_a_single_token_row_the_same_way() {
    assert_eq!(lane_split_shape(&[1, 1, 8], 2, 4), vec![1, 1, 2, 4]);
}

#[test]
fn should_keep_every_leading_axis() {
    assert_eq!(lane_split_shape(&[2, 3, 5, 12], 3, 4), vec![2, 3, 5, 3, 4]);
}
