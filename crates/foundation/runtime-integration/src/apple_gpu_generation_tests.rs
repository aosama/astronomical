use super::{apple_gpu_generation, is_m5_or_newer_gpu};

#[test]
fn should_read_the_generation_of_an_m5_pro() {
    assert_eq!(apple_gpu_generation("applegpu_g17s"), Some(17));
}

#[test]
fn should_count_an_m5_pro_and_max_as_m5_or_newer() {
    assert!(is_m5_or_newer_gpu("applegpu_g17g"));
    assert!(is_m5_or_newer_gpu("applegpu_g17s"));
    assert!(is_m5_or_newer_gpu("applegpu_g18d"));
}

#[test]
fn should_count_an_m4_as_older_than_m5() {
    assert!(!is_m5_or_newer_gpu("applegpu_g16s"));
    assert!(!is_m5_or_newer_gpu("applegpu_g13g"));
}

#[test]
fn should_need_one_more_generation_on_a_phone_as_mlx_does() {
    assert!(!is_m5_or_newer_gpu("applegpu_g17p"));
    assert!(is_m5_or_newer_gpu("applegpu_g18p"));
}

#[test]
fn should_refuse_a_name_without_two_digits_and_a_class_letter() {
    assert_eq!(apple_gpu_generation(""), None);
    assert_eq!(apple_gpu_generation("g7"), None);
    assert_eq!(apple_gpu_generation("applegpu"), None);
    assert_eq!(apple_gpu_generation("applegpu_g17"), None);
    assert!(!is_m5_or_newer_gpu("not a gpu"));
}
