use astronomical_neural_engine::program_shape::ProgramShape;

use super::{measured_for, program_surface_bytes};

#[test]
fn should_count_both_surfaces_as_fp16() {
    let shape = ProgramShape::new(4096, 2048, 960).expect("a valid shape");

    assert_eq!(program_surface_bytes(shape), (4096 + 2048) * 960 * 2);
}

#[test]
fn should_count_the_smallest_shape() {
    let shape = ProgramShape::new(1, 1, 1).expect("a valid shape");

    assert_eq!(program_surface_bytes(shape), 4);
}

#[test]
fn should_name_the_host_and_what_else_a_reading_depends_on() {
    let measured_for = measured_for("macOS 27.0.1 (26A434) on Mac99,1");

    assert!(measured_for.starts_with("macOS 27.0.1 (26A434) on Mac99,1"));
    assert!(measured_for.contains("measurement method"));
    assert!(measured_for.contains("output gates"));
}

#[test]
fn should_stay_on_one_line_because_the_record_is_read_by_lines() {
    assert_eq!(measured_for("macOS 27 on Mac99,1").lines().count(), 1);
}
