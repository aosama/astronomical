//! The transform pair must invert itself: if the analysis basis, the
//! transposed-convolution orientation or the envelope were wrong, a signal
//! would not come back. Values come from the Romeo and Juliet fixture.

use serial_test::serial;

use crate::speech_models::speech_test_support::{initialize_test_runtime, shakespeare_values};

use super::{HOP_SIZE, ShortTimeFourier, inverse_window_envelope, reflect_padded_indices};

#[test]
#[serial]
fn should_reconstruct_a_signal_from_its_own_spectrum() {
    let runtime = initialize_test_runtime();
    let fourier = ShortTimeFourier::new(&runtime).expect("the basis builds");
    let signal_values = shakespeare_values(600, 0);
    let signal = runtime
        .array_from_f32(&signal_values, &[600])
        .expect("upload");
    let spectrum = fourier
        .magnitude_and_phase(&runtime, &signal)
        .expect("analysis");
    assert_eq!(spectrum.shape(), [600 / HOP_SIZE as i32 + 1, 22]);
    let magnitude = runtime
        .slice(&spectrum, &[0, 0], &[121, 11], &[1, 1])
        .expect("slice");
    let phase = runtime
        .slice(&spectrum, &[0, 11], &[121, 22], &[1, 1])
        .expect("slice");
    let rebuilt = runtime
        .array_to_vec_f32(
            &fourier
                .inverse(&runtime, &magnitude, &phase)
                .expect("synthesis"),
        )
        .expect("read");
    assert_eq!(rebuilt.len(), 600);
    let worst_error = signal_values
        .iter()
        .zip(&rebuilt)
        .map(|(original, back)| (original - back).abs())
        .fold(0.0f32, f32::max);
    // A wrong basis, window or overlap orientation gives errors near the
    // signal's own size (about 1). What remains is float32 rounding through
    // the GPU's polar round trip, measured at about 1e-3 at worst and spread
    // evenly over the signal, far below anything audible.
    assert!(
        worst_error < 5e-3,
        "worst reconstruction error {worst_error}"
    );
}

#[test]
#[serial]
fn should_give_the_dc_bin_of_a_negative_signal_a_phase_of_plus_pi() {
    let runtime = initialize_test_runtime();
    let fourier = ShortTimeFourier::new(&runtime).expect("the basis builds");
    let signal = runtime.array_from_f32(&[-1.0; 40], &[40]).expect("upload");
    let spectrum = runtime
        .array_to_vec_f32(
            &fourier
                .magnitude_and_phase(&runtime, &signal)
                .expect("analysis"),
        )
        .expect("read");
    let first_frame_dc_phase = spectrum[11];
    assert!(
        (first_frame_dc_phase - std::f32::consts::PI).abs() < 1e-6,
        "{first_frame_dc_phase}"
    );
}

#[test]
fn should_reflect_without_repeating_the_edge_sample() {
    let indices = reflect_padded_indices(12);
    assert_eq!(&indices[..11], &[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    assert_eq!(&indices[indices.len() - 3..], &[3, 2, 1]);
}

// Edge: the very first sample is covered by one window's zero tip only, so it
// keeps a zero normaliser instead of dividing by nothing.
#[test]
fn should_leave_samples_no_window_reaches_unscaled() {
    let envelope = inverse_window_envelope(3);
    assert_eq!(envelope[0], 0.0);
    assert!(envelope[10] > 0.0);
}
