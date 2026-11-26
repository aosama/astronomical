//! Beside-source coverage for binding a denoising step's inputs. The forward pass and the Euler
//! update need the transformer on the GPU, so the model-serving image lifecycle runs cover them.

use super::DenoisingInputs;

#[test]
fn should_name_the_transformer_first_when_every_input_is_missing() {
    let rejection = DenoisingInputs::bind(None, None, None, None, None)
        .map(|_| ())
        .expect_err("nothing is bound");

    assert_eq!(rejection, "transformer is unavailable");
}
