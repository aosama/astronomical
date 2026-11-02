//! The float32 KL divergence reference against the Python formula on the
//! fixture logits pair.

use super::fixtures;

#[test]
#[ignore = "pending implementation of kl_divergence_reference_vs_probe"]
fn should_reproduce_the_python_kl_divergence_on_the_fixture_logits() {
    let fixture = fixtures::encoder_parity();
    let kl_fixture = &fixture.kl_divergence;

    let kl =
        astronomical_quantization::sensitivity::kl_divergence::kl_divergence_reference_vs_probe(
            &kl_fixture.reference_logits,
            &kl_fixture.probe_logits,
            kl_fixture.shape[2],
        );

    assert!((kl - kl_fixture.kl_reference_vs_probe).abs() < 1e-6);
}
