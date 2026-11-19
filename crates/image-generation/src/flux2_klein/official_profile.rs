//! Constants that identify the first executable distilled profile.

/// Request constants fixed by the reviewed FLUX.2 Klein profile.
pub struct Flux2KleinOfficialProfile;

impl Flux2KleinOfficialProfile {
    pub const fn guidance_thousandths() -> u32 {
        1_000
    }

    pub const fn inference_step_count() -> usize {
        4
    }
}
