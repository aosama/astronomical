mod common;
#[path = "common/flux2_klein_reference_oracle.rs"]
#[allow(dead_code)] // All-feature checks expose readers consumed only by the qualification binary.
mod flux2_klein_reference_oracle;
mod hermetic;
