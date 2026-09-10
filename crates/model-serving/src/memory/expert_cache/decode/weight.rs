//! Opaque per-expert payload the decode cache stores but never inspects.

/// One expert's weights, supplied by a model family.
///
/// The cache uses only the byte count. Families own tensors, stacking, and I/O.
pub trait ResidentExpertWeight: std::fmt::Debug {
    fn payload_bytes(&self) -> u64;
}
