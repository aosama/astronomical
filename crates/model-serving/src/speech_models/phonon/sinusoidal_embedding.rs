//! The sinusoidal embedding of one scalar that Phonon's continuous
//! conditioners use: cosines then sines of
//! `value / max_period^(index / (half - 1))`.

/// The embedding is `width` wide; a width under 4 or odd has no defined
/// frequencies and gives `None`.
pub(crate) fn sinusoidal_embedding(
    embedded_value: f32,
    width: usize,
    max_period: f32,
) -> Option<Vec<f32>> {
    if width < 4 || !width.is_multiple_of(2) {
        return None;
    }
    let half_width = width / 2;
    let phases: Vec<f32> = (0..half_width)
        .map(|index| embedded_value / max_period.powf(index as f32 / (half_width - 1) as f32))
        .collect();
    Some(
        phases
            .iter()
            .map(|phase| phase.cos())
            .chain(phases.iter().map(|phase| phase.sin()))
            .collect(),
    )
}

#[cfg(test)]
#[path = "sinusoidal_embedding_tests.rs"]
mod tests;
