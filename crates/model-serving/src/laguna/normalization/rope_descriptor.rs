/// Default rotary position embedding parameters for one attention kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LagunaDefaultRopeDescriptor {
    rope_theta: f64,
    rotary_dimension: u32,
}

impl LagunaDefaultRopeDescriptor {
    pub(super) const fn new(rope_theta: f64, rotary_dimension: u32) -> Self {
        Self {
            rope_theta,
            rotary_dimension,
        }
    }

    /// Returns the rotary frequency base.
    #[must_use]
    pub const fn rope_theta(&self) -> f64 {
        self.rope_theta
    }

    /// Returns the positive even head width receiving rotary embedding.
    #[must_use]
    pub const fn rotary_dimension(&self) -> u32 {
        self.rotary_dimension
    }
}

/// YaRN rotary position embedding parameters for one attention kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LagunaYarnRopeDescriptor {
    rope_theta: f64,
    factor: f64,
    original_maximum_position_count: u32,
    beta_slow: f64,
    beta_fast: f64,
    attention_factor: f64,
    rotary_dimension: u32,
}

impl LagunaYarnRopeDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub(super) const fn new(
        rope_theta: f64,
        factor: f64,
        original_maximum_position_count: u32,
        beta_slow: f64,
        beta_fast: f64,
        attention_factor: f64,
        rotary_dimension: u32,
    ) -> Self {
        Self {
            rope_theta,
            factor,
            original_maximum_position_count,
            beta_slow,
            beta_fast,
            attention_factor,
            rotary_dimension,
        }
    }

    /// Returns the rotary frequency base.
    #[must_use]
    pub const fn rope_theta(&self) -> f64 {
        self.rope_theta
    }

    /// Returns the YaRN context scaling factor without imposing fixture-specific equality.
    #[must_use]
    pub const fn factor(&self) -> f64 {
        self.factor
    }

    /// Returns the unscaled context position count.
    #[must_use]
    pub const fn original_maximum_position_count(&self) -> u32 {
        self.original_maximum_position_count
    }

    /// Returns the slow-rotation correction boundary.
    #[must_use]
    pub const fn beta_slow(&self) -> f64 {
        self.beta_slow
    }

    /// Returns the fast-rotation correction boundary.
    #[must_use]
    pub const fn beta_fast(&self) -> f64 {
        self.beta_fast
    }

    /// Returns the YaRN attention magnitude multiplier.
    #[must_use]
    pub const fn attention_factor(&self) -> f64 {
        self.attention_factor
    }

    /// Returns the positive even head width receiving rotary embedding.
    #[must_use]
    pub const fn rotary_dimension(&self) -> u32 {
        self.rotary_dimension
    }
}

/// Canonical default or YaRN rotary policy selected for an active attention kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LagunaRopeDescriptor {
    Default(LagunaDefaultRopeDescriptor),
    Yarn(LagunaYarnRopeDescriptor),
}

impl LagunaRopeDescriptor {
    /// Returns the rotary frequency base independent of scaling policy.
    #[must_use]
    pub const fn rope_theta(&self) -> f64 {
        match self {
            Self::Default(descriptor) => descriptor.rope_theta(),
            Self::Yarn(descriptor) => descriptor.rope_theta(),
        }
    }

    /// Returns the positive even head width receiving rotary embedding.
    #[must_use]
    pub const fn rotary_dimension(&self) -> u32 {
        match self {
            Self::Default(descriptor) => descriptor.rotary_dimension(),
            Self::Yarn(descriptor) => descriptor.rotary_dimension(),
        }
    }
}
