use crate::PreparedInferenceRequest;

/// Prepared K2 Horizon MoVA generation request accepted by the family engine.
#[derive(Debug)]
pub struct K2HorizonMoVAInferenceRequest {
    prompt_token_ids: Vec<u32>,
    max_output_tokens: u32,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    seed: Option<u64>,
    thinking_budget: Option<u16>,
    forced_thinking_transition_token_ids: Vec<u32>,
    natural_reasoning_end_token_ids: Vec<u32>,
}

impl K2HorizonMoVAInferenceRequest {
    #[must_use]
    pub fn new(
        prompt_token_ids: Vec<u32>,
        max_output_tokens: u32,
        temperature_thousandths: u16,
        top_p_thousandths: u16,
        seed: Option<u64>,
    ) -> Self {
        Self::with_thinking_budget(
            prompt_token_ids,
            max_output_tokens,
            temperature_thousandths,
            top_p_thousandths,
            seed,
            None,
            Vec::new(),
            Vec::new(),
        )
    }

    #[must_use]
    pub fn with_thinking_budget(
        prompt_token_ids: Vec<u32>,
        max_output_tokens: u32,
        temperature_thousandths: u16,
        top_p_thousandths: u16,
        seed: Option<u64>,
        thinking_budget: Option<u16>,
        forced_thinking_transition_token_ids: Vec<u32>,
        natural_reasoning_end_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            prompt_token_ids,
            max_output_tokens,
            temperature_thousandths,
            top_p_thousandths,
            seed,
            thinking_budget,
            forced_thinking_transition_token_ids,
            natural_reasoning_end_token_ids,
        }
    }

    #[must_use]
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    #[must_use]
    pub const fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens
    }

    #[must_use]
    pub const fn temperature_thousandths(&self) -> u16 {
        self.temperature_thousandths
    }

    #[must_use]
    pub const fn top_p_thousandths(&self) -> u16 {
        self.top_p_thousandths
    }

    #[must_use]
    pub const fn seed(&self) -> Option<u64> {
        self.seed
    }

    #[must_use]
    pub const fn thinking_budget(&self) -> Option<u16> {
        self.thinking_budget
    }

    #[must_use]
    pub fn forced_thinking_transition_token_ids(&self) -> &[u32] {
        &self.forced_thinking_transition_token_ids
    }

    #[must_use]
    pub fn natural_reasoning_end_token_ids(&self) -> &[u32] {
        &self.natural_reasoning_end_token_ids
    }
}

impl PreparedInferenceRequest for K2HorizonMoVAInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        self.prompt_token_ids.len()
    }
}
