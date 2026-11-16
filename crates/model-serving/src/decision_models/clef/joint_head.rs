//! Clef's joint schema head: the backbone's states over the whole sequence
//! in, one logit per option of every question out, as `JointSchemaHead`
//! computes them.
//!
//! The head reads four things from the backbone: every position (projected
//! into a memory the options and fields attend to), the last position (a
//! global summary), the mean over each question's instructions, and the mean
//! over each option's text, beside the mean of the option tokens' output
//! embeddings. Options are routed through the memory, each question becomes
//! a field that sees every other question, and an option's logit is a
//! lexical prior plus a gated agreement between its field and itself.
//!
//! Means over spans are taken as one matrix product with an averaging
//! matrix built on the host, rather than one slice per span: a request may
//! ask about hundreds of options, and one product keeps the graph small.

use astronomical_decision_contract::{DecisionFailure, HeadInputRequest};
use mlx_bridge::MlxArray;

use super::head_layers::{ClefHeadLayers, head_math};
use super::prompt::{ClefSequence, TokenSpan};

/// `functional.normalize`'s floor on a norm.
const NORMALIZE_EPSILON: f32 = 1e-12;
/// `cosine_similarity`'s floor on the product of norms.
const COSINE_EPSILON: f32 = 1e-8;
/// The ceiling `clamp(max=log(100))` puts on both logit scales.
const MAXIMUM_LOGIT_SCALE: f64 = 100.0;
/// What a question's softmax over its own options adds to every other
/// question's options: small enough that `exp` gives exactly zero.
const OTHER_QUESTION_SCORE_BIAS: f32 = -1.0e30;

/// What the head reads for one request, already laid out on the device.
pub(crate) struct ClefHeadInputs<'inputs> {
    /// `[length, hidden]` final normalized backbone states.
    pub(crate) hidden_states: &'inputs MlxArray,
    /// `[option tokens, hidden]` output-embedding rows of every option's
    /// tokens, option by option in scoring order.
    pub(crate) option_token_rows: &'inputs MlxArray,
    pub(crate) sequence: &'inputs ClefSequence,
}

/// The per-option terms the final logit is combined from.
struct OptionTerms {
    lexical_prior: Vec<f32>,
    cosine: Vec<f32>,
    residual: Vec<f32>,
    /// `[options, width]` field and normalized option rows, read out only
    /// when the request asked for head inputs.
    field_and_option_rows: Option<FieldAndOptionRows>,
}

impl OptionTermArrays {
    /// `[options, 4·width]`: `[field, option, field · option, |field − option|]`,
    /// what the residual scorer reads.
    pub(crate) fn residual_features(
        &self,
        runtime: &astronomical_runtime_integration::MlxRuntime,
    ) -> Result<MlxArray, DecisionFailure> {
        let (field_per_option, options) = (&self.field_per_option, &self.options);
        let product = runtime
            .multiply(field_per_option, options)
            .map_err(head_math)?;
        let distance = runtime
            .abs(
                &runtime
                    .subtract(field_per_option, options)
                    .map_err(head_math)?,
            )
            .map_err(head_math)?;
        runtime
            .concatenate_axis(&[field_per_option, options, &product, &distance], 1)
            .map_err(head_math)
    }
}

/// One question's per-option head inputs, in the order Clef scored them.
pub(crate) struct ClefQuestionReadout {
    pub(crate) field_vectors: Vec<Vec<f32>>,
    pub(crate) option_vectors: Vec<Vec<f32>>,
    pub(crate) lexical_priors: Vec<f32>,
    pub(crate) cosines: Vec<f32>,
}

/// Every question's logits and, when asked, its head inputs, in scoring order.
pub(crate) struct JointHeadScores {
    pub(crate) question_logits: Vec<Vec<f64>>,
    pub(crate) question_readouts: Option<Vec<ClefQuestionReadout>>,
}

/// Everything the final mix reads, per option in scoring order, as lazy
/// arrays: what serving reads back to the host, and what a whole-model
/// training mixes on the GPU with its trainable final layers.
pub(crate) struct OptionTermArrays {
    /// `[options]`.
    pub(crate) lexical_prior: MlxArray,
    /// `[options]`.
    pub(crate) cosine: MlxArray,
    /// `[options, width]` each option's field, after the field norm.
    pub(crate) field_per_option: MlxArray,
    /// `[options, width]` each routed option, after the option norm.
    pub(crate) options: MlxArray,
    /// How many options each question has, in question order.
    pub(crate) option_counts: Vec<usize>,
}

impl ClefHeadLayers<'_> {
    /// One logit per option, per question, in scoring order; with
    /// [`HeadInputRequest::Include`] also what the residual scorer and the
    /// final mix read for each option.
    pub(crate) fn joint_option_logits(
        &self,
        head_inputs: &ClefHeadInputs<'_>,
        routing_layer_count: usize,
        decoder_layer_count: usize,
        head_input_request: HeadInputRequest,
    ) -> Result<JointHeadScores, DecisionFailure> {
        let term_arrays =
            self.option_term_arrays(head_inputs, routing_layer_count, decoder_layer_count)?;
        let option_terms = self.option_terms(&term_arrays, head_input_request)?;
        let question_logits = self.combined_logits(&option_terms, &term_arrays.option_counts)?;
        let question_readouts = match &option_terms.field_and_option_rows {
            Some(feature_rows) => Some(question_readouts(
                &option_terms,
                &feature_rows.field_rows,
                &feature_rows.option_rows,
                &term_arrays.option_counts,
            )),
            None => None,
        };
        Ok(JointHeadScores {
            question_logits,
            question_readouts,
        })
    }

    /// The head up to the final mix, lazily: pooling, evidence routing, the
    /// field decoder, and each option's lexical prior, cosine, field and
    /// normalized option.
    pub(crate) fn option_term_arrays(
        &self,
        head_inputs: &ClefHeadInputs<'_>,
        routing_layer_count: usize,
        decoder_layer_count: usize,
    ) -> Result<OptionTermArrays, DecisionFailure> {
        let pooled = self.pooled_inputs(head_inputs)?;
        let (routed_options, fields) =
            self.routed_options_and_fields(&pooled, routing_layer_count, decoder_layer_count)?;
        let runtime = self.runtime;
        let anchors = self.unit_rows(
            &runtime
                .add(&pooled.question_vectors, &pooled.global_vector)
                .map_err(head_math)?,
        )?;
        let anchor_per_option = runtime
            .take_axis(&anchors, &pooled.question_of_option, 0)
            .map_err(head_math)?;
        let lexical_anchors = self.unit_rows(&pooled.lexical_options)?;
        let lexical_prior = self.row_dot(&lexical_anchors, &anchor_per_option)?;
        let options = self.layer_norm("option_norm", &routed_options)?;
        let field_per_option = runtime
            .take_axis(&fields, &pooled.question_of_option, 0)
            .map_err(head_math)?;
        let cosine = self.row_cosine(&field_per_option, &options)?;
        Ok(OptionTermArrays {
            lexical_prior,
            cosine,
            field_per_option,
            options,
            option_counts: pooled.option_counts,
        })
    }

    /// The options after evidence routing, and each question's field after
    /// the field decoder and its norm.
    fn routed_options_and_fields(
        &self,
        pooled: &PooledInputs,
        routing_layer_count: usize,
        decoder_layer_count: usize,
    ) -> Result<(MlxArray, MlxArray), DecisionFailure> {
        let mut routed_options = self.option_queries(pooled)?;
        for routing_layer_index in 0..routing_layer_count {
            routed_options = self.evidence_routing_layer(
                &format!("evidence_layers.{routing_layer_index}"),
                &routed_options,
                &pooled.memory,
            )?;
        }
        let routed_options = self.rows_of(&routed_options)?;
        let mut fields = self.initial_fields(pooled, &routed_options)?;
        for decoder_layer_index in 0..decoder_layer_count {
            fields = self.decoder_layer(
                &format!("layers.{decoder_layer_index}"),
                &fields,
                &pooled.memory,
            )?;
        }
        let fields = self.layer_norm("field_norm", &self.rows_of(&fields)?)?;
        Ok((routed_options, fields))
    }

    fn pooled_inputs(
        &self,
        head_inputs: &ClefHeadInputs<'_>,
    ) -> Result<PooledInputs, DecisionFailure> {
        let runtime = self.runtime;
        let sequence = head_inputs.sequence;
        let sequence_length = sequence.token_ids.len();
        let normalized = self.layer_norm("hidden_norm", head_inputs.hidden_states)?;
        let memory_rows = self.linear("memory_projection", &normalized, false)?;
        let width = memory_rows.shape()[1];
        let memory = runtime
            .reshape(&memory_rows, &[1, sequence_length as i32, width])
            .map_err(head_math)?;
        let hidden_size = normalized.shape()[1];
        let global_vector = runtime
            .slice(
                &normalized,
                &[sequence_length as i32 - 1, 0],
                &[sequence_length as i32, hidden_size],
                &[1, 1],
            )
            .map_err(head_math)?;
        let layout = SpanLayout::of(sequence);
        let question_vectors = self.pooled(&layout.question_spans, sequence_length, &normalized)?;
        let option_contexts = self.pooled(&layout.option_spans, sequence_length, &normalized)?;
        let lexical_options = self.pooled(
            &layout.option_token_spans,
            layout.option_token_count,
            head_inputs.option_token_rows,
        )?;
        let question_of_option = runtime
            .array_from_i32(
                &layout.question_of_option,
                &[layout.question_of_option.len() as i32],
            )
            .map_err(head_math)?;
        let type_ids = runtime
            .array_from_i32(&layout.type_ids, &[layout.type_ids.len() as i32])
            .map_err(head_math)?;
        let membership_bias = runtime
            .array_from_f32(
                &layout.membership_bias,
                &[
                    layout.question_spans.len() as i32,
                    layout.option_spans.len() as i32,
                ],
            )
            .map_err(head_math)?;
        Ok(PooledInputs {
            memory,
            global_vector,
            question_vectors,
            option_contexts,
            lexical_options,
            question_of_option,
            type_ids,
            membership_bias,
            option_counts: layout.option_counts,
        })
    }

    /// The mean of `rows` over each span: `[spans, width]`.
    fn pooled(
        &self,
        spans: &[TokenSpan],
        row_count: usize,
        rows: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let averaging_values = averaging_matrix(spans, row_count);
        let averaging = self
            .runtime
            .array_from_f32(&averaging_values, &[spans.len() as i32, row_count as i32])
            .map_err(head_math)?;
        self.runtime.matmul(&averaging, rows).map_err(head_math)
    }

    /// Each option's query before routing, `[1, options, width]`.
    fn option_queries(&self, pooled: &PooledInputs) -> Result<MlxArray, DecisionFailure> {
        let runtime = self.runtime;
        let question_vector_per_option = runtime
            .take_axis(&pooled.question_vectors, &pooled.question_of_option, 0)
            .map_err(head_math)?;
        let context_part =
            self.linear("option_context_projection", &pooled.option_contexts, false)?;
        let lexical_part =
            self.linear("option_lexical_projection", &pooled.lexical_options, false)?;
        let question_part = self.linear(
            "option_question_projection",
            &question_vector_per_option,
            false,
        )?;
        let option_queries = runtime
            .add(
                &runtime
                    .add(&context_part, &lexical_part)
                    .map_err(head_math)?,
                &question_part,
            )
            .map_err(head_math)?;
        let query_shape = option_queries.shape();
        runtime
            .reshape(&option_queries, &[1, query_shape[0], query_shape[1]])
            .map_err(head_math)
    }

    /// Each question's field before the decoder layers, `[1, questions, width]`:
    /// its projection, a summary of its routed options weighted by agreement
    /// with that projection, the global vector and its type.
    fn initial_fields(
        &self,
        pooled: &PooledInputs,
        routed_options: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let runtime = self.runtime;
        let base_fields = self.linear("question_projection", &pooled.question_vectors, false)?;
        let summaries = self.option_summaries(pooled, routed_options, &base_fields)?;
        let summary_part = self.layer_norm("option_summary_norm", &summaries)?;
        let global_part = self.linear("global_projection", &pooled.global_vector, false)?;
        let type_part = runtime
            .take_axis(
                self.weights.tensor("type_embedding.weight")?,
                &pooled.type_ids,
                0,
            )
            .map_err(head_math)?;
        let mut fields = runtime
            .add(&base_fields, &summary_part)
            .map_err(head_math)?;
        fields = runtime.add(&fields, &global_part).map_err(head_math)?;
        fields = runtime.add(&fields, &type_part).map_err(head_math)?;
        let fields_shape = fields.shape();
        runtime
            .reshape(&fields, &[1, fields_shape[0], fields_shape[1]])
            .map_err(head_math)
    }

    /// Each question's routed options averaged by a softmax, over that
    /// question's own options only, of their agreement with its field.
    fn option_summaries(
        &self,
        pooled: &PooledInputs,
        routed_options: &MlxArray,
        base_fields: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let runtime = self.runtime;
        let field_per_option = runtime
            .take_axis(base_fields, &pooled.question_of_option, 0)
            .map_err(head_math)?;
        let width = base_fields.shape()[1];
        let agreement = self.row_dot(routed_options, &field_per_option)?;
        let scaled_agreement = runtime
            .multiply_scalar(&agreement, 1.0 / (width as f32).sqrt())
            .map_err(head_math)?;
        let option_count = scaled_agreement.shape()[0];
        let agreement_row = runtime
            .reshape(&scaled_agreement, &[1, option_count])
            .map_err(head_math)?;
        let biased_agreement = runtime
            .add(&agreement_row, &pooled.membership_bias)
            .map_err(head_math)?;
        let routing_weights = runtime
            .softmax_axis(&biased_agreement, 1)
            .map_err(head_math)?;
        runtime
            .matmul(&routing_weights, routed_options)
            .map_err(head_math)
    }

    /// The three per-option terms, read back to the host.
    fn option_terms(
        &self,
        term_arrays: &OptionTermArrays,
        head_input_request: HeadInputRequest,
    ) -> Result<OptionTerms, DecisionFailure> {
        let residual = self.residual_scores(&term_arrays.residual_features(self.runtime)?)?;
        self.runtime
            .evaluate_arrays(&[&term_arrays.lexical_prior, &term_arrays.cosine, &residual])
            .map_err(head_math)?;
        let field_and_option_rows = field_and_option_rows(
            &term_arrays.field_per_option,
            &term_arrays.options,
            head_input_request,
        )?;
        Ok(OptionTerms {
            lexical_prior: term_arrays.lexical_prior.to_vec_f32().map_err(head_math)?,
            cosine: term_arrays.cosine.to_vec_f32().map_err(head_math)?,
            residual: residual.to_vec_f32().map_err(head_math)?,
            field_and_option_rows,
        })
    }

    /// `residual_scorer` over the residual features.
    fn residual_scores(&self, features: &MlxArray) -> Result<MlxArray, DecisionFailure> {
        let runtime = self.runtime;
        let hidden = runtime
            .gelu(&self.linear("residual_scorer.0", features, true)?)
            .map_err(head_math)?;
        let single_score = self.linear("residual_scorer.3", &hidden, true)?;
        let option_count = single_score.shape()[0];
        runtime
            .reshape(&single_score, &[option_count])
            .map_err(head_math)
    }

    /// `prior + sigmoid(gate) · (joint_scale · cosine + residual)`, per
    /// question, with both scales clamped as the reference clamps them.
    fn combined_logits(
        &self,
        option_terms: &OptionTerms,
        option_counts: &[usize],
    ) -> Result<Vec<Vec<f64>>, DecisionFailure> {
        let prior_scale = self
            .weights
            .scalar("prior_logit_scale")?
            .min(MAXIMUM_LOGIT_SCALE.ln())
            .exp();
        let joint_scale = self
            .weights
            .scalar("joint_logit_scale")?
            .min(MAXIMUM_LOGIT_SCALE.ln())
            .exp();
        let gate = 1.0 / (1.0 + (-self.weights.scalar("residual_gate")?).exp());
        let mut question_logits = Vec::with_capacity(option_counts.len());
        let mut option_index = 0;
        for option_count in option_counts {
            let mut option_logits = Vec::with_capacity(*option_count);
            for _ in 0..*option_count {
                let joint = joint_scale * f64::from(option_terms.cosine[option_index])
                    + f64::from(option_terms.residual[option_index]);
                option_logits.push(
                    prior_scale * f64::from(option_terms.lexical_prior[option_index])
                        + gate * joint,
                );
                option_index += 1;
            }
            question_logits.push(option_logits);
        }
        Ok(question_logits)
    }

    /// Each row divided by its norm, floored as `functional.normalize` floors it.
    fn unit_rows(&self, rows: &MlxArray) -> Result<MlxArray, DecisionFailure> {
        let runtime = self.runtime;
        let norms = runtime
            .linalg_norm_l2(rows, &[1], true)
            .map_err(head_math)?;
        let floor = runtime
            .array_from_f32(&[NORMALIZE_EPSILON], &[1])
            .map_err(head_math)?;
        runtime
            .divide(rows, &runtime.maximum(&norms, &floor).map_err(head_math)?)
            .map_err(head_math)
    }

    fn row_dot(
        &self,
        left_rows: &MlxArray,
        right_rows: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let runtime = self.runtime;
        runtime
            .sum_axis(
                &runtime.multiply(left_rows, right_rows).map_err(head_math)?,
                1,
                false,
            )
            .map_err(head_math)
    }

    /// `cosine_similarity` along each row.
    fn row_cosine(
        &self,
        left_rows: &MlxArray,
        right_rows: &MlxArray,
    ) -> Result<MlxArray, DecisionFailure> {
        let runtime = self.runtime;
        let dot = self.row_dot(left_rows, right_rows)?;
        let left_norms = runtime
            .linalg_norm_l2(left_rows, &[1], false)
            .map_err(head_math)?;
        let right_norms = runtime
            .linalg_norm_l2(right_rows, &[1], false)
            .map_err(head_math)?;
        let floor = runtime
            .array_from_f32(&[COSINE_EPSILON], &[1])
            .map_err(head_math)?;
        let denominator = runtime
            .maximum(
                &runtime
                    .multiply(&left_norms, &right_norms)
                    .map_err(head_math)?,
                &floor,
            )
            .map_err(head_math)?;
        runtime.divide(&dot, &denominator).map_err(head_math)
    }

    /// `[1, rows, width]` to `[rows, width]`.
    fn rows_of(&self, batched: &MlxArray) -> Result<MlxArray, DecisionFailure> {
        let batched_shape = batched.shape();
        self.runtime
            .reshape(batched, &[batched_shape[1], batched_shape[2]])
            .map_err(head_math)
    }
}

/// The field and normalized option rows, read to the host only when the
/// request asked for head inputs.
fn field_and_option_rows(
    field_per_option: &MlxArray,
    options: &MlxArray,
    head_input_request: HeadInputRequest,
) -> Result<Option<FieldAndOptionRows>, DecisionFailure> {
    match head_input_request {
        HeadInputRequest::Omit => Ok(None),
        HeadInputRequest::Include => Ok(Some(FieldAndOptionRows {
            field_rows: field_per_option.to_vec_f32().map_err(head_math)?,
            option_rows: options.to_vec_f32().map_err(head_math)?,
        })),
    }
}

/// Every option's field vector and normalized option vector, row after row,
/// as the host reads them back.
struct FieldAndOptionRows {
    field_rows: Vec<f32>,
    option_rows: Vec<f32>,
}

/// Cuts the flat per-option readout into one readout per question.
fn question_readouts(
    option_terms: &OptionTerms,
    field_rows: &[f32],
    option_rows: &[f32],
    option_counts: &[usize],
) -> Vec<ClefQuestionReadout> {
    let total_option_count = option_terms.lexical_prior.len().max(1);
    let width = option_rows.len() / total_option_count;
    let mut question_readouts = Vec::with_capacity(option_counts.len());
    let mut option_index = 0;
    for option_count in option_counts {
        let mut readout = ClefQuestionReadout {
            field_vectors: Vec::with_capacity(*option_count),
            option_vectors: Vec::with_capacity(*option_count),
            lexical_priors: Vec::with_capacity(*option_count),
            cosines: Vec::with_capacity(*option_count),
        };
        for _ in 0..*option_count {
            let row_range = option_index * width..(option_index + 1) * width;
            readout
                .field_vectors
                .push(field_rows[row_range.clone()].to_vec());
            readout.option_vectors.push(option_rows[row_range].to_vec());
            readout
                .lexical_priors
                .push(option_terms.lexical_prior[option_index]);
            readout.cosines.push(option_terms.cosine[option_index]);
            option_index += 1;
        }
        question_readouts.push(readout);
    }
    question_readouts
}

/// The pooled vectors and index arrays one request's head pass reads.
struct PooledInputs {
    memory: MlxArray,
    global_vector: MlxArray,
    question_vectors: MlxArray,
    option_contexts: MlxArray,
    lexical_options: MlxArray,
    question_of_option: MlxArray,
    type_ids: MlxArray,
    membership_bias: MlxArray,
    option_counts: Vec<usize>,
}

/// The spans and indices of one request, flattened across its questions.
pub(crate) struct SpanLayout {
    pub(crate) question_spans: Vec<TokenSpan>,
    pub(crate) option_spans: Vec<TokenSpan>,
    /// Each option's run within the concatenated option tokens.
    pub(crate) option_token_spans: Vec<TokenSpan>,
    pub(crate) option_token_count: usize,
    pub(crate) question_of_option: Vec<i32>,
    pub(crate) type_ids: Vec<i32>,
    pub(crate) option_counts: Vec<usize>,
    /// `[questions, options]`: zero where the option is the question's own.
    pub(crate) membership_bias: Vec<f32>,
}

impl SpanLayout {
    pub(crate) fn of(sequence: &ClefSequence) -> Self {
        let mut span_layout = Self {
            question_spans: Vec::new(),
            option_spans: Vec::new(),
            option_token_spans: Vec::new(),
            option_token_count: 0,
            question_of_option: Vec::new(),
            type_ids: Vec::new(),
            option_counts: Vec::new(),
            membership_bias: Vec::new(),
        };
        for (question_index, question_layout) in sequence.questions.iter().enumerate() {
            span_layout
                .question_spans
                .push(question_layout.instructions_span);
            span_layout
                .type_ids
                .push(question_layout.question_text.type_id as i32);
            span_layout
                .option_counts
                .push(question_layout.option_spans.len());
            for option_span in &question_layout.option_spans {
                span_layout.option_spans.push(*option_span);
                let option_token_length = option_span.end - option_span.start;
                span_layout.option_token_spans.push(TokenSpan {
                    start: span_layout.option_token_count,
                    end: span_layout.option_token_count + option_token_length,
                });
                span_layout.option_token_count += option_token_length;
                span_layout.question_of_option.push(question_index as i32);
            }
        }
        let option_count = span_layout.option_spans.len();
        for question_index in 0..sequence.questions.len() {
            for owning_question in &span_layout.question_of_option {
                span_layout
                    .membership_bias
                    .push(if *owning_question == question_index as i32 {
                        0.0
                    } else {
                        OTHER_QUESTION_SCORE_BIAS
                    });
            }
        }
        debug_assert_eq!(
            span_layout.membership_bias.len(),
            sequence.questions.len() * option_count
        );
        span_layout
    }

    /// The option tokens' identifiers, option by option, for the lexical read.
    pub(crate) fn option_token_ids(sequence: &ClefSequence) -> Vec<u32> {
        let mut option_token_ids = Vec::new();
        for question_layout in &sequence.questions {
            for option_span in &question_layout.option_spans {
                option_token_ids.extend(&sequence.token_ids[option_span.start..option_span.end]);
            }
        }
        option_token_ids
    }
}

/// A `[spans, columns]` matrix whose row averages its span's columns.
pub(crate) fn averaging_matrix(spans: &[TokenSpan], column_count: usize) -> Vec<f32> {
    let mut averaging_values = vec![0.0_f32; spans.len() * column_count];
    for (span_index, token_span) in spans.iter().enumerate() {
        let span_length = token_span.end.saturating_sub(token_span.start);
        if span_length == 0 {
            continue;
        }
        let share = 1.0 / span_length as f32;
        let row_start = span_index * column_count;
        averaging_values[row_start + token_span.start..row_start + token_span.end].fill(share);
    }
    averaging_values
}

#[cfg(test)]
#[path = "joint_head_tests.rs"]
mod tests;
