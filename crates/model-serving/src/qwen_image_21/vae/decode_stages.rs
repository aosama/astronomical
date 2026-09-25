//! The staged decode state machine the decoder walks, one graph stage per iteration.
//!
//! A complete decode builds every stage's graph in isolation: `conv_in`, the three middle-block
//! stages, each up block, then the output head. Splitting the walk into explicit stages is what
//! lets the decoder evaluate and release each stage before the next one is built, which is the
//! difference between decoding 1024×1024 inside the memory ceiling and being rejected by it.

use astronomical_runtime_integration::MlxArray;

/// The stage a decode is about to build.
#[derive(Clone, Copy, Debug)]
pub(super) enum DecodeStage {
    /// `decoder.conv_in`: latents to the middle-block width.
    ConvolutionInput,
    /// The middle block's first residual.
    MiddleResnetBeforeAttention,
    /// The middle block's spatial self-attention.
    MiddleAttention,
    /// The middle block's second residual.
    MiddleResnetAfterAttention,
    /// One residual up block, by index into `QwenImage21VaeDecoder::up_blocks`.
    UpBlock(usize),
    /// `norm_out → SiLU → conv_out` into RGBA pixels.
    OutputHead,
}

impl DecodeStage {
    /// The stage that consumes this stage's output, or `None` when this stage produces pixels.
    pub(super) fn next(self, up_block_count: usize) -> Option<Self> {
        match self {
            Self::ConvolutionInput => Some(Self::MiddleResnetBeforeAttention),
            Self::MiddleResnetBeforeAttention => Some(Self::MiddleAttention),
            Self::MiddleAttention => Some(Self::MiddleResnetAfterAttention),
            Self::MiddleResnetAfterAttention => Some(Self::UpBlock(0)),
            Self::UpBlock(block_index) if block_index + 1 == up_block_count => {
                Some(Self::OutputHead)
            }
            Self::UpBlock(block_index) => Some(Self::UpBlock(block_index + 1)),
            Self::OutputHead => None,
        }
    }
}

/// A decode in progress: the stage that just produced `hidden_states` and the stage to build next.
#[derive(Debug)]
pub(super) struct DecodeState {
    pub(super) hidden_states: MlxArray,
    pub(super) next_stage: DecodeStage,
}

/// The result of building and evaluating one decode stage.
#[derive(Debug)]
pub(super) enum DecodeAdvance {
    /// The stage produced activations for the next stage to consume.
    Decoding(DecodeState),
    /// The stage was the output head; these are the RGBA pixels.
    PixelsReady(MlxArray),
}
