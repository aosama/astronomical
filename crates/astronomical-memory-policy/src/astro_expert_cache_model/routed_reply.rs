//! A recording of which experts every layer chose for every token of a reply.
//!
//! Routing depends only on the model and the text, not on memory, so one
//! recording can be replayed against any `astro_expert_cache` capacity and any eviction order.
//! Recordings are kept in a compact binary file because a text form of the
//! same reply is three times the size and nobody reads it by eye.

/// The bytes every recording starts with, so a truncated or unrelated file
/// is refused instead of being replayed as nonsense.
const RECORDING_MAGIC: &[u8] = b"ASTROROUTES1";
/// Layer count, experts per token and experts per layer as `u16`, then the
/// token count as `u32`, all little-endian.
const RECORDING_HEADER_FIELD_BYTES: usize = 2 + 2 + 2 + 4;
/// One byte per expert identifier is enough for every layer this recording
/// form supports; larger layers are refused when decoding.
const MAXIMUM_RECORDED_EXPERTS_PER_LAYER: u16 = 256;

/// Why a recording could not be decoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RoutedReplyDecodeError {
    #[error("the file does not start with a routed-reply recording header")]
    NotARecording,
    #[error("the recording header is incomplete")]
    IncompleteHeader,
    #[error("a layer of {experts_per_layer} experts does not fit one byte per expert")]
    TooManyExpertsPerLayer { experts_per_layer: u16 },
    #[error(
        "the recording holds {actual_bytes} route bytes where its header promises {expected_bytes}"
    )]
    WrongRouteByteCount {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    #[error(
        "token {token_index} layer {layer_index} routes to {actual_expert_count} experts where the first layer of the first token routes to {expected_expert_count}"
    )]
    UnevenRoute {
        token_index: usize,
        layer_index: usize,
        expected_expert_count: usize,
        actual_expert_count: usize,
    },
    #[error("token {token_index} layer {layer_index} names expert {expert_id}, beyond the layer")]
    ExpertBeyondLayer {
        token_index: usize,
        layer_index: usize,
        expert_id: u16,
    },
}

/// One reply's routing: for each token, for each layer, the experts chosen.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutedReply {
    layer_count: usize,
    experts_per_token: usize,
    experts_per_layer: u16,
    /// Expert identifiers in token order, then layer order, then the order
    /// the recording listed them: one flat run keeps half a megabyte of
    /// routing in a single allocation.
    routed_expert_ids: Vec<u16>,
}

impl RoutedReply {
    /// Decodes a recording file.
    ///
    /// # Errors
    ///
    /// Returns the reason the bytes are not a complete, consistent recording.
    pub fn from_recording_bytes(recording_bytes: &[u8]) -> Result<Self, RoutedReplyDecodeError> {
        let header_bytes = recording_bytes
            .strip_prefix(RECORDING_MAGIC)
            .ok_or(RoutedReplyDecodeError::NotARecording)?;
        if header_bytes.len() < RECORDING_HEADER_FIELD_BYTES {
            return Err(RoutedReplyDecodeError::IncompleteHeader);
        }
        let (field_bytes, route_bytes) = header_bytes.split_at(RECORDING_HEADER_FIELD_BYTES);
        let layer_count = usize::from(u16::from_le_bytes([field_bytes[0], field_bytes[1]]));
        let experts_per_token = usize::from(u16::from_le_bytes([field_bytes[2], field_bytes[3]]));
        let experts_per_layer = u16::from_le_bytes([field_bytes[4], field_bytes[5]]);
        let token_count = u32::from_le_bytes([
            field_bytes[6],
            field_bytes[7],
            field_bytes[8],
            field_bytes[9],
        ]) as usize;
        if experts_per_layer > MAXIMUM_RECORDED_EXPERTS_PER_LAYER {
            return Err(RoutedReplyDecodeError::TooManyExpertsPerLayer { experts_per_layer });
        }
        let expected_bytes = token_count * layer_count * experts_per_token;
        if route_bytes.len() != expected_bytes {
            return Err(RoutedReplyDecodeError::WrongRouteByteCount {
                expected_bytes,
                actual_bytes: route_bytes.len(),
            });
        }
        let routed_reply = Self {
            layer_count,
            experts_per_token,
            experts_per_layer,
            routed_expert_ids: route_bytes.iter().copied().map(u16::from).collect(),
        };
        routed_reply.require_every_expert_within_its_layer()?;
        Ok(routed_reply)
    }

    /// Builds a recording from routes taken out of an engine: for each token,
    /// for each layer, the experts it routed to. Every layer of every token
    /// must route to the same number of experts.
    ///
    /// # Errors
    ///
    /// Returns why the routes cannot form a recording: layers too large for
    /// one byte per expert, a route of a different width, or an expert
    /// beyond its layer.
    pub fn from_token_routes(
        experts_per_layer: u16,
        token_routes: &[Vec<Vec<u16>>],
    ) -> Result<Self, RoutedReplyDecodeError> {
        if experts_per_layer > MAXIMUM_RECORDED_EXPERTS_PER_LAYER {
            return Err(RoutedReplyDecodeError::TooManyExpertsPerLayer { experts_per_layer });
        }
        let layer_count = token_routes.first().map_or(0, Vec::len);
        let experts_per_token = token_routes
            .first()
            .and_then(|layer_routes| layer_routes.first())
            .map_or(0, Vec::len);
        let mut routed_expert_ids =
            Vec::with_capacity(token_routes.len() * layer_count * experts_per_token);
        for (token_index, layer_routes) in token_routes.iter().enumerate() {
            for layer_index in 0..layer_count {
                let layer_route = layer_routes.get(layer_index).map_or(&[][..], Vec::as_slice);
                if layer_route.len() != experts_per_token {
                    return Err(RoutedReplyDecodeError::UnevenRoute {
                        token_index,
                        layer_index,
                        expected_expert_count: experts_per_token,
                        actual_expert_count: layer_route.len(),
                    });
                }
                routed_expert_ids.extend_from_slice(layer_route);
            }
        }
        let routed_reply = Self {
            layer_count,
            experts_per_token,
            experts_per_layer,
            routed_expert_ids,
        };
        routed_reply.require_every_expert_within_its_layer()?;
        Ok(routed_reply)
    }

    /// The recording file's bytes, which `from_recording_bytes` reads back.
    #[must_use]
    pub fn to_recording_bytes(&self) -> Vec<u8> {
        let mut recording_bytes = Vec::with_capacity(
            RECORDING_MAGIC.len() + RECORDING_HEADER_FIELD_BYTES + self.routed_expert_ids.len(),
        );
        recording_bytes.extend_from_slice(RECORDING_MAGIC);
        recording_bytes.extend_from_slice(&header_field_u16(self.layer_count).to_le_bytes());
        recording_bytes.extend_from_slice(&header_field_u16(self.experts_per_token).to_le_bytes());
        recording_bytes.extend_from_slice(&self.experts_per_layer.to_le_bytes());
        recording_bytes.extend_from_slice(
            &u32::try_from(self.token_count())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for &expert_id in &self.routed_expert_ids {
            // `require_every_expert_within_its_layer` bounds every identifier
            // by a layer of at most 256 experts.
            recording_bytes.push(u8::try_from(expert_id).unwrap_or(u8::MAX));
        }
        recording_bytes
    }

    /// How many experts every layer of every token routes to.
    #[must_use]
    pub const fn experts_per_token(&self) -> usize {
        self.experts_per_token
    }

    fn require_every_expert_within_its_layer(&self) -> Result<(), RoutedReplyDecodeError> {
        for token_index in 0..self.token_count() {
            for layer_index in 0..self.layer_count {
                for &expert_id in self.layer_route(token_index, layer_index) {
                    if expert_id >= self.experts_per_layer {
                        return Err(RoutedReplyDecodeError::ExpertBeyondLayer {
                            token_index,
                            layer_index,
                            expert_id,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    #[must_use]
    pub const fn experts_per_layer(&self) -> u16 {
        self.experts_per_layer
    }

    #[must_use]
    pub fn token_count(&self) -> usize {
        let route_length = self.layer_count * self.experts_per_token;
        self.routed_expert_ids
            .len()
            .checked_div(route_length)
            .unwrap_or(0)
    }

    /// The experts one layer chose for one token.
    ///
    /// # Panics
    ///
    /// Panics when the token or the layer is beyond the recording, which is
    /// a caller bug: both counts are available to check against.
    #[must_use]
    pub fn layer_route(&self, token_index: usize, layer_index: usize) -> &[u16] {
        assert!(layer_index < self.layer_count, "layer beyond the recording");
        let route_start = (token_index * self.layer_count + layer_index) * self.experts_per_token;
        &self.routed_expert_ids[route_start..route_start + self.experts_per_token]
    }
}

fn header_field_u16(header_value: usize) -> u16 {
    u16::try_from(header_value).unwrap_or(u16::MAX)
}
