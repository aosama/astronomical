//! A recording of which experts every layer chose for every token of a reply.
//!
//! Routing depends only on the model and the text, not on memory, so one
//! recording can be replayed against any `astro_expert_cache` capacity and any eviction order.
//! Recordings are kept in a compact binary file because a text form of the
//! same reply is three times the size and nobody reads it by eye.

/// The bytes every recording starts with, so a truncated or unrelated file
/// is refused instead of being replayed as nonsense.
const RECORDING_MAGIC: &[u8] = b"ASTROROUTES1";
/// The start of a recording whose layers hold more than 256 experts, where
/// an identifier takes two little-endian bytes. A model of 512 experts a
/// layer (Qwen3.8-Flash-Next) records this way; smaller layers keep the
/// one-byte form, so recordings made before this form still read.
const WIDE_RECORDING_MAGIC: &[u8] = b"ASTROROUTES2";
/// Layer count, experts per token and experts per layer as `u16`, then the
/// token count as `u32`, all little-endian.
const RECORDING_HEADER_FIELD_BYTES: usize = 2 + 2 + 2 + 4;
/// The most experts a layer may hold for one byte per identifier; the
/// one-byte form refuses larger layers when decoding.
const MAXIMUM_ONE_BYTE_EXPERTS_PER_LAYER: u16 = 256;

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

/// How many bytes a recording spends on each expert identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpertIdentifierWidth {
    OneByte,
    TwoBytes,
}

impl ExpertIdentifierWidth {
    /// The narrowest width that holds every identifier of such a layer.
    const fn for_layer_of(experts_per_layer: u16) -> Self {
        if experts_per_layer > MAXIMUM_ONE_BYTE_EXPERTS_PER_LAYER {
            Self::TwoBytes
        } else {
            Self::OneByte
        }
    }

    /// A recording's bytes after its magic, and the width the magic names;
    /// `None` when the bytes start with neither magic.
    fn of_recording(recording_bytes: &[u8]) -> Option<(&[u8], Self)> {
        if let Some(header_bytes) = recording_bytes.strip_prefix(WIDE_RECORDING_MAGIC) {
            return Some((header_bytes, Self::TwoBytes));
        }
        let header_bytes = recording_bytes.strip_prefix(RECORDING_MAGIC)?;
        Some((header_bytes, Self::OneByte))
    }

    const fn recording_magic(self) -> &'static [u8] {
        match self {
            Self::OneByte => RECORDING_MAGIC,
            Self::TwoBytes => WIDE_RECORDING_MAGIC,
        }
    }

    const fn byte_count(self) -> usize {
        match self {
            Self::OneByte => 1,
            Self::TwoBytes => 2,
        }
    }

    const fn holds_layer_of(self, experts_per_layer: u16) -> bool {
        matches!(self, Self::TwoBytes) || experts_per_layer <= MAXIMUM_ONE_BYTE_EXPERTS_PER_LAYER
    }

    /// The identifiers of a recording's route bytes, little-endian.
    fn decoded_expert_ids(self, route_bytes: &[u8]) -> Vec<u16> {
        let mut expert_ids = Vec::with_capacity(route_bytes.len() / self.byte_count());
        for expert_id_bytes in route_bytes.chunks_exact(self.byte_count()) {
            let high_byte = expert_id_bytes.get(1).copied().unwrap_or(0);
            expert_ids.push(u16::from_le_bytes([expert_id_bytes[0], high_byte]));
        }
        expert_ids
    }
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
        let (header_bytes, identifier_width) = ExpertIdentifierWidth::of_recording(recording_bytes)
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
        if !identifier_width.holds_layer_of(experts_per_layer) {
            return Err(RoutedReplyDecodeError::TooManyExpertsPerLayer { experts_per_layer });
        }
        let expected_bytes =
            token_count * layer_count * experts_per_token * identifier_width.byte_count();
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
            routed_expert_ids: identifier_width.decoded_expert_ids(route_bytes),
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
        let identifier_width = ExpertIdentifierWidth::for_layer_of(self.experts_per_layer);
        let recording_magic = identifier_width.recording_magic();
        let mut recording_bytes = Vec::with_capacity(
            recording_magic.len()
                + RECORDING_HEADER_FIELD_BYTES
                + self.routed_expert_ids.len() * identifier_width.byte_count(),
        );
        recording_bytes.extend_from_slice(recording_magic);
        recording_bytes.extend_from_slice(&header_field_u16(self.layer_count).to_le_bytes());
        recording_bytes.extend_from_slice(&header_field_u16(self.experts_per_token).to_le_bytes());
        recording_bytes.extend_from_slice(&self.experts_per_layer.to_le_bytes());
        recording_bytes.extend_from_slice(
            &u32::try_from(self.token_count())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for &expert_id in &self.routed_expert_ids {
            // `require_every_expert_within_its_layer` bounds every
            // identifier by its layer, which the width was chosen for.
            recording_bytes
                .extend_from_slice(&expert_id.to_le_bytes()[..identifier_width.byte_count()]);
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
