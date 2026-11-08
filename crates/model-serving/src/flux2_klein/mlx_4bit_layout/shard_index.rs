//! The `model.safetensors.index.json` mflux writes for each component: which
//! shard holds each stored tensor, and the quantization every packed
//! projection uses.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::affine_quantization_scheme::{AFFINE_QUANTIZATION_BITS, AFFINE_QUANTIZATION_GROUP_SIZE};
use crate::strict_json::DuplicateAwareJsonValue;

use super::super::artifact::Flux2KleinArtifactError;
use super::super::artifact_text_validation::is_safe_weight_shard_path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardIndexDocument {
    metadata: ShardIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

/// mflux records the scheme as strings; any other scheme would need different
/// packed shapes, so it is refused rather than reinterpreted.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardIndexMetadata {
    quantization_level: String,
    quantization_group_size: String,
    mflux_version: String,
}

/// One component's validated shard index.
pub(super) struct Mlx4BitShardIndex {
    /// Stored tensor name to the shard file, relative to the component folder.
    pub(super) weight_map: BTreeMap<String, String>,
    /// Every distinct shard file the index names, sorted.
    pub(super) shard_file_names: Vec<String>,
}

impl Mlx4BitShardIndex {
    pub(super) fn parse(
        index_file_name: &str,
        index_bytes: &[u8],
    ) -> Result<Self, Flux2KleinArtifactError> {
        let malformed_index = || Flux2KleinArtifactError::ArtifactFile {
            file_name: index_file_name.to_owned(),
        };
        let duplicate_aware = serde_json::from_slice::<DuplicateAwareJsonValue>(index_bytes)
            .map_err(|_| malformed_index())?;
        let document = serde_json::from_value::<ShardIndexDocument>(duplicate_aware.0)
            .map_err(|_| malformed_index())?;
        if !is_reviewed_quantization(&document.metadata) {
            return Err(Flux2KleinArtifactError::UnsupportedQuantization {
                file_name: index_file_name.to_owned(),
            });
        }
        let mut shard_file_names = BTreeSet::new();
        for shard_file_name in document.weight_map.values() {
            if !is_safe_weight_shard_path(shard_file_name) {
                return Err(Flux2KleinArtifactError::UnsupportedShardName {
                    shard_file_name: shard_file_name.clone(),
                });
            }
            shard_file_names.insert(shard_file_name.clone());
        }
        if shard_file_names.is_empty() {
            return Err(malformed_index());
        }
        Ok(Self {
            weight_map: document.weight_map,
            shard_file_names: shard_file_names.into_iter().collect(),
        })
    }
}

fn is_reviewed_quantization(metadata: &ShardIndexMetadata) -> bool {
    metadata.quantization_level == AFFINE_QUANTIZATION_BITS.to_string()
        && metadata.quantization_group_size == AFFINE_QUANTIZATION_GROUP_SIZE.to_string()
        && !metadata.mflux_version.is_empty()
}

#[cfg(test)]
mod tests {
    use super::Mlx4BitShardIndex;
    use crate::flux2_klein::artifact::Flux2KleinArtifactError;

    fn index_json(level: &str, group: &str, shard: &str) -> String {
        format!(
            r#"{{"metadata":{{"quantization_level":"{level}","quantization_group_size":"{group}","mflux_version":"0.17.5"}},"weight_map":{{"proj_out.weight":"{shard}","proj_out.scales":"{shard}"}}}}"#
        )
    }

    #[test]
    fn should_parse_a_four_bit_group_64_index() {
        let index =
            Mlx4BitShardIndex::parse("i.json", index_json("4", "64", "0.safetensors").as_bytes())
                .expect("the reviewed scheme should parse");
        assert_eq!(index.shard_file_names, ["0.safetensors"]);
        assert_eq!(index.weight_map.len(), 2);
    }

    #[test]
    fn should_refuse_any_other_quantization_scheme() {
        for (level, group) in [("8", "64"), ("4", "32"), ("3", "64")] {
            let refusal = Mlx4BitShardIndex::parse(
                "i.json",
                index_json(level, group, "0.safetensors").as_bytes(),
            );
            assert!(matches!(
                refusal,
                Err(Flux2KleinArtifactError::UnsupportedQuantization { .. })
            ));
        }
    }

    #[test]
    fn should_refuse_a_shard_path_that_leaves_the_component_folder() {
        let refusal = Mlx4BitShardIndex::parse(
            "i.json",
            index_json("4", "64", "../0.safetensors").as_bytes(),
        );
        assert!(matches!(
            refusal,
            Err(Flux2KleinArtifactError::UnsupportedShardName { .. })
        ));
    }

    #[test]
    fn should_refuse_an_index_with_unknown_fields_or_no_shards() {
        let unknown_field = r#"{"metadata":{"quantization_level":"4","quantization_group_size":"64","mflux_version":"1","extra":1},"weight_map":{"a":"0.safetensors"}}"#;
        let no_shards = r#"{"metadata":{"quantization_level":"4","quantization_group_size":"64","mflux_version":"1"},"weight_map":{}}"#;
        for malformed in [unknown_field, no_shards] {
            assert!(matches!(
                Mlx4BitShardIndex::parse("i.json", malformed.as_bytes()),
                Err(Flux2KleinArtifactError::ArtifactFile { .. })
            ));
        }
    }
}
