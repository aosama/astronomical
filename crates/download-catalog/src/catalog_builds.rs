//! Builds of one model: an entry may name the lead entry it is another build
//! of, and say what sets it apart.
//!
//! The Ornith 1.5 35B alone ships as ten downloads. Listed side by side they
//! read as ten different models, so the Models page folds every build under its
//! lead and suggests the build that suits a Mac. Each build keeps its own model
//! ID as its one name; the link adds no second name (see schema 5).

use serde::Deserialize;

use crate::catalog_classification::DownloadCatalogRole;
use crate::download_catalog::{DownloadCatalogEntry, DownloadCatalogError};

/// What a build changes compared with its lead, in the terms the page explains.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadCatalogBuildDifference {
    /// Keeps more of the original weights: more bits, more memory, usually a
    /// small gain in answer quality. The only kind the page suggests as a step up.
    MoreDetail,
    /// Carries a draft model that makes some replies quicker on some Macs; it
    /// answers the same, so it is never suggested as a better model.
    DraftModel,
    /// Lays the weights out for reading from the SSD on Macs with less memory.
    StreamingLayout,
}

impl DownloadCatalogBuildDifference {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        config::exhaustive_table!(self, {
            Self::MoreDetail => "more_detail",
            Self::DraftModel => "draft_model",
            Self::StreamingLayout => "streaming_layout",
        })
    }
}

/// The link from a build to its lead entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DownloadCatalogBuild {
    lead_model_id: String,
    difference: DownloadCatalogBuildDifference,
}

impl DownloadCatalogBuild {
    /// The model ID of the lead entry this is another build of.
    #[must_use]
    pub fn lead_model_id(&self) -> &str {
        &self.lead_model_id
    }

    #[must_use]
    pub const fn difference(&self) -> DownloadCatalogBuildDifference {
        self.difference
    }
}

/// Reads the two authored fields of one entry. They come as a pair: a lead with
/// no difference could not be explained, and a difference with no lead has
/// nothing to differ from.
pub(crate) fn parse_entry_build(
    build_of: Option<&str>,
    build_difference: Option<DownloadCatalogBuildDifference>,
    entry_index: usize,
) -> Result<Option<DownloadCatalogBuild>, DownloadCatalogError> {
    match (build_of, build_difference) {
        (None, None) => Ok(None),
        (Some(lead_model_id), Some(difference)) if !lead_model_id.trim().is_empty() => {
            Ok(Some(DownloadCatalogBuild {
                lead_model_id: lead_model_id.to_owned(),
                difference,
            }))
        }
        _ => Err(DownloadCatalogError::InvalidBuild { entry_index }),
    }
}

/// Checks every build against the whole catalog once all entries are read,
/// because a lead may be authored after its builds.
pub(crate) fn validate_catalog_builds(
    entries: &[DownloadCatalogEntry],
) -> Result<(), DownloadCatalogError> {
    for (entry_index, catalog_entry) in entries.iter().enumerate() {
        let Some(build) = catalog_entry.build() else {
            continue;
        };
        validate_one_build(entries, entry_index, catalog_entry, build)?;
    }
    Ok(())
}

/// A build is folded under its lead, so it must be listed where the lead is: an
/// experimental or retiring entry has its own section, and a Get started pick is
/// shown on its own. A lead that is itself a build would fold twice.
fn validate_one_build(
    entries: &[DownloadCatalogEntry],
    entry_index: usize,
    catalog_entry: &DownloadCatalogEntry,
    build: &DownloadCatalogBuild,
) -> Result<(), DownloadCatalogError> {
    if catalog_entry.get_started() || catalog_entry.experimental() || catalog_entry.retiring() {
        return Err(DownloadCatalogError::InvalidBuild { entry_index });
    }
    let Some((lead_index, lead_entry)) = entries
        .iter()
        .enumerate()
        .find(|(_, candidate)| candidate.model_id() == build.lead_model_id())
    else {
        return Err(DownloadCatalogError::UnknownBuildLead { entry_index });
    };
    if lead_index == entry_index {
        return Err(DownloadCatalogError::InvalidBuild { entry_index });
    }
    let lead_is_listed_apart = lead_entry.build().is_some()
        || lead_entry.experimental()
        || lead_entry.retiring()
        || !roles_share_a_section(catalog_entry.role(), lead_entry.role());
    if lead_is_listed_apart {
        return Err(DownloadCatalogError::InvalidBuildLead { entry_index });
    }
    Ok(())
}

/// Coding models are listed with chat models (each of them also chats), so a
/// chat build may fold under a coding lead; every other role has its own list.
fn roles_share_a_section(build_role: DownloadCatalogRole, lead_role: DownloadCatalogRole) -> bool {
    build_role == lead_role || (is_listed_with_chat(build_role) && is_listed_with_chat(lead_role))
}

const fn is_listed_with_chat(role: DownloadCatalogRole) -> bool {
    matches!(
        role,
        DownloadCatalogRole::Chat | DownloadCatalogRole::Coding
    )
}

#[cfg(test)]
#[path = "catalog_builds_tests.rs"]
mod catalog_builds_tests;
