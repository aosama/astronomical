//! The Anthropic Models API fields the shared `/v1/models` listing carries
//! beside its OpenAI fields, so Anthropic-protocol clients and
//! OpenAI-protocol clients read the same response. The two shapes do not
//! collide: Anthropic names the kind `type` where OpenAI uses `object`, and
//! dates a model with an RFC 3339 `created_at` where OpenAI uses Unix
//! `created`.

use serde::Serialize;

const SECONDS_PER_DAY: u64 = 86_400;

/// Per-model Anthropic fields: `type`, `display_name`, and `created_at`.
/// `created_at` is the advertisement time the OpenAI `created` field already
/// carries, not a release date: a local model has no release record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AnthropicModelIdentity {
    #[serde(rename = "type")]
    model_type: &'static str,
    /// Local models have no separate marketing name, so the display name is
    /// the model identifier a client sends back.
    display_name: String,
    created_at: String,
}

impl AnthropicModelIdentity {
    #[must_use]
    pub fn for_model(model_id: &str, created_unix_seconds: u64) -> Self {
        Self {
            model_type: "model",
            display_name: model_id.to_owned(),
            created_at: rfc3339_utc_from_unix_seconds(created_unix_seconds),
        }
    }
}

/// List-level Anthropic pagination fields. The local listing always fits one
/// page, so `has_more` is false and the cursors name the first and last
/// model, or are null for an empty list.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AnthropicModelListPage {
    has_more: bool,
    first_id: Option<String>,
    last_id: Option<String>,
}

impl AnthropicModelListPage {
    #[must_use]
    pub fn for_model_ids(first_model_id: Option<&str>, last_model_id: Option<&str>) -> Self {
        Self {
            has_more: false,
            first_id: first_model_id.map(str::to_owned),
            last_id: last_model_id.map(str::to_owned),
        }
    }
}

/// Formats Unix seconds as `YYYY-MM-DDTHH:MM:SSZ` in UTC.
pub(crate) fn rfc3339_utc_from_unix_seconds(unix_seconds: u64) -> String {
    let (year, month, day) = civil_date_from_unix_days(unix_seconds / SECONDS_PER_DAY);
    let seconds_of_day = unix_seconds % SECONDS_PER_DAY;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds_of_day / 3_600,
        (seconds_of_day % 3_600) / 60,
        seconds_of_day % 60
    )
}

/// Converts days since 1970-01-01 to a proleptic Gregorian (year, month,
/// day), using Howard Hinnant's `civil_from_days` algorithm: it shifts the
/// year to start in March so the leap day falls at the end, then splits
/// days into 400-year eras of exactly 146,097 days, with no lookup tables and
/// no date library. The arithmetic is exact for every day count a `u64` of
/// seconds divides into, which leaves the era offset far from overflow; RFC
/// 3339 needs a four-digit year, so the formatted timestamp is valid through
/// 9999-12-31, far beyond any advertisement time a local clock produces.
fn civil_date_from_unix_days(unix_days: u64) -> (u64, u64, u64) {
    let days_since_era_zero = unix_days + 719_468;
    let era = days_since_era_zero / 146_097;
    let day_of_era = days_since_era_zero - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    };
    let march_based_year = year_of_era + era * 400;
    let year = if month <= 2 {
        march_based_year + 1
    } else {
        march_based_year
    };
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::{AnthropicModelIdentity, AnthropicModelListPage, rfc3339_utc_from_unix_seconds};

    #[test]
    fn should_format_the_unix_epoch() {
        assert_eq!(rfc3339_utc_from_unix_seconds(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn should_format_both_ends_of_a_leap_day() {
        assert_eq!(
            rfc3339_utc_from_unix_seconds(951_782_400),
            "2000-02-29T00:00:00Z"
        );
        assert_eq!(
            rfc3339_utc_from_unix_seconds(951_868_799),
            "2000-02-29T23:59:59Z"
        );
    }

    #[test]
    fn should_roll_over_the_year_boundary() {
        assert_eq!(
            rfc3339_utc_from_unix_seconds(1_798_761_599),
            "2026-12-31T23:59:59Z"
        );
        assert_eq!(
            rfc3339_utc_from_unix_seconds(1_798_761_600),
            "2027-01-01T00:00:00Z"
        );
    }

    #[test]
    fn should_format_the_last_four_digit_year_second() {
        assert_eq!(
            rfc3339_utc_from_unix_seconds(253_402_300_799),
            "9999-12-31T23:59:59Z"
        );
    }

    #[test]
    fn should_use_the_model_identifier_as_its_display_name() {
        let model_identity = AnthropicModelIdentity::for_model("local-model", 0);

        let identity_document =
            serde_json::to_value(model_identity).expect("the identity should serialize");

        assert_eq!(identity_document["type"], "model");
        assert_eq!(identity_document["display_name"], "local-model");
        assert_eq!(identity_document["created_at"], "1970-01-01T00:00:00Z");
    }

    #[test]
    fn should_leave_cursors_null_for_an_empty_listing() {
        let page_document = serde_json::to_value(AnthropicModelListPage::for_model_ids(None, None))
            .expect("the page should serialize");

        assert_eq!(page_document["has_more"], false);
        assert!(page_document["first_id"].is_null());
        assert!(page_document["last_id"].is_null());
    }
}
