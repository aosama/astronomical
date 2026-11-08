//! Reads the YAML front matter of a Hugging Face model card (`README.md`).
//! Builds that ship no license file declare their license there, so discovery
//! reads it as license evidence; only exact `key: value` lines between the
//! opening and closing `---` count.

pub(super) fn has_front_matter_value(readme: &str, key: &str, expected_value: &str) -> bool {
    let expected_line = format!("{key}: {expected_value}");
    let mut lines = readme.lines().map(str::trim);
    if lines.next() != Some("---") {
        return false;
    }
    for line in lines {
        if line == "---" {
            break;
        }
        if line == expected_line {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::has_front_matter_value;

    const MODEL_CARD: &str = "---\nlicense: apache-2.0\nbase_model: black-forest-labs/FLUX.2-klein-4B\n---\n\nlicense: mit\n";

    #[test]
    fn should_find_an_exact_value_inside_the_front_matter() {
        assert!(has_front_matter_value(MODEL_CARD, "license", "apache-2.0"));
        assert!(has_front_matter_value(
            MODEL_CARD,
            "base_model",
            "black-forest-labs/FLUX.2-klein-4B"
        ));
    }

    #[test]
    fn should_ignore_values_after_the_front_matter_closes() {
        assert!(!has_front_matter_value(MODEL_CARD, "license", "mit"));
    }

    #[test]
    fn should_refuse_a_card_without_front_matter() {
        assert!(!has_front_matter_value(
            "license: apache-2.0\n",
            "license",
            "apache-2.0"
        ));
        assert!(!has_front_matter_value("", "license", "apache-2.0"));
    }
}
