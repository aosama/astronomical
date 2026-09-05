//! Prefix checker for concatenated BPE pieces treated as JSON UTF-8.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum JsonPrefixStatus {
    Invalid,
    Incomplete,
    Complete,
}

pub(super) fn status(text: &str) -> JsonPrefixStatus {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return JsonPrefixStatus::Incomplete;
    }
    match parse_value(bytes, 0) {
        Parse::Complete(end) => {
            let rest = skip_ws(bytes, end);
            if rest == bytes.len() {
                JsonPrefixStatus::Complete
            } else {
                JsonPrefixStatus::Invalid
            }
        }
        Parse::Incomplete => JsonPrefixStatus::Incomplete,
        Parse::Invalid => JsonPrefixStatus::Invalid,
    }
}

enum Parse {
    Complete(usize),
    Incomplete,
    Invalid,
}

fn parse_value(bytes: &[u8], start: usize) -> Parse {
    let index = skip_ws(bytes, start);
    if index >= bytes.len() {
        return Parse::Incomplete;
    }
    match bytes[index] {
        b'{' => parse_object(bytes, index + 1),
        b'[' => parse_array(bytes, index + 1),
        b'"' => parse_string(bytes, index + 1),
        b't' => parse_literal(bytes, index, b"true"),
        b'f' => parse_literal(bytes, index, b"false"),
        b'n' => parse_literal(bytes, index, b"null"),
        b'-' | b'0'..=b'9' => parse_number(bytes, index),
        _ => Parse::Invalid,
    }
}

fn parse_object(bytes: &[u8], mut index: usize) -> Parse {
    index = skip_ws(bytes, index);
    if index >= bytes.len() {
        return Parse::Incomplete;
    }
    if bytes[index] == b'}' {
        return Parse::Complete(index + 1);
    }
    loop {
        let key_start = skip_ws(bytes, index);
        if key_start >= bytes.len() {
            return Parse::Incomplete;
        }
        if bytes[key_start] != b'"' {
            return Parse::Invalid;
        }
        match parse_string(bytes, key_start + 1) {
            Parse::Complete(after_key) => {
                let colon = skip_ws(bytes, after_key);
                if colon >= bytes.len() {
                    return Parse::Incomplete;
                }
                if bytes[colon] != b':' {
                    return Parse::Invalid;
                }
                match parse_value(bytes, colon + 1) {
                    Parse::Complete(after_value) => {
                        index = skip_ws(bytes, after_value);
                        if index >= bytes.len() {
                            return Parse::Incomplete;
                        }
                        match bytes[index] {
                            b'}' => return Parse::Complete(index + 1),
                            b',' => {
                                index += 1;
                                continue;
                            }
                            _ => return Parse::Invalid,
                        }
                    }
                    other => return other,
                }
            }
            other => return other,
        }
    }
}

fn parse_array(bytes: &[u8], mut index: usize) -> Parse {
    index = skip_ws(bytes, index);
    if index >= bytes.len() {
        return Parse::Incomplete;
    }
    if bytes[index] == b']' {
        return Parse::Complete(index + 1);
    }
    loop {
        match parse_value(bytes, index) {
            Parse::Complete(after_value) => {
                index = skip_ws(bytes, after_value);
                if index >= bytes.len() {
                    return Parse::Incomplete;
                }
                match bytes[index] {
                    b']' => return Parse::Complete(index + 1),
                    b',' => {
                        index += 1;
                        continue;
                    }
                    _ => return Parse::Invalid,
                }
            }
            other => return other,
        }
    }
}

fn parse_string(bytes: &[u8], mut index: usize) -> Parse {
    while index < bytes.len() {
        match bytes[index] {
            b'"' => return Parse::Complete(index + 1),
            b'\\' => {
                index += 1;
                if index >= bytes.len() {
                    return Parse::Incomplete;
                }
                index += 1;
            }
            b if b < 0x20 => return Parse::Invalid,
            _ => index += 1,
        }
    }
    Parse::Incomplete
}

fn parse_literal(bytes: &[u8], start: usize, literal: &[u8]) -> Parse {
    let available = bytes.len().saturating_sub(start);
    if available < literal.len() {
        if literal.starts_with(&bytes[start..]) {
            Parse::Incomplete
        } else {
            Parse::Invalid
        }
    } else if bytes[start..start + literal.len()] == *literal {
        Parse::Complete(start + literal.len())
    } else {
        Parse::Invalid
    }
}

fn parse_number(bytes: &[u8], start: usize) -> Parse {
    let mut index = start;
    if bytes.get(index) == Some(&b'-') {
        index += 1;
        if index >= bytes.len() {
            return Parse::Incomplete;
        }
    }
    if index >= bytes.len() {
        return Parse::Incomplete;
    }
    if bytes[index] == b'0' {
        index += 1;
    } else if bytes[index].is_ascii_digit() {
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
    } else {
        return Parse::Invalid;
    }
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        if index >= bytes.len() {
            return Parse::Incomplete;
        }
        if !bytes[index].is_ascii_digit() {
            return Parse::Invalid;
        }
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
    }
    if matches!(bytes.get(index), Some(&b'e' | &b'E')) {
        index += 1;
        if matches!(bytes.get(index), Some(&b'+' | &b'-')) {
            index += 1;
        }
        if index >= bytes.len() {
            return Parse::Incomplete;
        }
        if !bytes[index].is_ascii_digit() {
            return Parse::Invalid;
        }
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
    }
    Parse::Complete(index)
}

fn skip_ws(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && matches!(bytes[index], b' ' | b'\n' | b'\r' | b'\t') {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::{JsonPrefixStatus, status};

    #[test]
    fn should_treat_empty_text_as_an_incomplete_json_prefix() {
        assert_eq!(status(""), JsonPrefixStatus::Incomplete);
    }

    #[test]
    fn should_accept_a_complete_romeo_object() {
        assert_eq!(
            status(r#"{"speaker":"Juliet","play":"Romeo and Juliet"}"#),
            JsonPrefixStatus::Complete
        );
    }

    #[test]
    fn should_keep_an_open_object_incomplete() {
        assert_eq!(
            status(r#"{"speaker":"Juliet""#),
            JsonPrefixStatus::Incomplete
        );
    }

    #[test]
    fn should_reject_prose_that_is_not_json() {
        assert_eq!(
            status("Two households, both alike in dignity."),
            JsonPrefixStatus::Invalid
        );
    }
}
