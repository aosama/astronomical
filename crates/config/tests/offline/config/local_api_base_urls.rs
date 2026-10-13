//! The published local API page must name the loopback base URL of every runtime
//! instance.
//!
//! The port is a product value that the page restates for readers, so the page and
//! the instance boundary are checked against each other here: a page that names a
//! port no instance binds sends a reader to a closed socket, and a page that misses
//! an instance hides a channel the app serves.

use std::collections::BTreeSet;
use std::path::PathBuf;

use astronomical_config::AstronomicalRuntimeInstance;

const LOCAL_API_PAGE_RELATIVE_PATH: &str = "../../site/local-api.html";
const LOOPBACK_ORIGIN_PREFIX: &str = "http://127.0.0.1:";

fn local_api_page() -> String {
    let page_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(LOCAL_API_PAGE_RELATIVE_PATH);
    std::fs::read_to_string(&page_path).expect("the local API page should be readable")
}

/// Loopback ports the page publishes as base URLs, so a port merely mentioned for
/// another reason is not mistaken for a published endpoint.
fn published_base_url_ports(page: &str) -> BTreeSet<u16> {
    page.match_indices(LOOPBACK_ORIGIN_PREFIX)
        .filter_map(|(origin_start, origin_prefix)| {
            let after_origin = &page[origin_start + origin_prefix.len()..];
            let port_digit_count = after_origin
                .chars()
                .take_while(char::is_ascii_digit)
                .count();
            if port_digit_count == 0 {
                return None;
            }
            let (port_text, after_port) = after_origin.split_at(port_digit_count);
            after_port
                .starts_with("/v1")
                .then(|| port_text.parse::<u16>().ok())
                .flatten()
        })
        .collect()
}

#[test]
fn should_publish_the_base_url_of_every_runtime_instance() {
    let published_ports = published_base_url_ports(&local_api_page());
    let bound_ports: BTreeSet<u16> = [
        AstronomicalRuntimeInstance::Stable,
        AstronomicalRuntimeInstance::Development,
    ]
    .into_iter()
    .map(|runtime_instance| runtime_instance.loopback_socket_addr().port())
    .collect();

    assert_eq!(
        published_ports, bound_ports,
        "the local API page must publish the base URL of every instance the app binds, and no other port"
    );
}
