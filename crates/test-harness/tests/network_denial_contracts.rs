// Contract coverage for harness-enforced network denial: the denied macros
// must make loopback socket creation fail loudly on macOS, stay idempotent
// across tests in one process, and defer to platform sandboxing elsewhere.

use astronomical_test_harness::network_denial::ensure_network_denied;
use astronomical_test_harness::offline_denied_test;

#[cfg(target_os = "macos")]
offline_denied_test!(
    denied_macro_blocks_binding_an_ephemeral_loopback_listener,
    {
        let bind_attempt = std::net::TcpListener::bind("127.0.0.1:0");
        assert!(
            bind_attempt.is_err(),
            "network denial must make loopback bind fail, got: {:?}",
            bind_attempt.map(|listener| listener.local_addr())
        );
    }
);

#[cfg(target_os = "macos")]
offline_denied_test!(denied_macro_blocks_an_outbound_loopback_connection, {
    let connect_attempt = std::net::TcpStream::connect("127.0.0.1:9");
    assert!(
        connect_attempt.is_err(),
        "network denial must make outbound connects fail, got: {:?}",
        connect_attempt.err()
    );
});

offline_denied_test!(denial_application_is_idempotent_within_one_process, {
    assert!(
        ensure_network_denied().is_ok(),
        "a second denial application in the same process must be a no-op success"
    );
});

#[cfg(not(target_os = "macos"))]
#[test]
fn network_denial_defers_to_platform_sandboxing_off_macos() {
    assert!(
        ensure_network_denied().is_ok(),
        "off macOS the harness must defer to platform sandboxing (Linux CI uses unshare -rn)"
    );
}
