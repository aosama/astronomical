//! Process-wide rustls crypto provider installation for the daemon.
//!
//! The workspace builds reqwest with `rustls-no-provider`, so TLS clients
//! refuse to build until some process-wide provider is installed. This
//! module installs ring — rustls's original provider, chosen over the
//! aws-lc default because a purely local application has no use for a
//! FIPS-lineage cloud-provider crypto stack, and ring's build avoids the
//! multi-second CMake build aws-lc-sys drags in (issue #1690).

/// Installs ring as the process-wide rustls crypto provider.
///
/// Idempotent: `install_default` fails only when a provider is already
/// installed, and every call site in this process installs the same
/// provider, so the first installation wins and later calls are harmless
/// no-ops. Called where TLS clients are built
/// (`ReqwestHubTransport::production`) so every construction path — the
/// daemon binary, unit tests, and any future embedder — is covered by one
/// line.
pub fn install_default_tls_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}
