//! Integration journeys for the `astronomical` CLI, exercised through the
//! public library surface against real stubs: a configurable daemon
//! speaking the actual framed protocol over a unix socket, and a loopback
//! HTTP stand-in. One module per verb, mirroring `src/verbs`; the shared
//! fakes and environment helpers live under `support`.

mod doctor;
mod embed;
mod launch;
mod models;
mod quantize;
mod respond;
mod status;
mod support;
mod validate_config;
