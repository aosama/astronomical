//! How the CLI reaches a running Astronomical instance. One module per
//! boundary: `ipc` speaks the framed protocol over the daemon's unix
//! socket, `loopback_http` speaks minimal HTTP over the loopback REST
//! surface.

pub mod ipc;
pub mod loopback_http;
