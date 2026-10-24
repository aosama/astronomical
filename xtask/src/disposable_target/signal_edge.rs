//! The self-pipe signal plumbing for the disposable-target lifecycle: a
//! tiny unsafe edge that installs HUP/INT/TERM handlers writing one byte
//! into a non-blocking pipe, plus the process-group liveness and signaling
//! helpers the supervisor polls with.

#![allow(unsafe_code)]
// The workspace denies unsafe code everywhere else; this file is the
// sanctioned edge that replaced the shell runner's signal traps and
// process-group kills.

use std::sync::atomic::{AtomicI32, Ordering};

static SIGNAL_PIPE_WRITE_FD: AtomicI32 = AtomicI32::new(-1);

extern "C" fn record_signal(signal_number: i32) {
    // Async-signal-safe: one byte written into the self-pipe; all real
    // handling happens on the supervising thread after poll(2).
    let pipe_write_fd = SIGNAL_PIPE_WRITE_FD.load(Ordering::Relaxed);
    if pipe_write_fd >= 0 {
        let signal_byte = signal_number as u8;
        unsafe {
            let _ = libc::write(
                pipe_write_fd,
                std::ptr::from_ref(&signal_byte).cast::<libc::c_void>(),
                1,
            );
        }
    }
}

extern "C" fn handle_hangup(signal_number: i32) {
    record_signal(signal_number);
}

extern "C" fn handle_interrupt(signal_number: i32) {
    record_signal(signal_number);
}

extern "C" fn handle_terminate(signal_number: i32) {
    record_signal(signal_number);
}

pub(crate) struct SignalPipe {
    pub(crate) read_fd: i32,
}

impl SignalPipe {
    /// Installs HUP/INT/TERM handlers that write one byte into a
    /// self-pipe; the supervisor polls that pipe instead of running any
    /// logic inside the handler itself.
    pub(crate) fn install() -> Option<SignalPipe> {
        let mut pipe_fds = [0 as libc::c_int; 2];
        if unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } != 0 {
            return None;
        }
        for pipe_fd in pipe_fds {
            unsafe {
                let descriptor_flags = libc::fcntl(pipe_fd, libc::F_GETFD);
                libc::fcntl(pipe_fd, libc::F_SETFD, descriptor_flags | libc::FD_CLOEXEC);
                // Non-blocking so a speculative drain after a completed
                // wait returns immediately instead of blocking on an
                // empty pipe.
                let status_flags = libc::fcntl(pipe_fd, libc::F_GETFL);
                libc::fcntl(pipe_fd, libc::F_SETFL, status_flags | libc::O_NONBLOCK);
            }
        }
        SIGNAL_PIPE_WRITE_FD.store(pipe_fds[1], Ordering::SeqCst);
        for (signal_number, handler) in [
            (
                libc::SIGHUP,
                handle_hangup as *const () as libc::sighandler_t,
            ),
            (
                libc::SIGINT,
                handle_interrupt as *const () as libc::sighandler_t,
            ),
            (
                libc::SIGTERM,
                handle_terminate as *const () as libc::sighandler_t,
            ),
        ] {
            let mut handler_action: libc::sigaction = unsafe { std::mem::zeroed() };
            handler_action.sa_sigaction = handler as usize;
            handler_action.sa_flags = libc::SA_RESTART;
            unsafe {
                libc::sigemptyset(&mut handler_action.sa_mask);
                libc::sigaction(signal_number, &handler_action, std::ptr::null_mut());
            }
        }
        Some(SignalPipe {
            read_fd: pipe_fds[0],
        })
    }

    pub(crate) fn received_signal(&self) -> Option<i32> {
        let mut signal_byte = [0_u8; 1];
        let bytes_read = unsafe {
            libc::read(
                self.read_fd,
                signal_byte.as_mut_ptr().cast::<libc::c_void>(),
                1,
            )
        };
        (bytes_read == 1).then_some(i32::from(signal_byte[0]))
    }
}

pub(crate) fn signal_name(signal_number: i32) -> &'static str {
    match signal_number {
        libc::SIGHUP => "HUP",
        libc::SIGINT => "INT",
        _ => "TERM",
    }
}

pub(crate) fn signal_exit_status(signal_number: i32) -> i32 {
    match signal_number {
        libc::SIGHUP => 129,
        libc::SIGINT => 130,
        _ => 143,
    }
}

pub(crate) fn process_group_is_alive(process_group_id: i32) -> bool {
    unsafe { libc::kill(-process_group_id, 0) == 0 }
}

pub(crate) fn signal_process_group(process_group_id: i32, signal_number: i32) {
    unsafe {
        libc::kill(-process_group_id, signal_number);
    }
}
