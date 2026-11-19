//! How much memory macOS has wired across the whole system, and how much
//! its kernel will keep wired.
//!
//! Graphics-processor work slows once the system's wired total passes the
//! kernel's wire limit (see `system_wired_ceiling` in the memory policy),
//! and that total includes what the kernel and other applications wire.
//! Only the kernel knows either figure, so both are read here.

// The workspace confines unsafe code to a few documented modules. This one
// makes two synchronous kernel calls, each into a uniquely writable local,
// and nothing unsafe leaves the file.
#![allow(unsafe_code)]

use std::sync::OnceLock;

/// One reading of the system's wired memory beside the kernel's wire limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MacosWiredMemoryReading {
    pub system_wired_bytes: u64,
    pub system_wire_limit_bytes: u64,
}

/// Reads the system's wired memory now. `None` when the kernel refuses
/// either figure; the caller then leaves its plan as it is.
pub fn read_macos_wired_memory() -> Option<MacosWiredMemoryReading> {
    Some(MacosWiredMemoryReading {
        system_wired_bytes: system_wired_bytes()?,
        system_wire_limit_bytes: system_wire_limit_bytes()?,
    })
}

fn system_wired_bytes() -> Option<u64> {
    let mut virtual_memory_statistics = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut statistics_word_count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: the buffer is a zeroed, uniquely owned `vm_statistics64` and
    // the count states its size in the words the call expects; the kernel
    // writes at most that many.
    #[allow(deprecated)]
    let kernel_return = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            virtual_memory_statistics.as_mut_ptr().cast(),
            &raw mut statistics_word_count,
        )
    };
    if kernel_return != libc::KERN_SUCCESS {
        return None;
    }
    // SAFETY: the call succeeded, so the kernel initialized the structure.
    let virtual_memory_statistics = unsafe { virtual_memory_statistics.assume_init() };
    // SAFETY: `sysconf` only reads a constant of the running system.
    let page_bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_bytes = u64::try_from(page_bytes).ok()?;
    Some(u64::from(virtual_memory_statistics.wire_count).saturating_mul(page_bytes))
}

/// The kernel sets its wire limit from the Mac's memory size at startup, so
/// it is read once.
fn system_wire_limit_bytes() -> Option<u64> {
    static SYSTEM_WIRE_LIMIT_BYTES: OnceLock<Option<u64>> = OnceLock::new();
    *SYSTEM_WIRE_LIMIT_BYTES.get_or_init(read_system_wire_limit_bytes)
}

fn read_system_wire_limit_bytes() -> Option<u64> {
    let mut system_wire_limit_bytes: u64 = 0;
    let mut value_byte_count = std::mem::size_of::<u64>();
    // SAFETY: the name is a NUL-terminated literal, the value buffer is a
    // uniquely owned `u64` and the length states its size.
    let status = unsafe {
        libc::sysctlbyname(
            c"vm.global_user_wire_limit".as_ptr(),
            (&raw mut system_wire_limit_bytes).cast(),
            &raw mut value_byte_count,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && system_wire_limit_bytes > 0).then_some(system_wire_limit_bytes)
}

#[cfg(test)]
mod tests {
    use super::read_macos_wired_memory;

    #[test]
    fn should_read_a_positive_wired_total_and_wire_limit() {
        let reading = read_macos_wired_memory().expect("macOS should report its wired memory");

        assert!(reading.system_wired_bytes > 0);
        assert!(reading.system_wire_limit_bytes > 0);
    }

    #[test]
    fn should_report_the_same_wire_limit_on_every_reading() {
        let first_reading = read_macos_wired_memory().expect("macOS should report its memory");
        let second_reading = read_macos_wired_memory().expect("macOS should report its memory");

        assert_eq!(
            first_reading.system_wire_limit_bytes,
            second_reading.system_wire_limit_bytes
        );
    }
}
