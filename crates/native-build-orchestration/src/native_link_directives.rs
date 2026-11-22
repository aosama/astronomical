//! The Rust link line for each platform, composed as pure functions.
//!
//! [`crate::native_linking`] prints these lines from a build script; keeping
//! their composition free of the filesystem and of `xcrun` lets each
//! platform's line be tested on any machine without a native build. The
//! macOS line is the one every Mac build links with today, and a contract
//! test pins it.

use std::path::{Path, PathBuf};

/// The Apple frameworks the pinned MLX Metal backend links against. MLX
/// pulls each of these directly; omitting one fails only at final binary
/// link time, far from its cause, so the set stays explicit here.
const MACOS_FRAMEWORK_NAMES: [&str; 6] = [
    "Metal",
    "Foundation",
    "IOKit",
    "CoreFoundation",
    "QuartzCore",
    "Accelerate",
];

/// Library directories the dynamic loader searches by default. A run path
/// pointing at one of them changes nothing, so none is emitted.
const SYSTEM_LIBRARY_DIRECTORIES: [&str; 5] =
    ["/lib", "/lib64", "/usr/lib", "/usr/lib64", "/usr/local/lib"];

/// The cargo directives and warnings for one link line.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct NativeLinkDirectives {
    /// Complete `cargo:` lines, in link order.
    pub cargo_directive_lines: Vec<String>,
    /// Entries that could not be linked faithfully, reported as cargo
    /// warnings rather than silently linked wrong.
    pub warning_messages: Vec<String>,
    /// The library files the line links by path, which must still exist.
    pub linked_library_files: Vec<PathBuf>,
}

/// The macOS line: the two static archives, `libc++`, the Metal backend's
/// frameworks, then Clang's compiler-rt archive, which MLX's C++ sources
/// reference (for example memory intrinsics).
pub fn macos_link_directive_lines(
    native_library_directory: &Path,
    clang_runtime_directory: &Path,
) -> Vec<String> {
    let mut cargo_directive_lines = vec![
        format!(
            "cargo:rustc-link-search=native={}",
            native_library_directory.display()
        ),
        "cargo:rustc-link-lib=static=mlxc".to_owned(),
        "cargo:rustc-link-lib=static=mlx".to_owned(),
        "cargo:rustc-link-lib=dylib=c++".to_owned(),
    ];
    for framework_name in MACOS_FRAMEWORK_NAMES {
        cargo_directive_lines.push(format!("cargo:rustc-link-lib=framework={framework_name}"));
    }
    cargo_directive_lines.push(format!(
        "cargo:rustc-link-search=native={}",
        clang_runtime_directory.display()
    ));
    cargo_directive_lines.push("cargo:rustc-link-lib=static=clang_rt.osx".to_owned());
    cargo_directive_lines
}

/// The Linux line: the two static archives, then the libraries MLX's
/// configure chose (read from the build's link library list), then the GNU
/// C++ runtime, which must come last because the archives reference it.
pub fn linux_link_directives(
    native_library_directory: &Path,
    mlx_link_library_list: &str,
) -> NativeLinkDirectives {
    let mut link_directives = NativeLinkDirectives {
        cargo_directive_lines: vec![
            format!(
                "cargo:rustc-link-search=native={}",
                native_library_directory.display()
            ),
            "cargo:rustc-link-lib=static=mlxc".to_owned(),
            "cargo:rustc-link-lib=static=mlx".to_owned(),
        ],
        warning_messages: Vec::new(),
        linked_library_files: Vec::new(),
    };
    for link_library_entry in mlx_link_library_list.lines() {
        push_link_library_entry(&mut link_directives, link_library_entry);
    }
    push_unique_line(
        &mut link_directives.cargo_directive_lines,
        "cargo:rustc-link-lib=dylib=stdc++".to_owned(),
    );
    link_directives
}

/// Adds one recorded `INTERFACE_LINK_LIBRARIES` entry to the line.
fn push_link_library_entry(link_directives: &mut NativeLinkDirectives, link_library_entry: &str) {
    let Some(linkable_entry) = linkable_entry_text(link_library_entry) else {
        return;
    };
    if let Some(library_name) = linkable_entry.strip_prefix("-l") {
        if library_name.is_empty() || library_name.starts_with(':') {
            link_directives.warning_messages.push(format!(
                "MLX links {linkable_entry:?}, which the Rust linker cannot name; it was not linked"
            ));
            return;
        }
        push_dynamic_library(link_directives, library_name);
    } else if linkable_entry.starts_with('-') {
        link_directives.warning_messages.push(format!(
            "MLX links with the linker flag {linkable_entry:?}, which a build script cannot pass to dependents; it was not linked"
        ));
    } else if linkable_entry.contains("::") {
        link_directives.warning_messages.push(format!(
            "MLX links the CMake target {linkable_entry:?}, which has no file to link; it was not linked"
        ));
    } else if Path::new(linkable_entry).is_absolute() {
        push_library_file(link_directives, Path::new(linkable_entry));
    } else {
        push_dynamic_library(link_directives, linkable_entry);
    }
}

/// Strips `$<LINK_ONLY:…>` and drops what never reaches a consumer's link:
/// blank entries, build-tree-only targets (`$<BUILD_INTERFACE:…>`, compiled
/// into `libmlx.a` already), and CMake's internal `::@` directory markers.
fn linkable_entry_text(link_library_entry: &str) -> Option<&str> {
    let mut entry_text = link_library_entry.trim();
    while let Some(wrapped_text) = link_only_contents(entry_text) {
        entry_text = wrapped_text.trim();
    }
    let is_unlinkable =
        entry_text.is_empty() || entry_text.starts_with("$<") || entry_text.starts_with("::@");
    if is_unlinkable {
        None
    } else {
        Some(entry_text)
    }
}

fn link_only_contents(entry_text: &str) -> Option<&str> {
    entry_text.strip_prefix("$<LINK_ONLY:")?.strip_suffix('>')
}

/// Links the exact library file CMake recorded. `+verbatim` passes its file
/// name to the linker unchanged (`-l:libopenblas.so.0`), so the linker takes
/// that file rather than whichever `libopenblas` a search folder holds first,
/// and a versioned file without an unversioned link still links. A shared
/// library also gets a run path, so this crate's own test binaries find it at
/// run time when it lives outside the loader's default directories.
fn push_library_file(link_directives: &mut NativeLinkDirectives, library_file_path: &Path) {
    let Some((library_file_name, is_shared_library)) = library_file_kind(library_file_path) else {
        link_directives.warning_messages.push(format!(
            "MLX links {}, which is neither a shared nor a static library; it was not linked",
            library_file_path.display()
        ));
        return;
    };
    if !link_directives
        .linked_library_files
        .contains(&library_file_path.to_path_buf())
    {
        link_directives
            .linked_library_files
            .push(library_file_path.to_path_buf());
    }
    let library_directory = library_file_path.parent().unwrap_or(Path::new("/"));
    push_unique_line(
        &mut link_directives.cargo_directive_lines,
        format!(
            "cargo:rustc-link-search=native={}",
            library_directory.display()
        ),
    );
    if is_shared_library {
        push_unique_line(
            &mut link_directives.cargo_directive_lines,
            format!("cargo:rustc-link-lib=dylib:+verbatim={library_file_name}"),
        );
        push_run_path(link_directives, library_directory);
    } else {
        push_unique_line(
            &mut link_directives.cargo_directive_lines,
            format!("cargo:rustc-link-lib=static:+verbatim={library_file_name}"),
        );
        link_directives.warning_messages.push(format!(
            "MLX links the static library {}; the libraries it depends on are not recorded and may need linking by hand",
            library_file_path.display()
        ));
    }
}

fn push_run_path(link_directives: &mut NativeLinkDirectives, library_directory: &Path) {
    let is_system_directory = SYSTEM_LIBRARY_DIRECTORIES
        .map(Path::new)
        .contains(&library_directory);
    if !is_system_directory {
        push_unique_line(
            &mut link_directives.cargo_directive_lines,
            format!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                library_directory.display()
            ),
        );
    }
}

/// `libopenblas.so` and `libopenblas.so.0` are shared libraries,
/// `libopenblas.a` is static. The `.so` must be the suffix or be followed by
/// a version, so `libfoo.solver.so` is shared and `libfoo.solver` is neither.
fn library_file_kind(library_file_path: &Path) -> Option<(&str, bool)> {
    let library_file_name = library_file_path.file_name()?.to_str()?;
    let mut file_name_segments = library_file_name.split('.');
    let library_stem = file_name_segments.next().unwrap_or_default();
    if !library_stem.starts_with("lib") || library_stem.len() == "lib".len() {
        return None;
    }
    let extension_segments: Vec<&str> = file_name_segments.collect();
    // Walk back over trailing version numbers (`.0`, `.3.27`); a shared
    // library's last other segment is `so`.
    let mut last_unversioned_segment = None;
    for extension_segment in extension_segments.iter().rev() {
        if extension_segment.parse::<u32>().is_err() {
            last_unversioned_segment = Some(*extension_segment);
            break;
        }
    }
    let is_shared_library = last_unversioned_segment == Some("so");
    let is_static_library = extension_segments.last() == Some(&"a");
    if is_shared_library || is_static_library {
        Some((library_file_name, is_shared_library))
    } else {
        None
    }
}

fn push_dynamic_library(link_directives: &mut NativeLinkDirectives, library_name: &str) {
    push_unique_line(
        &mut link_directives.cargo_directive_lines,
        format!("cargo:rustc-link-lib=dylib={library_name}"),
    );
}

/// MLX records a library once per CMake package that found it (BLAS and
/// LAPACK are often the same file), so repeats are dropped; a dynamic
/// library only needs naming once.
fn push_unique_line(cargo_directive_lines: &mut Vec<String>, cargo_directive_line: String) {
    if !cargo_directive_lines.contains(&cargo_directive_line) {
        cargo_directive_lines.push(cargo_directive_line);
    }
}

#[cfg(test)]
mod tests;
