//! The naming convention of Astronomical's Metal files, and the scope a
//! crate's place in the repository gives the kernels it owns.
//!
//! `astro_<scope>_[<owner>_]<operation>[_<variant>]`: `astro_` marks
//! Astronomical's own code, the scope says who may use the kernel, and the
//! owner (a family or weight format id) says whose it is. The scope and owner
//! are never chosen by hand: they follow from the owning crate, whose layering
//! already keeps general code out of families and formats.

/// Who may use a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelScope {
    /// `crates/engine/model-primitives`: any model.
    Primitive,
    /// `crates/families/<kind>/<family>`: one model family's architecture.
    Family,
    /// `crates/weight-formats/<format>`: one weight packing on disk.
    Format,
}

/// The scope and owner a crate's location gives its kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateKernelScope {
    pub scope: KernelScope,
    /// The family or format id; `None` for primitives.
    pub owner: Option<String>,
}

/// What a file under a `metal/` folder is, by its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFileKind {
    /// `<kernel>.metal`: one registered kernel, opening with its signature block.
    RegisteredKernel,
    /// `<name>_body.metal`: a body several registered kernels share; never registered.
    SharedBody,
    /// `<name>.h`: helper code that compiles on its own.
    Helper,
}

/// The folder, relative to the repository root, of the helper headers kernels
/// of every crate may include.
pub const SHARED_HEADER_DIRECTORY: &str = "crates/engine/model-primitives/src/metal_common";

/// The folder name that holds a crate's Metal files, beside the Rust that
/// launches them.
pub const KERNEL_DIRECTORY_NAME: &str = "metal";

const PRIMITIVE_CRATE_DIRECTORY: &str = "crates/engine/model-primitives";
const FAMILY_CRATES_DIRECTORY: &str = "crates/families/";
const FORMAT_CRATES_DIRECTORY: &str = "crates/weight-formats/";
const SHARED_BODY_SUFFIX: &str = "_body";

/// The scope of the crate at `crate_directory`, relative to the repository
/// root with `/` separators; `None` for a crate that may own no kernels.
///
/// A format crate's owner is its name up to the first `-`
/// (`jangtq2-qwen35` owns `jangtq2` kernels): the rest names the family the
/// format is packed for, which the kernel does not depend on.
pub fn crate_kernel_scope(crate_directory: &str) -> Option<CrateKernelScope> {
    let crate_directory = crate_directory.trim_end_matches('/');
    if crate_directory == PRIMITIVE_CRATE_DIRECTORY {
        return Some(CrateKernelScope {
            scope: KernelScope::Primitive,
            owner: None,
        });
    }
    if let Some(family_path) = crate_directory.strip_prefix(FAMILY_CRATES_DIRECTORY) {
        let [_kind, family] = family_path.split('/').collect::<Vec<_>>()[..] else {
            return None;
        };
        return Some(CrateKernelScope {
            scope: KernelScope::Family,
            owner: Some(family.replace('-', "_")),
        });
    }
    let format_crate = crate_directory.strip_prefix(FORMAT_CRATES_DIRECTORY)?;
    if format_crate.contains('/') {
        return None;
    }
    let format_owner = format_crate.split('-').next().unwrap_or(format_crate);
    Some(CrateKernelScope {
        scope: KernelScope::Format,
        owner: Some(format_owner.replace('-', "_")),
    })
}

impl CrateKernelScope {
    /// The prefix every Metal file of this crate starts with:
    /// `astro_primitive_`, `astro_family_qwen35_`, `astro_format_jangtq2_`.
    pub fn name_prefix(&self) -> String {
        let scope_word = match self.scope {
            KernelScope::Primitive => "primitive",
            KernelScope::Family => "family",
            KernelScope::Format => "format",
        };
        match &self.owner {
            Some(owner) => format!("astro_{scope_word}_{owner}_"),
            None => format!("astro_{scope_word}_"),
        }
    }
}

/// What the file `file_name` is, provided it follows the convention for a
/// crate whose files start with `name_prefix`.
///
/// # Errors
/// Names the rule the file breaks: an extension other than `.metal` or `.h`,
/// a wrong prefix, an empty operation, or characters outside `[a-z0-9_]`.
pub fn kernel_file_kind(file_name: &str, name_prefix: &str) -> Result<KernelFileKind, String> {
    let (stem, kind) = if let Some(stem) = file_name.strip_suffix(".metal") {
        let kind = if stem.ends_with(SHARED_BODY_SUFFIX) {
            KernelFileKind::SharedBody
        } else {
            KernelFileKind::RegisteredKernel
        };
        (stem, kind)
    } else if let Some(stem) = file_name.strip_suffix(".h") {
        (stem, KernelFileKind::Helper)
    } else {
        return Err(format!(
            "{file_name}: a Metal folder holds only .metal and .h files"
        ));
    };
    let Some(operation) = stem.strip_prefix(name_prefix) else {
        return Err(format!(
            "{file_name}: its name must start with {name_prefix}"
        ));
    };
    if operation.is_empty() || operation == SHARED_BODY_SUFFIX.trim_start_matches('_') {
        return Err(format!(
            "{file_name}: the name needs an operation after {name_prefix}"
        ));
    }
    if !stem.bytes().all(|name_byte| {
        name_byte.is_ascii_lowercase() || name_byte.is_ascii_digit() || name_byte == b'_'
    }) {
        return Err(format!(
            "{file_name}: a name uses only lowercase letters, digits and _"
        ));
    }
    Ok(kind)
}

/// The Rust constant a registered kernel's declaration is generated under:
/// its name in capitals.
pub fn declaration_constant_name(kernel_name: &str) -> String {
    kernel_name.to_ascii_uppercase()
}
