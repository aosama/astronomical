//! Which changed paths open which CI lane, and which count as static
//! content. `classify_scope` turns these answers into the verdict; keeping the
//! path rules apart means a new lane adds its paths here without touching how
//! the verdict is computed from git history.

/// Paths whose change can alter the text `cargo xtask notices` generates:
/// the resolved dependency graph, the pinned cargo-about, its configuration
/// and template, and the committed artifact itself. Deliberately narrower
/// than the digest inputs in `notices.rs`: workspace crates are `private`
/// (ignored by about.toml), so their manifests and license files only move
/// the generated text through `Cargo.lock`, and gating on every manifest edit
/// would pay a harvest for workspace version bumps that cannot change it.
const NOTICES_FRESHNESS_INPUT_PATHS: &[&str] = &[
    "Cargo.lock",
    "third-party/tool-pins/Cargo.toml",
    "third-party/about.toml",
    "third-party/rust-dependency-notices.hbs",
    "third-party/RUST_DEPENDENCY_NOTICES",
    "third-party/RUST_DEPENDENCY_NOTICES.digest",
];

/// Paths whose change gives a default-branch run something to publish: the
/// `sccache`, native download and SwiftPM entries of the `github_actions_cache`
/// are keyed on these, and pull requests can only restore what the default
/// branch saved. Code-only merges saved nothing in every default-branch run
/// measured, so only these still need that run. The workflow and the
/// `astronomical-cache` action join them because they define the
/// `github_actions_cache` keys. What the xtask binary saved in the
/// `github_actions_cache` is keyed on stays out on purpose: those paths alone touched 11 of
/// 60 merges, each a full run of about 12 job-minutes, while a missed binary costs about
/// 9 seconds to rebuild until the next full run saves it.
const GITHUB_ACTIONS_CACHE_INPUT_PATH_PREFIXES: &[&str] = &[
    "Cargo.lock",
    "rust-toolchain.toml",
    "third-party/tool-pins/",
    "third-party/native-dependency-manifest.cmake",
    "third-party/pins/",
    ".github/workflows/ci.yml",
    ".github/actions/",
    "apps/astronomical-menu/",
    "apps/thin-talk/",
];

/// Everything the Swift and Node lane builds or reads. It compiles no Rust and reads
/// nothing else, so a change outside these cannot change what the lane reports.
const SWIFT_NODE_INPUT_PATH_PREFIXES: &[&str] = &[
    "apps/astronomical-menu/",
    "apps/thin-talk/",
    "apps/daemon/console/",
    "scripts/internal/",
    "scripts/ci/",
    "site/schemas/",
    ".node-version",
    ".github/workflows/ci.yml",
    ".github/actions/",
];

/// The release catalog's data file. The engine compiles it in, so it is not
/// static content; but a change to it can only change what the catalog crate
/// parses, which the catalog lane checks on Linux in about a minute, so on its
/// own (with only static content beside it) it needs none of the macOS lanes.
const CATALOG_DATA_PATH: &str = "registry/download_catalog.json";

/// Everything the catalog lane builds and tests: the catalog data and the
/// crate that parses it. A change to either opens the lane.
const CATALOG_CHECK_INPUT_PATH_PREFIXES: &[&str] =
    &[CATALOG_DATA_PATH, "crates/tooling/download-catalog/"];

/// The OpenCode plugin: a standalone npm package that compiles no Rust and
/// talks to the app only over loopback HTTP, so a change inside it alone (with
/// only static content beside it) is proven by the Linux plugin lane and needs
/// none of the macOS lanes.
const OPENCODE_PLUGIN_DIRECTORY: &str = "integrations/opencode/";

/// The `/v1/models` sample both sides of the contract read: a Rust test holds
/// it to what the contract types serialize, and the plugin's tests parse it.
/// It is an input of both lanes, so a field rename fails on whichever side has
/// not followed.
const MODEL_LIST_FIXTURE_PATH: &str =
    "crates/contracts/rest-contract/fixtures/openai_model_list.json";

/// Everything the OpenCode plugin lane installs, builds or reads.
const OPENCODE_PLUGIN_CHECK_INPUT_PATH_PREFIXES: &[&str] = &[
    OPENCODE_PLUGIN_DIRECTORY,
    MODEL_LIST_FIXTURE_PATH,
    ".node-version",
    ".github/workflows/ci.yml",
];

/// Classifies one page/site artifact change as static-only. POSIX `case`
/// pattern semantics: `*` spans `/`, so `site/*` covers nested paths.
pub fn is_static_only_path(changed_path: &str) -> bool {
    is_published_content_path(changed_path) || is_repository_housekeeping_path(changed_path)
}

pub(crate) fn is_published_content_path(changed_path: &str) -> bool {
    changed_path.starts_with("site/")
        || changed_path.ends_with(".md")
        || changed_path.starts_with(".github/assets/")
        || changed_path == ".github/workflows/pages.yml"
}

/// None of these reach a build or a test: a fresh checkout has nothing to ignore.
pub(crate) fn is_repository_housekeeping_path(changed_path: &str) -> bool {
    changed_path == ".github/dependabot.yml"
        || changed_path == ".gitignore"
        || changed_path.ends_with("/.gitignore")
}

/// True when a change to this path gives a default-branch run an entry of the
/// `github_actions_cache` to save.
pub fn is_github_actions_cache_input_path(changed_path: &str) -> bool {
    for github_actions_cache_input_prefix in GITHUB_ACTIONS_CACHE_INPUT_PATH_PREFIXES {
        if changed_path.starts_with(github_actions_cache_input_prefix) {
            return true;
        }
    }
    false
}

/// True when the changed path can make the committed dependency notices
/// stale, so the pull request must prove them fresh before it merges.
pub fn is_notices_freshness_input_path(changed_path: &str) -> bool {
    NOTICES_FRESHNESS_INPUT_PATHS.contains(&changed_path)
}

/// True when a change to this path can change what the catalog lane checks.
pub fn is_catalog_check_input_path(changed_path: &str) -> bool {
    for catalog_check_input_prefix in CATALOG_CHECK_INPUT_PATH_PREFIXES {
        if changed_path.starts_with(catalog_check_input_prefix) {
            return true;
        }
    }
    false
}

/// True when a change to this path can change what the OpenCode plugin lane checks.
pub fn is_opencode_plugin_check_input_path(changed_path: &str) -> bool {
    for opencode_plugin_input_prefix in OPENCODE_PLUGIN_CHECK_INPUT_PATH_PREFIXES {
        if changed_path.starts_with(opencode_plugin_input_prefix) {
            return true;
        }
    }
    false
}

pub(crate) fn any_opencode_plugin_check_input(changed_paths: &[String]) -> bool {
    for changed_path in changed_paths {
        if is_opencode_plugin_check_input_path(changed_path) {
            return true;
        }
    }
    false
}

pub(crate) fn is_catalog_data_path(changed_path: &str) -> bool {
    changed_path == CATALOG_DATA_PATH
}

pub(crate) fn is_opencode_plugin_path(changed_path: &str) -> bool {
    changed_path.starts_with(OPENCODE_PLUGIN_DIRECTORY)
}

/// The shape of a change one Linux lane proves on its own: it touches the
/// lane's own paths and otherwise only static content. A plain function
/// pointer names the lane's paths so each lane's rule stays a named function.
pub(crate) fn touches_only_lane_and_static_paths(
    changed_paths: &[String],
    is_lane_path: fn(&str) -> bool,
) -> bool {
    let mut touches_lane_path = false;
    for changed_path in changed_paths {
        if is_lane_path(changed_path) {
            touches_lane_path = true;
        } else if !is_static_only_path(changed_path) {
            return false;
        }
    }
    touches_lane_path
}

pub(crate) fn any_catalog_check_input(changed_paths: &[String]) -> bool {
    for changed_path in changed_paths {
        if is_catalog_check_input_path(changed_path) {
            return true;
        }
    }
    false
}

/// True when a change touches only the catalog data and static content, so
/// the catalog lane alone proves it.
pub(crate) fn is_catalog_only_change(changed_paths: &[String]) -> bool {
    touches_only_lane_and_static_paths(changed_paths, is_catalog_data_path)
}

/// True when a change touches only the OpenCode plugin and static content, so
/// the plugin lane alone proves it.
pub(crate) fn is_opencode_plugin_only_change(changed_paths: &[String]) -> bool {
    touches_only_lane_and_static_paths(changed_paths, is_opencode_plugin_path)
}

/// True when a change to this path can change what the Swift and Node lane checks.
pub fn is_swift_node_input_path(changed_path: &str) -> bool {
    for swift_node_input_prefix in SWIFT_NODE_INPUT_PATH_PREFIXES {
        if changed_path.starts_with(swift_node_input_prefix) {
            return true;
        }
    }
    false
}

pub(crate) fn any_swift_node_input(changed_paths: &[String]) -> bool {
    for changed_path in changed_paths {
        if is_swift_node_input_path(changed_path) {
            return true;
        }
    }
    false
}

pub(crate) fn any_notices_freshness_input(changed_paths: &[String]) -> bool {
    for changed_path in changed_paths {
        if is_notices_freshness_input_path(changed_path) {
            return true;
        }
    }
    false
}
