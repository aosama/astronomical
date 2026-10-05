//! Coverage engine contracts that stay hermetic: gap naming, drift
//! detection, and the exclusions registry.

mod mlx_c_coverage;

use mlx_c_coverage::coverage_engine::{
    BridgeInventory, CoverageGaps, Exclusions, HeaderSurface, coverage_gaps, parse_header_surface,
};
use std::collections::BTreeSet;

fn surface_with_functions(names: &[&str]) -> HeaderSurface {
    HeaderSurface {
        functions: names.iter().map(|name| (*name).to_owned()).collect(),
        types: BTreeSet::new(),
    }
}

fn inventory_with_functions(names: &[&str]) -> BridgeInventory {
    BridgeInventory {
        functions: names.iter().map(|name| (*name).to_owned()).collect(),
        types: BTreeSet::new(),
    }
}

const NO_EXCLUSIONS: Exclusions = &[];

#[test]
fn should_name_every_unbridged_symbol_in_the_gap_report() {
    let surface = surface_with_functions(&["mlx_sum", "mlx_fft_fft", "mlx_linalg_inv"]);
    let bridged = inventory_with_functions(&["mlx_sum"]);

    let gaps = coverage_gaps(&surface, &bridged, NO_EXCLUSIONS);

    assert!(
        gaps.unbridged_functions.contains("mlx_fft_fft")
            && gaps.unbridged_functions.contains("mlx_linalg_inv"),
        "every unbridged symbol must be named: {:?}",
        gaps
    );
    assert_eq!(gaps.report().matches("unbridged function").count(), 2);
}

#[test]
fn should_flag_stale_bridged_symbols_missing_from_the_headers() {
    let surface = surface_with_functions(&["mlx_sum"]);
    let bridged = inventory_with_functions(&["mlx_sum", "mlx_retired_operation"]);

    let gaps = coverage_gaps(&surface, &bridged, NO_EXCLUSIONS);

    assert!(
        gaps.stale_functions.contains("mlx_retired_operation"),
        "a bridge entry with no upstream declaration is stale: {:?}",
        gaps
    );
    assert!(gaps.unbridged_functions.is_empty());
}

#[test]
fn should_detect_a_simulated_upstream_addition_as_a_drift_gap() {
    // A future pin adds a new public operation; the contract must fail and
    // name it before any bridge work happens.
    let fixture_header = "int mlx_sum(mlx_array* res, const mlx_stream s);\n\
         int mlx_zz_future_operation(mlx_array* res, const mlx_array a, const mlx_stream s);\n";
    let surface = parse_header_surface(fixture_header);
    let bridged = inventory_with_functions(&["mlx_sum"]);

    let gaps = coverage_gaps(&surface, &bridged, NO_EXCLUSIONS);

    assert!(
        gaps.unbridged_functions.contains("mlx_zz_future_operation"),
        "an upstream addition must surface as a named drift gap: {:?}",
        surface
    );
}

#[test]
fn should_skip_internal_underscore_prefixed_declarations() {
    // The headers declare `_mlx_*` helpers marked internal; they are outside
    // the public surface this contract guards.
    let fixture_header =
        "int mlx_array_eval(mlx_array arr);\nint _mlx_array_wait(const mlx_array arr);\n";

    let surface = parse_header_surface(fixture_header);

    assert!(
        surface.functions.contains("mlx_array_eval"),
        "{:?}",
        surface
    );
    assert!(
        !surface.functions.iter().any(|name| name.contains("wait")),
        "internal helpers stay out of the surface: {:?}",
        surface
    );
}

#[test]
fn should_collect_typedef_types_from_the_headers() {
    let fixture_header = "typedef struct mlx_array_ mlx_array;\n\
         typedef enum mlx_fft_norm_ { MLX_FFT_NORM_NONE } mlx_fft_norm;\n\
         int mlx_fft_fft(mlx_array* res, mlx_fft_norm norm, const mlx_stream s);\n";

    let surface = parse_header_surface(fixture_header);

    assert!(surface.types.contains("mlx_array"), "{:?}", surface);
    assert!(surface.types.contains("mlx_array_"), "{:?}", surface);
    assert!(surface.types.contains("mlx_fft_norm_"), "{:?}", surface);
    assert!(surface.functions.contains("mlx_fft_fft"), "{:?}", surface);
}

#[test]
fn should_collect_the_closing_alias_of_a_struct_typedef_with_members() {
    // Struct bodies carry a `;` per member, so the alias only appears in the
    // `}`-leading closer fragment; opaque public handles are declared this
    // way throughout the pinned headers.
    let fixture_header = "typedef struct mlx_stream_ {\n\
         void* ctx;\n\
         } mlx_stream;\n\
         int mlx_stream_to_string(mlx_string** res, const mlx_stream s);\n";

    let surface = parse_header_surface(fixture_header);

    assert!(surface.types.contains("mlx_stream_"), "{:?}", surface);
    assert!(surface.types.contains("mlx_stream"), "{:?}", surface);
    assert!(
        surface.functions.contains("mlx_stream_to_string"),
        "{:?}",
        surface
    );
}

#[test]
fn should_parse_the_first_declaration_after_preprocessor_guards() {
    // Header files open with `#ifdef`/`#endif` guards around `extern "C"`;
    // preprocessor text must never leak into a return type or hide
    // conditionally compiled declarations.
    let fixture_header = "#ifndef MLX_GUARD_H\n#define MLX_GUARD_H\n\
         #ifdef __cplusplus\nextern \"C\" {\n#endif\n\
         int mlx_version(mlx_string* str_);\n\
         #ifdef HAS_BFLOAT16\n\
         int mlx_array_item_bfloat16(uint16_t* res, const mlx_array arr);\n\
         #endif\n\
         #ifdef __cplusplus\n}\n#endif\n#endif\n";

    let surface = parse_header_surface(fixture_header);

    assert!(surface.functions.contains("mlx_version"), "{:?}", surface);
    assert!(
        surface.functions.contains("mlx_array_item_bfloat16"),
        "{:?}",
        surface
    );
}

#[test]
fn should_suppress_a_justified_stale_bridge_entry() {
    // Bridged symbols with no header declaration (macros bindgen cannot see)
    // stay legitimate when the registry documents them.
    let surface = surface_with_functions(&["mlx_sum"]);
    let bridged = inventory_with_functions(&["mlx_sum", "mlx_error"]);
    let exclusions: Exclusions = &[(
        "mlx_error",
        "A preprocessor macro, not a function declaration.",
    )];

    let gaps = coverage_gaps(&surface, &bridged, exclusions);

    assert!(gaps.is_empty(), "{:?}", gaps);
}

#[test]
fn should_reject_exclusions_naming_unknown_symbols() {
    let surface = surface_with_functions(&["mlx_sum"]);
    let bridged = inventory_with_functions(&["mlx_sum"]);
    let exclusions: Exclusions = &[("mlx_typo_symbol", "Reasonable words, no such symbol.")];

    let gaps = coverage_gaps(&surface, &bridged, exclusions);

    assert!(
        gaps.unjustified_exclusions
            .contains(&"mlx_typo_symbol".to_owned()),
        "an exclusion for a symbol that exists nowhere must be rejected: {:?}",
        gaps
    );
}

#[test]
fn should_accept_exclusions_only_with_justification() {
    let surface = surface_with_functions(&["mlx_sum", "mlx_backend_only_operation"]);
    let bridged = inventory_with_functions(&["mlx_sum"]);
    let exclusions: Exclusions = &[(
        "mlx_backend_only_operation",
        "The backend is absent from the platform native image.",
    )];

    let gaps = coverage_gaps(&surface, &bridged, exclusions);

    assert!(gaps.is_empty(), "{:?}", gaps);
}

#[test]
fn should_reject_exclusions_without_justification() {
    let surface = surface_with_functions(&["mlx_backend_only_operation"]);
    let bridged = inventory_with_functions(&[]);
    let exclusions: Exclusions = &[("mlx_backend_only_operation", "")];

    let gaps = coverage_gaps(&surface, &bridged, exclusions);

    assert!(
        gaps.unjustified_exclusions
            .contains(&"mlx_backend_only_operation".to_owned()),
        "an empty justification must be rejected: {:?}",
        gaps
    );
}

#[test]
fn should_report_a_clean_contract_when_surface_and_bridge_agree() {
    let surface = surface_with_functions(&["mlx_sum", "mlx_fft_fft"]);
    let bridged = inventory_with_functions(&["mlx_sum", "mlx_fft_fft"]);

    let gaps = coverage_gaps(&surface, &bridged, NO_EXCLUSIONS);

    assert!(gaps.is_empty(), "{:?}", gaps);
    assert!(CoverageGaps::default().is_empty());
}
