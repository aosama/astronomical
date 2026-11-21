//! Guards the rule that the Mac App Store build sends no problem reports:
//! its privacy declarations say no data is collected (issue #2413).

use std::path::PathBuf;

use config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};

use crate::ResolvedRuntimeConfigResolver;
use crate::application::problem_report_desk_for;

fn resolver_for(instance_paths: AstronomicalInstancePaths) -> ResolvedRuntimeConfigResolver {
    ResolvedRuntimeConfigResolver::for_instance(instance_paths, PathBuf::new())
}

#[test]
fn should_give_the_app_store_build_no_report_desk() {
    let app_store_paths = AstronomicalInstancePaths::for_application_group_container(
        PathBuf::from("/Users/example/Library/Group Containers/TEAM123456.astronomical"),
        AstronomicalRuntimeInstance::Stable,
    );
    assert!(problem_report_desk_for(&resolver_for(app_store_paths)).is_none());
}

#[test]
fn should_give_the_direct_download_build_a_report_desk() {
    let state_home = tempfile::tempdir().expect("temporary home");
    let direct_paths = AstronomicalInstancePaths::for_home_directory(
        state_home.path().to_path_buf(),
        AstronomicalRuntimeInstance::Stable,
    );
    assert!(problem_report_desk_for(&resolver_for(direct_paths)).is_some());
}
