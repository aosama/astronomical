use config::ProblemReportChoice;

use super::SharingSettings;

#[test]
fn should_ask_about_reports_and_leave_the_count_unasked_by_default() {
    let sharing_settings = SharingSettings::default();
    assert_eq!(
        sharing_settings.problem_report_choice,
        ProblemReportChoice::Ask
    );
    assert_eq!(sharing_settings.daily_count_answer, None);
}

#[test]
fn should_name_each_changed_setting_and_nothing_when_unchanged() {
    let current = SharingSettings::default();
    assert!(current.changed_setting_names(&current).is_empty());
    let both_changed = SharingSettings {
        problem_report_choice: ProblemReportChoice::Never,
        daily_count_answer: Some(false),
    };
    assert_eq!(
        current.changed_setting_names(&both_changed),
        ["sharing.problem_reports", "sharing.daily_count"]
    );
    let only_answered = SharingSettings {
        daily_count_answer: Some(true),
        ..SharingSettings::default()
    };
    assert_eq!(
        current.changed_setting_names(&only_answered),
        ["sharing.daily_count"]
    );
}
