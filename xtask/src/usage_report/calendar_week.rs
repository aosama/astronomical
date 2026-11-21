//! Calendar arithmetic for grouping releases by week. Release dates arrive as
//! `YYYY-MM-DD` text from the GitHub API; xtask carries no date library, and
//! the days-from-civil conversion (Howard Hinnant's algorithm) is exact for
//! every Gregorian date, so a few lines of integer math beat a new dependency.

/// The Monday that starts the week holding `iso_date`, as `YYYY-MM-DD`, or
/// `None` when the text is not a valid date.
pub fn monday_of_week(iso_date: &str) -> Option<String> {
    let (year, month, day) = parse_iso_date(iso_date)?;
    let days_since_epoch = days_from_civil(year, month, day);
    // 1970-01-01 was a Thursday, so shifting by 3 puts Monday at remainder 0.
    let days_since_monday = (days_since_epoch + 3).rem_euclid(7);
    let (monday_year, monday_month, monday_day) =
        civil_from_days(days_since_epoch - days_since_monday);
    Some(format!(
        "{monday_year:04}-{monday_month:02}-{monday_day:02}"
    ))
}

fn parse_iso_date(iso_date: &str) -> Option<(i64, u32, u32)> {
    let date_text = iso_date.get(..10)?;
    let mut date_parts = date_text.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    Some((year, month, day))
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 31,
    }
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let march_based_year = if month <= 2 { year - 1 } else { year };
    let era = march_based_year.div_euclid(400);
    let year_of_era = march_based_year - era * 400;
    let march_based_month = i64::from((month + 9) % 12);
    let day_of_year = (153 * march_based_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let shifted_days = days_since_epoch + 719_468;
    let era = shifted_days.div_euclid(146_097);
    let day_of_era = shifted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let march_based_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * march_based_month + 2) / 5 + 1) as u32;
    let month = if march_based_month < 10 {
        march_based_month + 3
    } else {
        march_based_month - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}
