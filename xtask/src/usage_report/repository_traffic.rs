//! The repository's public counters and its visitor traffic. GitHub keeps
//! traffic for 14 days only and shows it to people with write access, so the
//! report reads it fresh each run and says so when it is unavailable.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositoryCounters {
    pub star_count: u64,
    pub fork_count: u64,
    pub watcher_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrafficTotals {
    pub total_count: u64,
    pub unique_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Referrer {
    pub source_name: String,
    pub view_count: u64,
    pub unique_visitor_count: u64,
}

/// Parses `stars<TAB>forks<TAB>watchers`.
pub fn parse_repository_counters_tsv(counters_tsv_text: &str) -> Option<RepositoryCounters> {
    let [star_count, fork_count, watcher_count] = parse_counts::<3>(counters_tsv_text.trim())?;
    Some(RepositoryCounters {
        star_count,
        fork_count,
        watcher_count,
    })
}

/// Parses `count<TAB>uniques` from the views or clones endpoint.
pub fn parse_traffic_totals_tsv(traffic_tsv_text: &str) -> Option<TrafficTotals> {
    let [total_count, unique_count] = parse_counts::<2>(traffic_tsv_text.trim())?;
    Some(TrafficTotals {
        total_count,
        unique_count,
    })
}

/// Parses `referrer<TAB>count<TAB>uniques` lines, skipping malformed ones.
pub fn parse_referrers_tsv(referrers_tsv_text: &str) -> Vec<Referrer> {
    let mut referrers = Vec::new();
    for referrer_line in referrers_tsv_text.lines() {
        let Some((source_name, counts_text)) = referrer_line.split_once('\t') else {
            continue;
        };
        let Some([view_count, unique_visitor_count]) = parse_counts::<2>(counts_text) else {
            continue;
        };
        referrers.push(Referrer {
            source_name: source_name.to_owned(),
            view_count,
            unique_visitor_count,
        });
    }
    referrers
}

fn parse_counts<const FIELD_COUNT: usize>(counts_tsv_line: &str) -> Option<[u64; FIELD_COUNT]> {
    let mut parsed_counts = [0u64; FIELD_COUNT];
    let mut count_fields = counts_tsv_line.split('\t');
    for parsed_count in &mut parsed_counts {
        *parsed_count = count_fields.next()?.trim().parse().ok()?;
    }
    if count_fields.next().is_some() {
        return None;
    }
    Some(parsed_counts)
}
