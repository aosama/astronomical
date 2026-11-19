//! The one header and footer every page of the public website carries.
//!
//! The pages are hand-written static HTML served by GitHub Pages with no
//! build step, so each page holds its own copy of this markup. Those copies
//! drifted apart (seven different top menus across eight pages) until the
//! links were defined here once and stamped into every page by
//! `cargo xtask site-layout`. Rendering happens at commit time rather than in
//! the browser so a visitor without JavaScript, and every search engine, sees
//! the same menu, and the pages stay plain files a reviewer can read.

/// Shown under the brand in every footer.
pub(crate) const SITE_TAGLINE: &str = "Local language, vision, and image models for Apple silicon.";

/// Tells whoever opens a page why editing its header or footer by hand is
/// pointless: the next `cargo xtask site-layout` run would overwrite it.
const GENERATED_NOTICE: &str = "<!-- Shared by every page: edit xtask/src/site_layout/site_header_and_footer.rs, then run `cargo xtask site-layout`. -->";

pub(crate) enum SiteLinkTarget {
    /// A page of this site, as a path from the site root (`site/`), which
    /// is rewritten relative to each page so the site also works from disk.
    SitePage(&'static str),
    External(&'static str),
}

pub(crate) struct SiteLink {
    pub(crate) label: &'static str,
    pub(crate) target: SiteLinkTarget,
    /// Pages under this folder belong to this link's section, so the link
    /// is shown as current there although it points at another page.
    pub(crate) section_folder: Option<&'static str>,
}

/// The menu, in the order a visitor reads it. The maintainer chose to show
/// every destination at the top rather than tucking Privacy and Discord
/// into the footer, so header and footer list the same links.
pub(crate) const SITE_LINKS: [SiteLink; 7] = [
    site_page_link("Models", "models.html"),
    SiteLink {
        label: "Field notes",
        target: SiteLinkTarget::SitePage("index.html#reports"),
        section_folder: Some("reports/"),
    },
    site_page_link("Local API", "local-api.html"),
    site_page_link("Support", "support.html"),
    site_page_link("Privacy", "privacy.html"),
    external_link("Discord", "https://discord.gg/E8a5qmhRhr"),
    external_link("Source", "https://github.com/aosama/astronomical"),
];

const fn site_page_link(label: &'static str, site_path: &'static str) -> SiteLink {
    SiteLink {
        label,
        target: SiteLinkTarget::SitePage(site_path),
        section_folder: None,
    }
}

const fn external_link(label: &'static str, address: &'static str) -> SiteLink {
    SiteLink {
        label,
        target: SiteLinkTarget::External(address),
        section_folder: None,
    }
}

/// `./` for a page at the site root, `../../` for a page two folders down.
pub(crate) fn site_root_prefix(page_path: &str) -> String {
    let folder_depth = page_path.matches('/').count();
    if folder_depth == 0 {
        "./".to_owned()
    } else {
        "../".repeat(folder_depth)
    }
}

/// The `aria-current` value for a link on this page: `page` on the page it
/// opens, `true` inside the section it heads, nothing elsewhere.
pub(crate) fn current_marker(site_link: &SiteLink, page_path: &str) -> Option<&'static str> {
    if let Some(section_folder) = site_link.section_folder
        && page_path.starts_with(section_folder)
    {
        return Some("true");
    }
    match site_link.target {
        SiteLinkTarget::SitePage(site_path) if site_path == page_path => Some("page"),
        _ => None,
    }
}

fn link_markup(site_link: &SiteLink, page_path: &str, should_mark_current: bool) -> String {
    let address = match site_link.target {
        SiteLinkTarget::SitePage(site_path) => {
            format!("{}{site_path}", site_root_prefix(page_path))
        }
        SiteLinkTarget::External(address) => address.to_owned(),
    };
    let current_attribute = match current_marker(site_link, page_path) {
        Some(marker) if should_mark_current => format!(" aria-current=\"{marker}\""),
        _ => String::new(),
    };
    format!(
        "<a href=\"{address}\"{current_attribute}>{}</a>",
        site_link.label
    )
}

fn brand_markup(page_path: &str, mark_size_pixels: u32, aria_label: &str) -> Vec<String> {
    let prefix = site_root_prefix(page_path);
    vec![
        format!("  <a class=\"brand\" href=\"{prefix}index.html\"{aria_label}>"),
        format!(
            "    <img src=\"{prefix}assets/mark.svg\" alt=\"\" width=\"{mark_size_pixels}\" height=\"{mark_size_pixels}\">"
        ),
        "    <span>astronomical</span>".to_owned(),
        "  </a>".to_owned(),
    ]
}

/// Joins element lines so the first replaces the opening tag in place and
/// the rest keep the indentation that opening tag had in the page.
fn indent_lines(element_lines: &[String], indentation: &str) -> String {
    element_lines.join(&format!("\n{indentation}"))
}

pub(crate) fn render_site_header(page_path: &str, indentation: &str) -> String {
    let mut header_lines = vec![
        "<header class=\"site-header\">".to_owned(),
        format!("  {GENERATED_NOTICE}"),
    ];
    header_lines.extend(brand_markup(
        page_path,
        32,
        " aria-label=\"Astronomical home\"",
    ));
    header_lines.push("  <nav class=\"site-nav\" aria-label=\"Primary navigation\">".to_owned());
    for site_link in &SITE_LINKS {
        header_lines.push(format!("    {}", link_markup(site_link, page_path, true)));
    }
    header_lines.push("  </nav>".to_owned());
    header_lines.push("</header>".to_owned());
    indent_lines(&header_lines, indentation)
}

pub(crate) fn render_site_footer(page_path: &str, indentation: &str) -> String {
    let mut footer_lines = vec![
        "<footer class=\"site-footer\">".to_owned(),
        format!("  {GENERATED_NOTICE}"),
    ];
    footer_lines.extend(brand_markup(page_path, 26, ""));
    footer_lines.push(format!("  <p>{SITE_TAGLINE}</p>"));
    footer_lines.push("  <nav aria-label=\"Site links\">".to_owned());
    for site_link in &SITE_LINKS {
        footer_lines.push(format!("    {}", link_markup(site_link, page_path, false)));
    }
    footer_lines.push("  </nav>".to_owned());
    footer_lines.push("</footer>".to_owned());
    indent_lines(&footer_lines, indentation)
}
