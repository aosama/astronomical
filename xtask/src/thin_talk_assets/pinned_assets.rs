//! The pinned third-party web assets the Thin Talk canvas renders with.
//! Bumping one means changing its package path and digest here together.

use super::PinnedAsset;

/// Scripts and stylesheets: vendored name, package path under the CDN, and
/// pinned SHA-256.
pub(crate) const PINNED_ASSETS: [PinnedAsset; 8] = [
    PinnedAsset {
        vendored_name: "marked.min.js",
        package_path: "marked@15.0.7/marked.min.js",
        sha256: Some("934e3e36e9e2da0afb1a6e75075bb0f09af05293a844e84a7477ef40911c349a"),
    },
    PinnedAsset {
        vendored_name: "purify.min.js",
        package_path: "dompurify@3.2.4/dist/purify.min.js",
        sha256: Some("8eb41b658831fab175fad9bcd00fcb2d84e0ed3a25a55053d4ecd4444b8b43a0"),
    },
    PinnedAsset {
        vendored_name: "morphdom-umd.min.js",
        package_path: "morphdom@2.7.7/dist/morphdom-umd.min.js",
        sha256: Some("ad1aaf5441eb2798b99dd03a41bea26562cd634dfc9845d3b8c5fbce560a6bac"),
    },
    PinnedAsset {
        vendored_name: "highlight.min.js",
        package_path: "@highlightjs/cdn-assets@11.11.1/highlight.min.js",
        sha256: Some("c4a399dd6f488bc97a3546e3476747b3e714c99c57b9473154c6fb8d259b9381"),
    },
    PinnedAsset {
        vendored_name: "highlight-github-dark.min.css",
        package_path: "@highlightjs/cdn-assets@11.11.1/styles/github-dark.min.css",
        sha256: Some("9f208d022102b1d0c7aebfecd8e42ca7997d5de636649d2b31ea63093d809019"),
    },
    PinnedAsset {
        vendored_name: "katex/katex.min.js",
        package_path: "katex@0.16.22/dist/katex.min.js",
        sha256: Some("e8d885505949f3a5f4abdd5dd0d53696bd1371ad26ffbf4f310dcd77c8cdae89"),
    },
    PinnedAsset {
        vendored_name: "katex/katex.min.css",
        package_path: "katex@0.16.22/dist/katex.min.css",
        sha256: Some("19095127357ed6d29fe0a63a6b000c913a89f7f1963b765dd3715e97c9852e75"),
    },
    PinnedAsset {
        vendored_name: "mermaid.min.js",
        package_path: "mermaid@11.12.2/dist/mermaid.min.js",
        sha256: Some("d0830a6c05546e9edb8fe20a8f545f3e0dc7c4c3134d584bad9c13a99d7a71e0"),
    },
];

/// KaTeX renders its fonts through relative URLs; serving them as plain
/// assets keeps the stylesheet unchanged and auditable.
pub(crate) const PINNED_FONTS: [PinnedAsset; 20] = [
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_AMS-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_AMS-Regular.woff2",
        sha256: Some("0cdd387c9590a1a9f9794560022dbb59654a7d86f187aa0c81495ad42d3a7308"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Caligraphic-Bold.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Caligraphic-Bold.woff2",
        sha256: Some("de7701e42cf1f4cf0b766c03fb27977207eee2f4fd5d76fa82188406da43ea4c"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Caligraphic-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Caligraphic-Regular.woff2",
        sha256: Some("5d53e70ad607c2352162dec9e0923fb54ecdafaccbf604cd8dcf7d00facb989b"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Fraktur-Bold.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Fraktur-Bold.woff2",
        sha256: Some("74444efd593c005e3f4573b44524704c0af0a937fe911cca9e94068d0d140d3f"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Fraktur-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Fraktur-Regular.woff2",
        sha256: Some("51814d270d06ff0255dba0799994fa4d8c84d11f09951d47595f4abb1f3602dc"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Main-Bold.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Main-Bold.woff2",
        sha256: Some("0f60d1b897938ec918c8ce073092411baf9438f6739465693ff18b0f9d20b021"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Main-BoldItalic.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Main-BoldItalic.woff2",
        sha256: Some("99cd42a3c072d918f2f44984a807cf7aa16e13545fd0875fc07c6c65f99e715b"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Main-Italic.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Main-Italic.woff2",
        sha256: Some("97479ca6cce906abc961ecac96faa5f9ca2e61b8e7670d475826bcdee9a7c267"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Main-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Main-Regular.woff2",
        sha256: Some("c2342cd8b869e01752a9321dc17213fc40d4d04c79688c1d43f2cf316abd7866"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Math-BoldItalic.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Math-BoldItalic.woff2",
        sha256: Some("dc47344dbb6cb5b655c8460d561f4df5f501b90c804ad3c6cec65fe322351ab1"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Math-Italic.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Math-Italic.woff2",
        sha256: Some("7af58c5ec8f132a2ddde9027c6d7814decce4d3b822a11192a42a20e2e973264"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_SansSerif-Bold.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_SansSerif-Bold.woff2",
        sha256: Some("e99ae51144bf1232efcc1bfe5add36262c6866b0faab24fa75740e1b98577a62"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_SansSerif-Italic.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_SansSerif-Italic.woff2",
        sha256: Some("00b26ac825e2095056396e0553b8ac26d3f8ad158c3826e28b4c45b385c4714a"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_SansSerif-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_SansSerif-Regular.woff2",
        sha256: Some("68e8c73ef42afd3ccec58bf0fba302cce448938e7fc020a5e31f8a952eee1342"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Script-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Script-Regular.woff2",
        sha256: Some("036d4e95149b69ff9bcc0cd55771efeb25ffa3947293e69acd78d5ac328c684b"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Size1-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Size1-Regular.woff2",
        sha256: Some("6b47c40166b6dbe21a5dfca7718413f2147fd2399be1ba605d8ad39cedf25dfe"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Size2-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Size2-Regular.woff2",
        sha256: Some("d04c54219f9eaec6d4d4fd42dfb28785975a4794d6b2fc71e566b9cd6db842dd"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Size3-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Size3-Regular.woff2",
        sha256: Some("73d591271b1604960cb10bb90fee021670af7297017e0e98480b332d11f51995"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Size4-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Size4-Regular.woff2",
        sha256: Some("a4af7d414440a1c1790825cfb700cf9cf43b0f2c4b04f0ebc523011ad9853ec0"),
    },
    PinnedAsset {
        vendored_name: "katex/fonts/KaTeX_Typewriter-Regular.woff2",
        package_path: "katex@0.16.22/dist/fonts/KaTeX_Typewriter-Regular.woff2",
        sha256: Some("71d517d67827787cfabdf186914cc3358eda539e37931941f2b2fd4a21f68c0b"),
    },
];

/// Licenses are fetched once and kept; they carry no digest pin because
/// upstream repackages them without changing the licensed terms.
pub(crate) const PINNED_LICENSES: [PinnedAsset; 6] = [
    PinnedAsset {
        vendored_name: "LICENSE-marked.md",
        package_path: "marked@15.0.7/LICENSE.md",
        sha256: None,
    },
    PinnedAsset {
        vendored_name: "LICENSE-dompurify.txt",
        package_path: "dompurify@3.2.4/LICENSE",
        sha256: None,
    },
    PinnedAsset {
        vendored_name: "LICENSE-morphdom.txt",
        package_path: "morphdom@2.7.7/LICENSE",
        sha256: None,
    },
    PinnedAsset {
        vendored_name: "LICENSE-highlightjs.txt",
        package_path: "@highlightjs/cdn-assets@11.11.1/LICENSE",
        sha256: None,
    },
    PinnedAsset {
        vendored_name: "katex/LICENSE-katex.txt",
        package_path: "katex@0.16.22/LICENSE",
        sha256: None,
    },
    PinnedAsset {
        vendored_name: "LICENSE-mermaid.txt",
        package_path: "mermaid@11.12.2/LICENSE",
        sha256: None,
    },
];
