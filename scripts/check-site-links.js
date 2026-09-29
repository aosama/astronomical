#!/usr/bin/env node
// Owns the static-site link contract: every relative href/src inside site/
// must resolve to a real file in the published output. Kept dependency-free so
// the static CI job can run it on a plain Node checkout with no install step.

"use strict";

const fs = require("node:fs");
const path = require("node:path");

const siteRootDirectory = path.join(__dirname, "..", "site");
const skippedUrlPrefixes = ["http://", "https://", "mailto:", "data:", "javascript:"];

function listHtmlFilesRecursively(directory, directoryBase) {
    const htmlFiles = [];
    for (const entry of fs.readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
        const entryPath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            htmlFiles.push(...listHtmlFilesRecursively(entryPath, directoryBase));
        } else if (entry.isFile() && entry.name.endsWith(".html")) {
            htmlFiles.push(path.relative(directoryBase, entryPath));
        }
    }
    return htmlFiles;
}

function extractLinkTargets(htmlContent) {
    const linkTargets = [];
    const attributePattern = /(?:href|src)="([^"]*)"/g;
    let attributeMatch = attributePattern.exec(htmlContent);
    while (attributeMatch !== null) {
        linkTargets.push(attributeMatch[1]);
        attributeMatch = attributePattern.exec(htmlContent);
    }
    return linkTargets;
}

function stripFragmentAndQuery(linkTarget) {
    const fragmentIndex = linkTarget.indexOf("#");
    const trimmedTarget = fragmentIndex === -1 ? linkTarget : linkTarget.slice(0, fragmentIndex);
    const queryIndex = trimmedTarget.indexOf("?");
    return queryIndex === -1 ? trimmedTarget : trimmedTarget.slice(0, queryIndex);
}

function isSkippedLink(linkTarget) {
    return (
        linkTarget === "" ||
        linkTarget.startsWith("#") ||
        skippedUrlPrefixes.some((prefix) => linkTarget.startsWith(prefix))
    );
}

function resolveLinkTarget(htmlFilePath, linkTarget) {
    const withoutFragmentOrQuery = stripFragmentAndQuery(linkTarget);
    let decodedTarget;
    try {
        decodedTarget = decodeURIComponent(withoutFragmentOrQuery);
    } catch {
        return { error: `percent-encoded target does not decode: ${linkTarget}` };
    }
    if (decodedTarget.startsWith("/")) {
        return { resolvedPath: path.join(siteRootDirectory, decodedTarget.slice(1)) };
    }
    return { resolvedPath: path.join(siteRootDirectory, path.dirname(htmlFilePath), decodedTarget) };
}

function checkSiteLinks() {
    if (!fs.existsSync(siteRootDirectory)) {
        throw new Error(`static site directory does not exist: site/`);
    }
    const htmlFiles = listHtmlFilesRecursively(siteRootDirectory, siteRootDirectory);
    const violations = [];
    let checkedLinkCount = 0;

    for (const htmlFilePath of htmlFiles) {
        const htmlContent = fs.readFileSync(path.join(siteRootDirectory, htmlFilePath), "utf8");
        for (const linkTarget of extractLinkTargets(htmlContent)) {
            if (isSkippedLink(linkTarget)) {
                continue;
            }
            checkedLinkCount += 1;
            const resolution = resolveLinkTarget(htmlFilePath, linkTarget);
            if (resolution.error !== undefined) {
                violations.push(`${htmlFilePath}: ${resolution.error}`);
                continue;
            }
            if (!fs.existsSync(resolution.resolvedPath)) {
                violations.push(`${htmlFilePath}: broken link target "${linkTarget}"`);
                continue;
            }
            if (fs.statSync(resolution.resolvedPath).isDirectory() &&
                !fs.existsSync(path.join(resolution.resolvedPath, "index.html"))) {
                violations.push(`${htmlFilePath}: directory target "${linkTarget}" has no index.html`);
            }
        }
    }

    return { htmlFiles, checkedLinkCount, violations };
}

function main() {
    console.log(`[check-site-links] status=start site_dir=site/`);
    const { htmlFiles, checkedLinkCount, violations } = checkSiteLinks();
    console.log(
        `[check-site-links] status=summary html_files=${htmlFiles.length} ` +
            `links_checked=${checkedLinkCount} violations=${violations.length}`,
    );
    for (const violation of violations) {
        console.error(`[check-site-links] violation: ${violation}`);
    }
    if (violations.length > 0) {
        process.exitCode = 1;
    }
}

if (require.main === module) {
    main();
}

module.exports = {
    extractLinkTargets,
    checkSiteLinks,
};
