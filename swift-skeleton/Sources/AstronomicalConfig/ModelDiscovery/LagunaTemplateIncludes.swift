import Foundation;
import Darwin;

/**
 * Chat-template include resolution for Laguna discovery: walks the static
 * single-quoted `{% include 'name' %}` graph reachable from the selected root
 * template, bounding depth, fan-out, and total bytes before any file is
 * buffered. Split from Laguna.swift because the walk is its own job.
 */
internal enum LagunaTemplateIncludes {
    internal static let MAXIMUM_TEMPLATE_BYTES: UInt64 = 512 * 1024;
    private static let MAXIMUM_TEMPLATE_SOURCE_COUNT: Int = 16;
    private static let MAXIMUM_TEMPLATE_INCLUDE_DEPTH: Int = 8;

    /**
     * Returns the set of include file names the root template transitively
     * pulls in, or nil when any bound is exceeded, a cycle exists, an include
     * name is unsafe, or an included file cannot be read at its bound.
     */
    internal static func validateTemplateSources(
        modelDirectory: FilePath,
        rootTemplateSource: String,
        standaloneRootFileName: String?
    ) -> Set<String>? {
        if (rootTemplateSource.isEmpty || UInt64(rootTemplateSource.utf8.count) > self.MAXIMUM_TEMPLATE_BYTES) {
            return nil;
        }
        guard let rootIncludeNames: Array<String> = self.templateIncludeNames(templateSource: rootTemplateSource)
        else {
            return nil;
        }
        var pendingIncludes: Array<PendingTemplateInclude> = Array<PendingTemplateInclude>();
        for rootIncludeName: String in rootIncludeNames {
            var rootAncestorNames: Array<String> = Array<String>();
            if let standaloneFileName: String = standaloneRootFileName {
                rootAncestorNames.append(standaloneFileName);
            }
            pendingIncludes.append(PendingTemplateInclude(
                includeName: rootIncludeName,
                includeDepth: 1,
                ancestorNames: rootAncestorNames
            ));
        }
        var includedNames: Set<String> = Set<String>();
        var totalTemplateBytes: UInt64 = UInt64(rootTemplateSource.utf8.count);
        while (pendingIncludes.isEmpty == false) {
            let pendingInclude: PendingTemplateInclude = pendingIncludes.removeLast();
            if (pendingInclude.includeDepth > self.MAXIMUM_TEMPLATE_INCLUDE_DEPTH
                || pendingInclude.ancestorNames.contains(pendingInclude.includeName)
                || self.isSafeTemplateIncludeName(includeName: pendingInclude.includeName) == false)
            {
                return nil;
            }
            if (includedNames.contains(pendingInclude.includeName)) {
                continue;
            }
            includedNames.insert(pendingInclude.includeName);
            if (includedNames.count + 1 > self.MAXIMUM_TEMPLATE_SOURCE_COUNT) {
                return nil;
            }
            guard let includeBytes: Data = Laguna.readBoundedFile(
                filePath: modelDirectory.appending(component: pendingInclude.includeName),
                maximumBytes: self.MAXIMUM_TEMPLATE_BYTES
            ) else {
                return nil;
            }
            let (partialValue: combinedTemplateBytes, overflow: didOverflowTemplateBytes) = totalTemplateBytes
                .addingReportingOverflow(UInt64(includeBytes.count));
            if (didOverflowTemplateBytes) {
                return nil;
            }
            totalTemplateBytes = combinedTemplateBytes;
            if (totalTemplateBytes > self.MAXIMUM_TEMPLATE_BYTES) {
                return nil;
            }
            guard let includeSource: String = String(bytes: includeBytes, encoding: .utf8) else {
                return nil;
            }
            guard let childIncludeNames: Array<String> = self.templateIncludeNames(templateSource: includeSource)
            else {
                return nil;
            }
            var childAncestorNames: Array<String> = pendingInclude.ancestorNames;
            childAncestorNames.append(pendingInclude.includeName);
            for childIncludeName: String in childIncludeNames {
                pendingIncludes.append(PendingTemplateInclude(
                    includeName: childIncludeName,
                    includeDepth: pendingInclude.includeDepth + 1,
                    ancestorNames: childAncestorNames
                ));
            }
        }
        return includedNames;
    }

    /**
     * One queued include to resolve, carrying the depth and the include chain
     * that selected it so cycles fail deterministically.
     */
    private struct PendingTemplateInclude {
        let includeName: String;
        let includeDepth: Int;
        let ancestorNames: Array<String>;
    }

    /**
     * Finds the static single-quoted include syntax accepted by Laguna
     * startup.
     */
    private static func templateIncludeNames(templateSource: String) -> Array<String>? {
        var includeNames: Array<String> = Array<String>();
        var remainingSource: Substring = Substring(templateSource);
        while true {
            guard let directiveStartRange: Range<String.Index> = remainingSource.range(of: "{%") else {
                break;
            }
            remainingSource = remainingSource[remainingSource.index(directiveStartRange.lowerBound, offsetBy: 2)...];
            guard let directiveEndRange: Range<String.Index> = remainingSource.range(of: "%}") else {
                return nil;
            }
            var directiveBodyText: String = String(remainingSource[..<directiveEndRange.lowerBound])
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines);
            while (directiveBodyText.hasPrefix("-")) {
                directiveBodyText.removeFirst();
            }
            while (directiveBodyText.hasSuffix("-")) {
                directiveBodyText.removeLast();
            }
            directiveBodyText = directiveBodyText.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines);
            if (directiveBodyText.hasPrefix("include")) {
                var includeExpressionText: String = String(directiveBodyText.dropFirst("include".count))
                    .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines);
                if (includeExpressionText.hasPrefix("'") == false || includeExpressionText.hasSuffix("'") == false) {
                    return nil;
                }
                includeExpressionText.removeFirst();
                includeExpressionText.removeLast();
                if (includeExpressionText.isEmpty || includeExpressionText.contains("'")) {
                    return nil;
                }
                includeNames.append(includeExpressionText);
            }
            remainingSource = remainingSource[remainingSource.index(directiveEndRange.lowerBound, offsetBy: 2)...];
        }
        return includeNames;
    }

    private static func isSafeTemplateIncludeName(includeName: String) -> Bool {
        if (includeName.utf8.count > 255 || includeName.contains("\\")) {
            return false;
        }
        let includePath: FilePath = FilePath(string: includeName);
        if (includePath.isAbsolute) {
            return false;
        }
        let includePathSegments: Array<Substring> = includeName.split(
            omittingEmptySubsequences: true,
            whereSeparator: { (separatorCharacter: Character) in return separatorCharacter == "/"; }
        );
        if (includePathSegments.first == ".") {
            return false;
        }
        for includePathSegment: Substring in includePathSegments {
            if (includePathSegment == "..") {
                return false;
            }
        }
        return true;
    }
}
