import Foundation

/**
 * Builds strict JSON object schemas in-process for `astronomical schema
 * object`, porting schema.rs. The document shape follows the fm convention:
 * `additionalProperties: false` and a `required` list at every object level
 * so downstream tools can never silently accept undeclared keys. Keys
 * serialize alphabetically, mirroring serde_json's sorted maps.
 */
public enum SchemaCommand {

    /// Builds the schema document as a plain JSON value tree.
    public static func buildSchemaDocument(
        _ schemaArguments: SchemaArguments
    ) -> Dictionary<String, Any> {
        let rootNode: SchemaObjectNode = SchemaObjectNode();
        for schemaProperty: SchemaPropertyInput in schemaArguments.properties {
            SchemaCommand.insertProperty(rootNode, schemaProperty: schemaProperty, segmentIndex: 0);
        }
        let rootValue: Any = SchemaNode.branch(rootNode).intoValue();
        var rootDocument: Dictionary<String, Any> = rootValue as? Dictionary<String, Any> ?? [:];
        rootDocument["title"] = schemaArguments.objectName;
        return rootDocument;
    }

    /// Renders the schema as one pretty-printed JSON document plus newline,
    /// mirroring `serde_json::to_string_pretty` (two-space indent, sorted keys).
    public static func run(
        _ schemaArguments: SchemaArguments,
        renderedOutput: TextOutputWriter
    ) -> Bool {
        let schemaDocument: Dictionary<String, Any> = SchemaCommand.buildSchemaDocument(schemaArguments);
        guard let renderedData: Data = try? JSONSerialization.data(
            withJSONObject: schemaDocument,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        ) else {
            return false;
        }
        var renderedText: String = String(decoding: renderedData, as: UTF8.self);
        renderedText += "\n";
        return renderedOutput.write(renderedText);
    }

    private static func insertProperty(
        _ objectNode: SchemaObjectNode,
        schemaProperty: SchemaPropertyInput,
        segmentIndex: Int
    ) -> Void {
        let pathSegments: Array<String> = schemaProperty.dottedPath.split(separator: ".").map(String.init);
        let segmentName: String = pathSegments[segmentIndex];
        let isFinalSegment: Bool = segmentIndex == pathSegments.count - 1;
        if (isFinalSegment) {
            if let existingChildIndex: Int = objectNode.childIndex(named: segmentName) {
                // Command-line parsing rejects duplicate and conflicting
                // paths, so this only matters for direct library callers: a
                // later leaf wins its position, and a leaf-over-branch clash
                // keeps the branch.
                if case .leaf = objectNode.childNodes[existingChildIndex].1 {
                    objectNode.childNodes[existingChildIndex].1 = .leaf(schemaProperty);
                }
                return;
            }
            objectNode.childNodes.append((segmentName, .leaf(schemaProperty)));
            return;
        }
        if objectNode.childIndex(named: segmentName) == nil {
            objectNode.childNodes.append((segmentName, .branch(SchemaObjectNode())));
        }
        guard let childIndex: Int = objectNode.childIndex(named: segmentName) else {
            return;
        }
        guard case var childObjectNode = objectNode.childNodes[childIndex].1, case .branch = childObjectNode else {
            return;
        }
        // The node is mutable so nested inserts can descend.
        SchemaCommand.insertProperty(&childObjectNode, schemaProperty: schemaProperty, segmentIndex: segmentIndex + 1);
        objectNode.childNodes[childIndex].1 = childObjectNode;
    }

    private static func insertProperty(
        _ objectNode: inout SchemaNode,
        schemaProperty: SchemaPropertyInput,
        segmentIndex: Int
    ) -> Void {
        guard case let .branch(nestedNode) = objectNode else {
            return;
        }
        // SchemaObjectNode is a reference type: nested inserts mutate through it.
        SchemaCommand.insertProperty(nestedNode, schemaProperty: schemaProperty, segmentIndex: segmentIndex);
    }
}

/// One nesting level of the schema tree, keeping insertion order.
final class SchemaObjectNode {

    var childNodes: Array<(String, SchemaNode)> = [];

    func childIndex(named segmentName: String) -> Int? {
        return self.childNodes.firstIndex { (childEntry: (String, SchemaNode)) -> Bool in
            return childEntry.0 == segmentName;
        }
    }
}

/// One node in the schema tree: a leaf property or a nested object.
indirect enum SchemaNode {

    case leaf(SchemaPropertyInput)
    case branch(SchemaObjectNode)

    var isOptional: Bool {
        switch (self) {
        case let .leaf(property):
            return property.isOptional
        case .branch:
            return false
        }
    }

    func intoValue() -> Any {
        switch (self) {
        case let .leaf(property):
            return SchemaNode.leafPropertyValue(property)
        case let .branch(branchNode):
            var propertiesDocument: Dictionary<String, Any> = [:];
            var requiredNames: Array<Any> = [];
            for (segmentName, childNode): (String, SchemaNode) in branchNode.childNodes {
                if !childNode.isOptional {
                    requiredNames.append(segmentName);
                }
                propertiesDocument[segmentName] = childNode.intoValue();
            }
            return [
                "type": "object",
                "properties": propertiesDocument,
                "required": requiredNames,
                "additionalProperties": false,
            ];
        }
    }

    private static func leafPropertyValue(_ schemaProperty: SchemaPropertyInput) -> Any {
        if (schemaProperty.isArray) {
            return [
                "type": "array",
                "items": ["type": schemaProperty.kind.jsonTypeName],
            ];
        }
        if let descriptionText: String = schemaProperty.description {
            return [
                "type": schemaProperty.kind.jsonTypeName,
                "description": descriptionText,
            ];
        }
        return ["type": schemaProperty.kind.jsonTypeName];
    }
}
