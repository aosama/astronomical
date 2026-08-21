// Rejects additive wire drift for documents whose complete value participates in correctness
// decisions. Ordinary status DTOs remain forward-tolerant, but worker policy equality is only
// meaningful when Swift has represented every field emitted by the bundled Rust producer.

private struct WireCodingKey: CodingKey, Hashable {
  let stringValue: String
  let intValue: Int?

  init?(stringValue: String) {
    self.stringValue = stringValue
    intValue = nil
  }

  init?(intValue: Int) {
    stringValue = String(intValue)
    self.intValue = intValue
  }
}

extension Decoder {
  func rejectUnknownKeys<Key>(_ codingKeyType: Key.Type) throws
  where Key: CodingKey & CaseIterable {
    let wireContainer = try container(keyedBy: WireCodingKey.self)
    let allowedKeyNames = Set(codingKeyType.allCases.map(\.stringValue))
    guard let unknownCodingKey = wireContainer.allKeys.first(where: {
      !allowedKeyNames.contains($0.stringValue)
    }) else { return }

    throw DecodingError.dataCorrupted(
      DecodingError.Context(
        codingPath: codingPath + [unknownCodingKey],
        debugDescription: "Unexpected worker runtime policy field '\(unknownCodingKey.stringValue)'"
      ))
  }
}

extension KeyedDecodingContainer {
  func decodeRequiredNullable<Value>(_ type: Value.Type, forKey key: Key) throws -> Value?
  where Value: Decodable {
    guard contains(key) else {
      throw DecodingError.keyNotFound(
        key,
        DecodingError.Context(
          codingPath: codingPath,
          debugDescription: "Required nullable worker runtime policy field is missing"
        ))
    }
    return try decodeIfPresent(type, forKey: key)
  }

  func decodeValueOrOmission<Value>(_ type: Value.Type, forKey key: Key) throws -> Value?
  where Value: Decodable {
    guard contains(key) else { return nil }
    return try decode(type, forKey: key)
  }
}
