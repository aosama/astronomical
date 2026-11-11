# Misaki American English lexicon

`us_gold.json` and `us_silver.json` are copied unchanged from the Misaki
grapheme-to-phoneme project, `misaki/data/` at commit
`fba1236595f2d2bf21d414ba6e57d25256afada3`
(<https://github.com/hexgrad/misaki>), under the Apache License 2.0 in
`MISAKI_LICENSE`.

Kokoro-82M was trained on phonemes from this lexicon, so speech sounds right
only when the same entries spell each word. "Gold" entries were checked by
hand; "silver" entries fill gaps and are consulted after gold.

| File | SHA-256 |
| --- | --- |
| `us_gold.json` | `dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f` |
| `us_silver.json` | `de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475` |

To update, copy both files from a newer Misaki commit, record the commit and
digests here, and run `cargo test -p astronomical-english-phonemes`.
