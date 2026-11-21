# Reading text aloud

`astronomical speak` turns text into speech, saves it as an audio file and
prints the path. Its options (reading from a file, the voice, the pace, where
to save) are in `astronomical --help`.

## Doing it well

- **Clean the text first.** Remove markup, code and links that would be read
  out letter by letter.
- **Split a long text** into parts of a few paragraphs, saved with numbered
  names. Tell the person they will get several files, to play in order.
- **To play it now,** `afplay PATH` plays a saved file on a Mac.
- The first use downloads the speech model; ask first.
