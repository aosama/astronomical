# Reading text aloud

`astronomical speak` turns text into speech, saves it as an audio file and
prints the path. It does not play anything. Its options (reading from a file,
the voice, the pace, where to save) are in `astronomical --help`.

## When the person wants to hear it

"Read this aloud" or "say this" means the person expects to hear it, so save
it, then play it: `afplay PATH` plays the saved file on a Mac. Only say it was
played after `afplay` finished without an error; otherwise say where the file
is so they can open it.

## Doing it well

- **Clean the text first.** Remove markup, code and links that would be read
  out letter by letter.
- **Split a long text** into parts of a few paragraphs, saved with numbered
  names. Tell the person they will get several files, to play in order.
- **Play parts in order,** one `afplay` per file, when a text was split.
- The first use downloads the speech model; ask first.
