# Reading text aloud

`astronomical speak` reads text aloud on this Mac's speakers and returns when
it has finished. With the output option it saves the audio to a file instead
and plays nothing. Its options (reading from a file, the voice, the pace,
saving) are in `astronomical --help`.

## Choosing between them

- "Read this aloud", "say this", "tell me": speak it. Nothing is saved.
- "Make an audio file of this", "save it for later": use the output option and
  tell the person the path the command printed.

## Doing it well

- **Clean the text first.** Remove markup, code and links that would be read
  out letter by letter.
- **Split a long text** into parts of a few paragraphs, one command per part,
  in order. Each one returns when it has finished speaking.
- **If playback fails** (for example, no working sound output), the message
  says so; offer to save the audio as a file instead.
- The first use downloads the speech model; ask first.
