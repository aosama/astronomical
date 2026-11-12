/** The opening piece is closed once it reaches this many characters: about a
 * sentence, enough that "Sure!" alone never opens a reply. Kokoro speaks about
 * 40 times faster than real time on the Mac measured, so a sentence or two
 * comes back in a fraction of a second and the listener hears sound almost at
 * once. A one-word opening would be worse than none: the piece behind it takes
 * longer to make than "Sure!" takes to say, and the gap would be heard. */
export const OPENING_PIECE_MINIMUM_CHARACTERS = 80;

/** The second piece's ceiling. It is made while the opening plays, and the
 * opening can be only a few seconds long, so it stays small enough (about 25
 * seconds of audio, under a second to make) to arrive before the opening ends. */
export const SECOND_PIECE_MAXIMUM_CHARACTERS = 400;

/** Every later piece's ceiling: well under the daemon's 4,096-character limit
 * and about a minute of audio, a few megabytes of WAV, so a reply is held as
 * a handful of modest responses instead of one large one. Each is made while
 * the piece before it, a minute or so long, plays. */
export const PIECE_MAXIMUM_CHARACTERS = 1000;

// A sentence ends at . ! ? or … (with any closing quote or bracket) followed
// by space. Requiring the space keeps "1.5" and "example.com" whole.
const SENTENCE_BREAK_RE = /(?<=[.!?…]["')\]]*)\s+/u;

/**
 * Splits speakable text into the pieces fetched one after another: an opening
 * piece of at least OPENING_PIECE_MINIMUM_CHARACTERS (sentences are added
 * until it has them), a second of up to SECOND_PIECE_MAXIMUM_CHARACTERS, then
 * pieces of up to PIECE_MAXIMUM_CHARACTERS. Pieces break only at sentence
 * ends, so the voice never pauses mid-sentence between two pieces. A single
 * sentence longer than a piece is broken at its last comma or space instead.
 */
export function speechPiecesOf(speakableText: string): string[] {
  const sentences = speakableText
    .split(/\n{2,}/)
    .flatMap((paragraph) => sentencesOf(paragraph))
    .flatMap((sentence) => splitOversizedSentence(sentence, PIECE_MAXIMUM_CHARACTERS));
  const pieces: string[] = [];
  let currentPiece = "";
  for (const sentence of sentences) {
    const joinedPiece = currentPiece.length === 0 ? sentence : `${currentPiece} ${sentence}`;
    if (currentPiece.length > 0 && isPieceFull(pieces.length, currentPiece, joinedPiece)) {
      pieces.push(currentPiece);
      currentPiece = sentence;
    } else {
      currentPiece = joinedPiece;
    }
  }
  if (currentPiece.length > 0) {
    pieces.push(currentPiece);
  }
  return pieces;
}

/** Whether the piece being built should close before taking one more
 * sentence: the opening closes once long enough, the others once the next
 * sentence would pass their ceiling. */
function isPieceFull(closedPieceCount: number, currentPiece: string, joinedPiece: string): boolean {
  if (closedPieceCount === 0) {
    return currentPiece.length >= OPENING_PIECE_MINIMUM_CHARACTERS;
  }
  const ceilingCharacters = closedPieceCount === 1 ? SECOND_PIECE_MAXIMUM_CHARACTERS : PIECE_MAXIMUM_CHARACTERS;
  return joinedPiece.length > ceilingCharacters;
}

const SENTENCE_END_RE = /[.!?…:;]["')\]]*$/;

/** A paragraph's sentences, the last one ended with a full stop if it has
 * none: pieces are joined by spaces, so without it the voice would run one
 * paragraph into the next with no pause. */
function sentencesOf(paragraph: string): string[] {
  const sentences = paragraph.split(SENTENCE_BREAK_RE).map((sentence) => sentence.trim()).filter((sentence) => sentence.length > 0);
  const lastIndex = sentences.length - 1;
  const lastSentence = sentences[lastIndex];
  if (lastSentence !== undefined && !SENTENCE_END_RE.test(lastSentence)) {
    sentences[lastIndex] = `${lastSentence}.`;
  }
  return sentences;
}

function splitOversizedSentence(sentence: string, ceilingCharacters: number): string[] {
  const fragments: string[] = [];
  let remainingSentence = sentence;
  while (remainingSentence.length > ceilingCharacters) {
    const fittingStart = remainingSentence.slice(0, ceilingCharacters);
    const commaBreak = fittingStart.lastIndexOf(", ");
    const spaceBreak = fittingStart.lastIndexOf(" ");
    const breakIndex = commaBreak > ceilingCharacters / 2 ? commaBreak + 1 : spaceBreak > 0 ? spaceBreak : ceilingCharacters;
    fragments.push(remainingSentence.slice(0, breakIndex).trim());
    remainingSentence = remainingSentence.slice(breakIndex).trim();
  }
  if (remainingSentence.length > 0) {
    fragments.push(remainingSentence);
  }
  return fragments;
}
