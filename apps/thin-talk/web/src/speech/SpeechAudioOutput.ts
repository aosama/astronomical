/** The slice of the Web Audio API speech uses, so a test can stand in for it. */
export interface SpeechAudioContext {
  readonly currentTime: number;
  readonly state: string;
  readonly destination: unknown;
  resume(): Promise<void>;
  suspend(): Promise<void>;
  decodeAudioData(audioBytes: ArrayBuffer): Promise<SpeechAudioBuffer>;
  createBufferSource(): SpeechAudioSource;
}

export interface SpeechAudioBuffer {
  readonly duration: number;
}

export interface SpeechAudioSource {
  buffer: SpeechAudioBuffer | null;
  onended: (() => void) | null;
  connect(destination: unknown): unknown;
  start(when: number): void;
  stop(): void;
}

/** The page's Web Audio, which an `<audio>` element cannot replace here: the
 * page's content policy refuses every media source. Throws when the page has
 * no Web Audio, which the caller reports as a failure to speak. */
export function createWebAudioContext(): SpeechAudioContext {
  const audioContextConstructor =
    window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (audioContextConstructor === undefined) {
    throw new Error("this page has no Web Audio");
  }
  return new audioContextConstructor() as unknown as SpeechAudioContext;
}

/** How many decoded pieces are held at once: the one playing and the one
 * queued right behind it. Decoded samples are 32-bit floats, eight times the
 * size of the WAV they came from, so a long reply is decoded a piece ahead of
 * playback, never all at once. Counting pieces rather than seconds matters:
 * a piece lasts about a minute, so a seconds rule would leave the next one
 * undecoded when the current one ends, and the join would gap. The opening
 * piece is short, and the one after it is already decoded behind it, so the
 * first join is covered too. */
const DECODED_PIECES_HELD = 2;
/** The first piece starts this far after "now", so its first syllable is not
 * clipped while the output wakes up. */
const START_DELAY_SECONDS = 0.05;
/** How long the context gets to start running. It normally does so within
 * milliseconds, but an output device that has to wake (a Bluetooth speaker)
 * can take several seconds, and waiting only costs a cancellable "Preparing…";
 * WebKit leaves the request pending, rather than failing it, when it decides
 * the page may not make sound yet. */
const START_RUNNING_TIMEOUT_MILLISECONDS = 10_000;

/** What the output tells whoever asked it to play a reply. */
export interface SpeechAudioCallbacks {
  /** The first piece was scheduled on a running context: sound is starting. */
  onFirstSound: () => void;
  /** Every piece has played to its end. */
  onFinished: () => void;
  /** The output could not make sound at all. */
  onFailed: () => void;
}

/**
 * Plays a reply's spoken pieces back to back, without gaps, on one Web Audio
 * context.
 *
 * Web Audio, not an `<audio>` element: the page's content policy refuses
 * every media source, and decoding fetched bytes into buffers needs no media
 * URL at all. Scheduling each buffer at the exact end of the one before keeps
 * the joins silent, which separate elements started on `ended` events do not.
 *
 * WebKit lets a page start sound only from a click, so `wake()` must run
 * synchronously inside the click, before any await; the context it resumes
 * then stays allowed to play pieces that arrive later. Nothing is scheduled
 * until that context reports it is running, so the button never says the
 * reply is playing while the speaker stays silent.
 */
export class SpeechAudioOutput {
  private readonly createContext: () => SpeechAudioContext;
  private audioContext: SpeechAudioContext | null = null;
  private waitingPieces: ArrayBuffer[] = [];
  private scheduledSources = new Set<SpeechAudioSource>();
  private scheduledEndTime = 0;
  private isDecoding = false;
  private hasAllPieces = false;
  // Counts silences: a decode that began before one must not schedule after it.
  private playbackGeneration = 0;
  private isRunning: Promise<boolean> = Promise.resolve(false);
  private callbacks: SpeechAudioCallbacks = { onFirstSound: noop, onFinished: noop, onFailed: noop };

  public constructor(createContext: () => SpeechAudioContext) {
    this.createContext = createContext;
  }

  /** Starts a new reply: forgets anything queued and wakes the output. Call
   * from the click itself; throws when no audio context can be made. */
  public wake(callbacks: SpeechAudioCallbacks): void {
    this.silence();
    this.callbacks = callbacks;
    if (this.audioContext === null) {
      this.audioContext = this.createContext();
    }
    this.isRunning = SpeechAudioOutput.startRunning(this.audioContext);
  }

  /** Queues one piece's WAV bytes behind the ones before it. */
  public enqueue(audioBytes: ArrayBuffer): void {
    this.waitingPieces.push(audioBytes);
    void this.scheduleWaitingPieces();
  }

  /** Says no more pieces are coming, so the reply ends when the queue drains. */
  public markComplete(): void {
    this.hasAllPieces = true;
    this.finishIfDrained();
  }

  /** Stops the sound at once and lets the output sleep. */
  public silence(): void {
    this.playbackGeneration += 1;
    this.waitingPieces = [];
    this.hasAllPieces = false;
    this.isDecoding = false;
    for (const scheduledSource of this.scheduledSources) {
      scheduledSource.onended = null;
      scheduledSource.stop();
    }
    this.scheduledSources.clear();
    this.scheduledEndTime = 0;
    if (this.audioContext !== null) {
      // Suspended whatever the state reads: a resume still pending when Stop
      // is pressed would otherwise leave a silent context running, and a
      // running context keeps holding the speaker.
      this.audioContext.suspend().catch(noop);
    }
  }

  /** True once the context runs; false when it refused or never answered. */
  private static startRunning(audioContext: SpeechAudioContext): Promise<boolean> {
    return new Promise<boolean>((settle) => {
      const timeout = setTimeout(() => settle(false), START_RUNNING_TIMEOUT_MILLISECONDS);
      audioContext.resume().then(
        () => {
          clearTimeout(timeout);
          settle(true);
        },
        () => {
          clearTimeout(timeout);
          settle(false);
        },
      );
    });
  }

  private async scheduleWaitingPieces(): Promise<void> {
    const audioContext = this.audioContext;
    if (audioContext === null || this.isDecoding || this.waitingPieces.length === 0) {
      return;
    }
    const scheduledGeneration = this.playbackGeneration;
    this.isDecoding = true;
    const canMakeSound = await this.isRunning;
    if (scheduledGeneration !== this.playbackGeneration) {
      // Silenced while waiting: this call is stale, and the pieces now
      // waiting belong to a later reading it must not take from.
      return;
    }
    if (!canMakeSound) {
      const onFailed = this.callbacks.onFailed;
      this.silence();
      onFailed();
      return;
    }
    while (this.waitingPieces.length > 0 && this.scheduledSources.size < DECODED_PIECES_HELD) {
      const pieceBytes = this.waitingPieces.shift() as ArrayBuffer;
      const decodedPiece = await SpeechAudioOutput.decodedOrNull(audioContext, pieceBytes);
      if (scheduledGeneration !== this.playbackGeneration) {
        return;
      }
      if (decodedPiece !== null) {
        this.startAtQueueEnd(audioContext, decodedPiece);
      }
    }
    this.isDecoding = false;
    this.finishIfDrained();
  }

  /** One undecodable piece is skipped rather than ending the reply: the rest
   * of the answer is still worth hearing. */
  private static async decodedOrNull(audioContext: SpeechAudioContext, pieceBytes: ArrayBuffer): Promise<SpeechAudioBuffer | null> {
    try {
      return await audioContext.decodeAudioData(pieceBytes);
    } catch {
      return null;
    }
  }

  private startAtQueueEnd(audioContext: SpeechAudioContext, decodedPiece: SpeechAudioBuffer): void {
    const isFirstPiece = this.scheduledSources.size === 0 && this.scheduledEndTime === 0;
    const startTime = Math.max(audioContext.currentTime + START_DELAY_SECONDS, this.scheduledEndTime);
    const pieceSource = audioContext.createBufferSource();
    pieceSource.buffer = decodedPiece;
    pieceSource.connect(audioContext.destination);
    pieceSource.onended = () => this.pieceEnded(pieceSource);
    pieceSource.start(startTime);
    this.scheduledSources.add(pieceSource);
    this.scheduledEndTime = startTime + decodedPiece.duration;
    if (isFirstPiece) {
      this.callbacks.onFirstSound();
    }
  }

  private pieceEnded(pieceSource: SpeechAudioSource): void {
    this.scheduledSources.delete(pieceSource);
    void this.scheduleWaitingPieces();
    this.finishIfDrained();
  }

  private finishIfDrained(): void {
    const isDrained = this.hasAllPieces && !this.isDecoding && this.waitingPieces.length === 0 && this.scheduledSources.size === 0;
    if (!isDrained) {
      return;
    }
    const onFinished = this.callbacks.onFinished;
    this.silence();
    onFinished();
  }
}

function noop(): void {
  return undefined;
}
