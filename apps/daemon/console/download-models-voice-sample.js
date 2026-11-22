// Astronomical Observatory Models page: a short spoken sample of a read-aloud
// model, played when its speaker icon under On this Mac is clicked, so a person
// can hear a voice before using it. One sample plays at a time; clicking the
// playing model's icon again stops it.

const DOWNLOAD_MODELS_SPEECH_URL = "/v1/audio/speech";
const DOWNLOAD_MODELS_VOICE_SAMPLE_TEXT = "Hello. This is how I sound when I read aloud.";

// Which model's sample is loading or playing, and in which phase, or null.
// The first click loads the speech model, which takes a second or two, so the
// icon shows that it is working before any sound comes.
let downloadModelsVoiceSample = null;
let downloadModelsVoiceSampleAudio = null;
// The object URL holding the fetched speech, freed whenever the sample stops,
// so repeated samples do not keep their audio in memory.
let downloadModelsVoiceSampleUrl = null;

// A two-sample silent WAV. Safari lets an audio element play later only if it
// was first played inside the click; the speech arrives a second or two after
// the click, so the element is unlocked with this silence and reused.
const DOWNLOAD_MODELS_SILENT_AUDIO_URL = "data:audio/wav;base64,UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQIAAACAgA==";

function downloadModelsVoiceSamplePhase(huggingfaceId) {
    return downloadModelsVoiceSample?.huggingfaceId === huggingfaceId ? downloadModelsVoiceSample.phase : null;
}

function setDownloadModelsVoiceSample(nextVoiceSample) {
    downloadModelsVoiceSample = nextVoiceSample;
    renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
}

function stopDownloadModelsVoiceSample() {
    if (downloadModelsVoiceSampleAudio) {
        downloadModelsVoiceSampleAudio.pause();
        downloadModelsVoiceSampleAudio = null;
    }
    if (downloadModelsVoiceSampleUrl) {
        URL.revokeObjectURL(downloadModelsVoiceSampleUrl);
        downloadModelsVoiceSampleUrl = null;
    }
    setDownloadModelsVoiceSample(null);
}

// Runs inside the click, before any await, so the element counts as started
// by the person. A refused unlock is ignored: the later play reports failure.
function unlockDownloadModelsVoiceSampleAudio() {
    const sampleAudio = new Audio(DOWNLOAD_MODELS_SILENT_AUDIO_URL);
    sampleAudio.play()?.catch?.(() => {});
    return sampleAudio;
}

// Plays the sample, or stops it when this model's sample is already loading
// or playing. A failure leaves a short note in the icon's tooltip rather than
// an error on the page: the sample is a convenience.
async function toggleDownloadModelsVoiceSample(installedRow) {
    const wasThisModel = downloadModelsVoiceSamplePhase(installedRow.huggingfaceId) !== null;
    stopDownloadModelsVoiceSample();
    if (wasThisModel) return;
    let sampleAudio = null;
    try {
        sampleAudio = unlockDownloadModelsVoiceSampleAudio();
        downloadModelsVoiceSampleAudio = sampleAudio;
        setDownloadModelsVoiceSample({ huggingfaceId: installedRow.huggingfaceId, phase: "loading" });
        const sampleUrl = await fetchDownloadModelsVoiceSampleUrl(downloadModelsRowName(installedRow));
        // A stop or another model's sample while the speech was on its way
        // replaced this element; its speech is no longer wanted.
        if (downloadModelsVoiceSampleAudio !== sampleAudio) {
            URL.revokeObjectURL(sampleUrl);
            return;
        }
        downloadModelsVoiceSampleUrl = sampleUrl;
        sampleAudio.addEventListener("ended", stopDownloadModelsVoiceSample);
        sampleAudio.src = sampleUrl;
        await sampleAudio.play();
        setDownloadModelsVoiceSample({ huggingfaceId: installedRow.huggingfaceId, phase: "playing" });
    } catch {
        // A sample stopped or replaced meanwhile is not this click's failure.
        if (downloadModelsVoiceSampleAudio !== sampleAudio) return;
        setDownloadModelsVoiceSample({ huggingfaceId: installedRow.huggingfaceId, phase: "failed" });
    }
}

async function fetchDownloadModelsVoiceSampleUrl(modelId) {
    const speechResponse = await fetch(DOWNLOAD_MODELS_SPEECH_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelId, input: DOWNLOAD_MODELS_VOICE_SAMPLE_TEXT })
    });
    if (!speechResponse.ok) throw new Error("the engine did not read the sample aloud");
    return URL.createObjectURL(await speechResponse.blob());
}

function downloadModelsVoiceSampleTooltip(huggingfaceId) {
    const phase = downloadModelsVoiceSamplePhase(huggingfaceId);
    if (phase === "loading") return "Getting the voice ready…";
    if (phase === "playing") return "Stop the sample";
    if (phase === "failed") return "The sample could not play. Try again.";
    return "Hear a sample";
}
