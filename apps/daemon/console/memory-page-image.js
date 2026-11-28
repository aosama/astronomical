// The picture "Copy as image" makes, on Under the hood and on Home: one
// sentence a stranger understands ("A 22.9 GB model in an 8 GB limit"),
// three large figures, the last reply's token columns and a footer naming
// the model and the app, drawn on a canvas at a size made for a post. The
// whole-Mac bar and the wire limit stay on the page: in a post they needed
// explaining, and the people who would share it are not the ones who know
// what a wire limit is.

const MEMORY_PAGE_IMAGE_WIDTH_PIXELS = 1200;
const MEMORY_PAGE_IMAGE_HEIGHT_PIXELS = 630;
const MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS = 64;
const MEMORY_PAGE_IMAGE_FONT_FAMILY = "-apple-system, system-ui, sans-serif";
const MEMORY_PAGE_IMAGE_SITE = "getastronomical.com";

// The colours the page is drawn in, read from the page so the image follows
// the light or dark appearance the person is looking at.
function memoryPageImagePalette() {
    const pageStyle = getComputedStyle(document.documentElement);
    const colorOf = (tokenName) => pageStyle.getPropertyValue(tokenName).trim();
    return {
        background: colorOf("--panel"), ink: colorOf("--ink"), inkDim: colorOf("--ink-dim"),
        accent: colorOf("--accent"), memoryServed: colorOf("--resident"), storageRead: colorOf("--accent")
    };
}

// "a 16 GB limit" but "an 8 GB limit": the article follows how the number
// is spoken, and 8, 11, 18 and the eighties are the whole numbers a limit
// can take that begin with a vowel sound.
function memoryPageImageLimitPhrase(limitGigabytes) {
    const spokenWithVowel = /^8/.test(String(limitGigabytes)) || limitGigabytes === 11 || limitGigabytes === 18;
    return (spokenWithVowel ? "an " : "a ") + limitGigabytes + " GB limit";
}

// The sentence at the top. Without a model size or a limit it falls back to
// the page's own headline, so a picture is never made of empty words.
function memoryPageImageHeadline(statusDocument, pageHeadline) {
    const modelSizeBytes = statusDocument?.ready_model_size_bytes || 0;
    const limitGigabytes = memoryPageLimitGigabytes(statusDocument || {});
    if (!(modelSizeBytes > 0) || !(limitGigabytes > 0)) return pageHeadline;
    return "A " + formatMemoryPageGigabytes(modelSizeBytes) + " model in " + memoryPageImageLimitPhrase(limitGigabytes);
}

// The three large figures. A figure the app does not know yet reads "—".
function memoryPageImageFigures(statusDocument, expertStatsDocument) {
    const tokensPerSecond = statusDocument?.serving_session?.average_generation_tok_per_second || 0;
    const memoryServedPercent = memoryPageMemoryServedPercent(expertStatsDocument?.last_reply);
    return [
        { value: tokensPerSecond > 0 ? String(Math.round(tokensPerSecond)) : "—", label: "tokens a second", isAccent: true },
        { value: memoryServedPercent === null ? "—" : memoryServedPercent + "%", label: "of each token's experts already in memory", isAccent: false },
        { value: "0", label: "cloud accounts", isAccent: false }
    ];
}

// The footer's left side: the model, then the app and its version.
function memoryPageImageFooter(statusDocument) {
    const footerParts = [statusDocument?.ready_model_id || "No model loaded"];
    const applicationVersion = statusDocument?.application?.version;
    footerParts.push(applicationVersion ? "Astronomical " + applicationVersion : "Astronomical");
    return footerParts.join(" · ");
}

function drawMemoryPageImageFigures(drawingContext, figures, palette, figuresTop) {
    const figureWidth = (MEMORY_PAGE_IMAGE_WIDTH_PIXELS - 2 * MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS) / figures.length;
    figures.forEach((figure, figureIndex) => {
        const figureLeft = MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS + figureIndex * figureWidth;
        drawingContext.fillStyle = figure.isAccent ? palette.accent : palette.ink;
        drawingContext.font = "700 52px " + MEMORY_PAGE_IMAGE_FONT_FAMILY;
        drawingContext.fillText(figure.value, figureLeft, figuresTop);
        drawingContext.fillStyle = palette.inkDim;
        drawingContext.font = "21px " + MEMORY_PAGE_IMAGE_FONT_FAMILY;
        drawingContext.fillText(figure.label, figureLeft, figuresTop + 34, figureWidth - 24);
    });
}

// One column per token: the memory-served part on top, the share read from
// the SSD below it, to the same scale as the page's own strip.
function drawMemoryPageImageTokens(drawingContext, tokenPresentations, palette, stripTop, columnHeight) {
    const stripWidth = MEMORY_PAGE_IMAGE_WIDTH_PIXELS - 2 * MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS, columnGap = 2;
    // Columns narrow to fit a long reply on one row instead of wrapping it.
    const columnWidth = Math.min(10, (stripWidth - columnGap * tokenPresentations.length) / Math.max(1, tokenPresentations.length));
    tokenPresentations.forEach((token, tokenIndex) => {
        const columnLeft = MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS + tokenIndex * (columnWidth + columnGap);
        const storageHeight = (token.storageSharePercent / 100) * columnHeight;
        drawingContext.fillStyle = palette.memoryServed;
        drawingContext.fillRect(columnLeft, stripTop, columnWidth, columnHeight - storageHeight);
        drawingContext.fillStyle = palette.storageRead;
        drawingContext.fillRect(columnLeft, stripTop + columnHeight - storageHeight, columnWidth, storageHeight);
    });
}

// Draws the picture from the page's two documents.
function drawMemoryPageImage(imageCanvas, statusDocument, expertStatsDocument, palette) {
    const presentation = memoryPagePresentation(statusDocument, expertStatsDocument);
    imageCanvas.width = MEMORY_PAGE_IMAGE_WIDTH_PIXELS;
    imageCanvas.height = MEMORY_PAGE_IMAGE_HEIGHT_PIXELS;
    const drawingContext = imageCanvas.getContext("2d");
    drawingContext.fillStyle = palette.background;
    drawingContext.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
    drawingContext.textBaseline = "alphabetic";
    drawingContext.fillStyle = palette.ink;
    drawingContext.font = "600 24px " + MEMORY_PAGE_IMAGE_FONT_FAMILY;
    drawingContext.fillText("Astronomical", MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS, 82);
    drawingContext.font = "700 58px " + MEMORY_PAGE_IMAGE_FONT_FAMILY;
    drawingContext.fillText(memoryPageImageHeadline(statusDocument, presentation.headline),
        MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS, 168, MEMORY_PAGE_IMAGE_WIDTH_PIXELS - 2 * MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS);
    drawMemoryPageImageFigures(drawingContext, memoryPageImageFigures(statusDocument, expertStatsDocument), palette, 270);
    drawMemoryPageImageTokens(drawingContext, presentation.tokens, palette, 350, 150);
    drawingContext.fillStyle = palette.inkDim;
    drawingContext.font = "18px " + MEMORY_PAGE_IMAGE_FONT_FAMILY;
    drawingContext.fillText(memoryPageImageFooter(statusDocument), MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS, 580);
    drawingContext.fillStyle = palette.accent;
    drawingContext.font = "600 20px " + MEMORY_PAGE_IMAGE_FONT_FAMILY;
    drawingContext.textAlign = "right";
    drawingContext.fillText(MEMORY_PAGE_IMAGE_SITE, MEMORY_PAGE_IMAGE_WIDTH_PIXELS - MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS, 580);
    drawingContext.textAlign = "left";
}

// Copies the picture to the clipboard; where the browser will not allow
// that, saves it as a file so the picture is never lost. `feedbackElementId`
// names the line beside the button that was pressed.
async function copyMemoryPageImage(feedbackElementId) {
    const imageCanvas = document.createElement("canvas");
    drawMemoryPageImage(imageCanvas, memoryPageStatusDocument, memoryPageExpertStatsDocument, memoryPageImagePalette());
    const imageBlob = await new Promise((resolve) => imageCanvas.toBlob(resolve, "image/png"));
    const feedbackElement = document.getElementById(feedbackElementId);
    try {
        await navigator.clipboard.write([new ClipboardItem({ "image/png": imageBlob })]);
        if (feedbackElement) feedbackElement.textContent = "Copied. Paste it into a post.";
    } catch (clipboardError) {
        const downloadLink = document.createElement("a");
        downloadLink.href = URL.createObjectURL(imageBlob);
        downloadLink.download = "astronomical-memory.png";
        downloadLink.click();
        // WebKit can cancel a download whose address is revoked in the same
        // task as the click, so the address is released a moment later.
        setTimeout(() => URL.revokeObjectURL(downloadLink.href), 1000);
        if (feedbackElement) feedbackElement.textContent = "Saved as astronomical-memory.png.";
    }
}
