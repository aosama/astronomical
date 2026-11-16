// The Memory page's shareable picture: the headline, the Mac's memory bar
// with the wire limit, the four facts and the token columns, drawn on a
// canvas at a size made for a post, then copied or saved.

const MEMORY_PAGE_IMAGE_WIDTH_PIXELS = 1200;
const MEMORY_PAGE_IMAGE_HEIGHT_PIXELS = 675;
const MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS = 64;

// The colours the page is drawn in, read from the page so the image follows
// the light or dark appearance the person is looking at.
function memoryPageImagePalette() {
    const pageStyle = getComputedStyle(document.documentElement);
    const colorOf = (tokenName) => pageStyle.getPropertyValue(tokenName).trim();
    return {
        background: colorOf("--panel"), ink: colorOf("--ink"), inkDim: colorOf("--ink-dim"), track: colorOf("--panel-2"),
        parts: {
            other: colorOf("--ink-faint"), fixed: colorOf("--ink-dim"), experts: colorOf("--resident"),
            conversation: colorOf("--memory-context"), working: colorOf("--accent")
        },
        memoryServed: colorOf("--resident"), storageRead: colorOf("--accent"),
        wireLimit: colorOf("--danger") || colorOf("--accent")
    };
}

// Where each legend entry of the picture goes: entries run left to right
// and start a new line when the next would cross the right margin.
function memoryPageImageLegendPlacements(legendTextWidths, legendTop) {
    const legendLeftEdge = MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS;
    const legendRightEdge = MEMORY_PAGE_IMAGE_WIDTH_PIXELS - MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS;
    let legendLeft = legendLeftEdge;
    let legendLineTop = legendTop;
    return legendTextWidths.map((legendTextWidth) => {
        const legendEntryWidth = 24 + legendTextWidth;
        if (legendLeft > legendLeftEdge && legendLeft + legendEntryWidth > legendRightEdge) {
            legendLeft = legendLeftEdge;
            legendLineTop += 30;
        }
        const legendPlacement = { left: legendLeft, top: legendLineTop };
        legendLeft += legendEntryWidth + 28;
        return legendPlacement;
    });
}

function drawMemoryPageImageBar(drawingContext, barPresentation, palette, barTop) {
    const barLeft = MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS, barHeight = 56;
    const barWidth = MEMORY_PAGE_IMAGE_WIDTH_PIXELS - 2 * MEMORY_PAGE_IMAGE_SIDE_MARGIN_PIXELS;
    drawingContext.fillStyle = palette.track;
    drawingContext.fillRect(barLeft, barTop, barWidth, barHeight);
    let partLeft = barLeft;
    for (const barPart of barPresentation.parts) {
        const partWidth = (barPart.widthPercent / 100) * barWidth;
        drawingContext.fillStyle = palette.parts[barPart.key];
        drawingContext.fillRect(partLeft, barTop, partWidth, barHeight);
        partLeft += partWidth;
    }
    // Near-empty parts stay on the bar but leave the legend, where they
    // would only read "0.0 GB".
    const legendParts = barPresentation.parts.filter((barPart) => !barPart.isEmpty);
    drawingContext.font = "22px -apple-system, system-ui, sans-serif";
    const legendTexts = legendParts.map((barPart) => barPart.label + " " + barPart.text);
    const legendPlacements = memoryPageImageLegendPlacements(
        legendTexts.map((legendText) => drawingContext.measureText(legendText).width), barTop + barHeight + 24);
    legendParts.forEach((barPart, legendIndex) => {
        const legendPlacement = legendPlacements[legendIndex];
        drawingContext.fillStyle = palette.parts[barPart.key];
        drawingContext.fillRect(legendPlacement.left, legendPlacement.top, 16, 16);
        drawingContext.fillStyle = palette.inkDim;
        drawingContext.fillText(legendTexts[legendIndex], legendPlacement.left + 24, legendPlacement.top + 15);
    });
}

function drawMemoryPageImageWireLimit(drawingContext, barPresentation, palette, barTop) {
    if (barPresentation.wireLimitPercent === null) return;
    const barLeft = 64, barWidth = MEMORY_PAGE_IMAGE_WIDTH_PIXELS - 128;
    const markerLeft = barLeft + (barPresentation.wireLimitPercent / 100) * barWidth;
    drawingContext.fillStyle = palette.wireLimit;
    drawingContext.fillRect(markerLeft - 1.5, barTop - 34, 3, 100);
    drawingContext.font = "22px -apple-system, system-ui, sans-serif";
    drawingContext.textAlign = "right";
    drawingContext.fillText(barPresentation.wireLimitText, markerLeft - 12, barTop - 14);
    drawingContext.textAlign = "left";
}

function drawMemoryPageImageFacts(drawingContext, facts, palette, factsTop) {
    const factWidth = (MEMORY_PAGE_IMAGE_WIDTH_PIXELS - 128) / facts.length;
    facts.forEach((fact, factIndex) => {
        const factLeft = 64 + factIndex * factWidth;
        drawingContext.fillStyle = palette.inkDim;
        drawingContext.font = "22px -apple-system, system-ui, sans-serif";
        drawingContext.fillText(fact.label, factLeft, factsTop);
        drawingContext.fillStyle = palette.ink;
        // A long value such as "27 of 384 experts" shrinks to its column
        // rather than running into the next fact.
        let valueFontPixels = 40;
        do {
            drawingContext.font = "600 " + valueFontPixels + "px -apple-system, system-ui, sans-serif";
            valueFontPixels -= 2;
        } while (valueFontPixels > 20 && drawingContext.measureText(fact.value).width > factWidth - 24);
        drawingContext.fillText(fact.value, factLeft, factsTop + 48);
    });
}

function drawMemoryPageImageTokens(drawingContext, tokenPresentations, palette, stripTop) {
    const stripWidth = MEMORY_PAGE_IMAGE_WIDTH_PIXELS - 128, columnHeight = 80, columnGap = 2;
    // Columns narrow to fit a long reply on one row instead of wrapping it.
    const columnWidth = Math.min(10, (stripWidth - columnGap * tokenPresentations.length) / Math.max(1, tokenPresentations.length));
    tokenPresentations.forEach((token, tokenIndex) => {
        const columnLeft = 64 + tokenIndex * (columnWidth + columnGap);
        const storageHeight = (token.storageSharePercent / 100) * columnHeight;
        drawingContext.fillStyle = palette.memoryServed;
        drawingContext.fillRect(columnLeft, stripTop, columnWidth, columnHeight - storageHeight);
        drawingContext.fillStyle = palette.storageRead;
        drawingContext.fillRect(columnLeft, stripTop + columnHeight - storageHeight, columnWidth, storageHeight);
    });
}

// Draws the page's facts as one picture sized for a post.
function drawMemoryPageImage(imageCanvas, presentation, palette) {
    imageCanvas.width = MEMORY_PAGE_IMAGE_WIDTH_PIXELS;
    imageCanvas.height = MEMORY_PAGE_IMAGE_HEIGHT_PIXELS;
    const drawingContext = imageCanvas.getContext("2d");
    drawingContext.fillStyle = palette.background;
    drawingContext.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
    drawingContext.textBaseline = "alphabetic";
    drawingContext.fillStyle = palette.inkDim;
    drawingContext.font = "24px -apple-system, system-ui, sans-serif";
    drawingContext.fillText(presentation.modelLine + " · Astronomical", 64, 74);
    drawingContext.fillStyle = palette.ink;
    drawingContext.font = "600 46px -apple-system, system-ui, sans-serif";
    drawingContext.fillText(presentation.headline, 64, 134);
    if (presentation.bar) {
        drawMemoryPageImageBar(drawingContext, presentation.bar, palette, 228);
        drawMemoryPageImageWireLimit(drawingContext, presentation.bar, palette, 228);
    }
    drawMemoryPageImageFacts(drawingContext, presentation.facts, palette, 430);
    drawingContext.fillStyle = palette.inkDim;
    drawingContext.font = "22px -apple-system, system-ui, sans-serif";
    drawingContext.fillText(presentation.tokenNote, 64, 540);
    drawMemoryPageImageTokens(drawingContext, presentation.tokens, palette, 562);
}

function setMemoryPageCopyFeedback(feedbackText) {
    document.getElementById("memory-page-copy-feedback").textContent = feedbackText;
}

// Copies the picture to the clipboard; where the browser will not allow
// that, saves it as a file so the picture is never lost.
async function copyMemoryPageImage() {
    const imageCanvas = document.createElement("canvas");
    drawMemoryPageImage(imageCanvas,
        memoryPagePresentation(memoryPageStatusDocument, memoryPageExpertStatsDocument), memoryPageImagePalette());
    const imageBlob = await new Promise((resolve) => imageCanvas.toBlob(resolve, "image/png"));
    try {
        await navigator.clipboard.write([new ClipboardItem({ "image/png": imageBlob })]);
        setMemoryPageCopyFeedback("Copied. Paste it into a post.");
    } catch (clipboardError) {
        const downloadLink = document.createElement("a");
        downloadLink.href = URL.createObjectURL(imageBlob);
        downloadLink.download = "astronomical-memory.png";
        downloadLink.click();
        // WebKit can cancel a download whose address is revoked in the same
        // task as the click, so the address is released a moment later.
        setTimeout(() => URL.revokeObjectURL(downloadLink.href), 1000);
        setMemoryPageCopyFeedback("Saved as astronomical-memory.png.");
    }
}
