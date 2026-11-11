/** How much sharper than the screen a diagram is drawn for Word, which shows
 * the picture at print size where screen resolution looks soft. */
const WORD_DIAGRAM_SCALE = 2;

/** Draws an SVG diagram to PNG data on a solid ground colour (a CSS colour),
 * or null when the page cannot. */
export type DiagramRasteriser = (diagram: SVGSVGElement, groundColour: string) => Promise<string | null>;

function diagramPixelSize(diagram: SVGSVGElement): { width: number; height: number } | null {
  const drawnBox = diagram.getBoundingClientRect();
  const viewBox = diagram.viewBox?.baseVal;
  const width = drawnBox.width > 0 ? drawnBox.width : (viewBox?.width ?? 0);
  const height = drawnBox.height > 0 ? drawnBox.height : (viewBox?.height ?? 0);
  return width > 0 && height > 0 ? { width, height } : null;
}

/**
 * Draws a diagram through a canvas. Word cannot show an SVG without a PNG
 * beside it, and older versions cannot show one at all, so a Word document
 * gets the PNG alone. Diagrams here are drawn with plain SVG text (the
 * diagram pipeline turns HTML labels off), which is what lets a canvas read
 * them back: a drawing with embedded HTML would taint it.
 */
export const rasteriseDiagramOnCanvas: DiagramRasteriser = async (diagram, groundColour) => {
  const pixelSize = diagramPixelSize(diagram);
  if (pixelSize === null) {
    return null;
  }
  const sizedDiagram = diagram.cloneNode(true) as SVGSVGElement;
  sizedDiagram.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  sizedDiagram.setAttribute("width", String(pixelSize.width));
  sizedDiagram.setAttribute("height", String(pixelSize.height));
  const diagramSource = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(sizedDiagram))}`;
  const loadedDiagram = new Image();
  try {
    await new Promise<void>((resolve, reject) => {
      loadedDiagram.onload = () => resolve();
      loadedDiagram.onerror = () => reject(new Error("the diagram did not load as an image"));
      loadedDiagram.src = diagramSource;
    });
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(pixelSize.width * WORD_DIAGRAM_SCALE);
    canvas.height = Math.round(pixelSize.height * WORD_DIAGRAM_SCALE);
    const drawingContext = canvas.getContext("2d");
    if (drawingContext === null) {
      return null;
    }
    // A diagram's own ground is transparent; on a page of another colour its
    // lines could vanish, so the ground it was drawn for is painted first.
    drawingContext.fillStyle = groundColour;
    drawingContext.fillRect(0, 0, canvas.width, canvas.height);
    drawingContext.drawImage(loadedDiagram, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png");
  } catch {
    return null;
  }
};
