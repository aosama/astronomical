/** How much sharper than the screen a diagram is drawn for Word, which shows
 * the picture at print size where screen resolution looks soft. */
const WORD_DIAGRAM_SCALE = 2;

/** A diagram drawn as a picture: its PNG data, and the size it was drawn
 * for in CSS pixels (the data itself is sharper than that). */
export interface DiagramPicture {
  readonly dataURL: string;
  readonly widthPixels: number;
  readonly heightPixels: number;
}

/** Draws an SVG diagram to a picture on a solid ground colour (a CSS
 * colour), or null when the page cannot. */
export type DiagramRasteriser = (diagram: SVGSVGElement, groundColour: string) => Promise<DiagramPicture | null>;

/** The largest a diagram is shown on an exported page, in CSS pixels: the
 * width of the text column, and a height that leaves room for the heading
 * above it. A tall flowchart at full width ran past the page, so Word moved
 * it to the next one and left the first half empty. */
export const MAXIMUM_DIAGRAM_WIDTH_PIXELS = 600;
export const MAXIMUM_DIAGRAM_HEIGHT_PIXELS = 460;

/** The size to show a diagram at: its own size, shrunk in proportion until
 * it fits both limits, and never enlarged. */
export function fittedDiagramSize(widthPixels: number, heightPixels: number): { width: number; height: number } {
  const shrinkFactor = Math.min(1, MAXIMUM_DIAGRAM_WIDTH_PIXELS / widthPixels, MAXIMUM_DIAGRAM_HEIGHT_PIXELS / heightPixels);
  return { width: Math.round(widthPixels * shrinkFactor), height: Math.round(heightPixels * shrinkFactor) };
}

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
    return { dataURL: canvas.toDataURL("image/png"), widthPixels: pixelSize.width, heightPixels: pixelSize.height };
  } catch {
    return null;
  }
};
