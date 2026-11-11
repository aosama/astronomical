/**
 * Astro drawn as a small constellation: six stars and the seven lines that
 * join them. It stands wherever Astro used to be drawn as a character (the
 * home screen, the welcome, Your Astro and beside each answer). The message
 * field has none: it is where people type.
 *
 * The character read as made for small children to older teenagers, and a
 * face kept promising feelings the app could not back up. A star figure says
 * only "Astro is here" when still and "Astro is working" while its lines draw
 * in turn, and it suits an app named after the sky.
 *
 * The lines take the colours of the person's chosen look (Rainbow, Aurora or
 * Sunset, the `--astro-trail-*` colours), so that choice still shows. Motion
 * comes from the stylesheet, keyed on the figure's state, so it costs no timer
 * and holds still under Reduce Motion. Every coordinate is fixed text, never
 * conversation content, so the markup needs no escaping.
 */

/** Star positions on a 24 by 24 grid: a loose ring around one centre star. */
const STARS: readonly (readonly [number, number])[] = [
  [12, 3.5],
  [20, 8.5],
  [17.5, 19],
  [7, 20],
  [3.5, 10],
  [12, 12.5],
];

/** Which stars each line joins, in the order the lines draw while working. */
const LINKS: readonly (readonly [number, number])[] = [
  [0, 5],
  [5, 1],
  [1, 2],
  [2, 5],
  [5, 3],
  [3, 4],
  [4, 0],
];

/** The look's four colours, taken in turn by the lines. */
const LOOK_COLOUR_COUNT = 4;

function starPoint(starIndex: number): readonly [number, number] {
  const star = STARS[starIndex];
  if (star === undefined) {
    throw new Error(`no star ${starIndex}`);
  }
  return star;
}

function linkHtml(linkIndex: number, fromStar: number, toStar: number): string {
  const [fromX, fromY] = starPoint(fromStar);
  const [toX, toY] = starPoint(toStar);
  const lookColour = (linkIndex % LOOK_COLOUR_COUNT) + 1;
  // pathLength 1 lets one dash pattern draw every line, whatever its length.
  return (
    `<line class="astro-constellation__link" style="--link-order:${linkIndex};stroke:var(--astro-trail-${lookColour})" ` +
    `x1="${fromX}" y1="${fromY}" x2="${toX}" y2="${toY}" pathLength="1"/>`
  );
}

function starHtml(starIndex: number): string {
  const [x, y] = starPoint(starIndex);
  return `<circle class="astro-constellation__star" style="--star-order:${starIndex}" cx="${x}" cy="${y}" r="1.7"/>`;
}

/** The figure's markup. It is decoration: the words beside it say what Astro is doing. */
export function astroConstellationHtml(): string {
  const links = LINKS.map(([fromStar, toStar], linkIndex) => linkHtml(linkIndex, fromStar, toStar)).join("");
  const stars = STARS.map((_star, starIndex) => starHtml(starIndex)).join("");
  return `<svg class="astro-constellation" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${links}${stars}</svg>`;
}

/** How many lines draw while Astro works, which the stylesheet's timing assumes. */
export const ASTRO_CONSTELLATION_LINK_COUNT = LINKS.length;
