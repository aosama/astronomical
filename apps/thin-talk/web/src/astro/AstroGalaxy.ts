/**
 * Astro drawn as a small spiral galaxy: one bright core and two arms of stars
 * that shrink and fade as they wind outward. It stands wherever Astro is
 * drawn (the home screen, the welcome, Your Astro and beside each answer).
 * The message field has none: it is where people type.
 *
 * Why only dots: an earlier figure joined its stars with lines and read as a
 * spider's web. Stars of different sizes and brightness with nothing between
 * them read as depth, and a galaxy has a motion of its own: it holds still at
 * rest and turns while Astro works.
 *
 * The stars are Astro's ink (silver) unless the person picked a colour look,
 * whose four colours then tint the arms in turn; the core stays ink so the
 * figure keeps one bright centre. Motion comes from the stylesheet, keyed on
 * the figure's state, so it costs no timer and holds still under Reduce
 * Motion. Every coordinate is fixed text, never conversation content, so the
 * markup needs no escaping.
 */

/** The figure's square drawing grid, and its centre where the core sits. */
const GRID_SIZE = 24;
const GRID_CENTRE = GRID_SIZE / 2;

const CORE_RADIUS = 2.1;
const ARM_COUNT = 2;
const STARS_PER_ARM = 7;

/** Where an arm leaves the core, and how far it winds between two stars. */
const ARM_START_ANGLE_RADIANS = 0.3;
const ARM_WINDING_RADIANS_PER_STAR = 0.62;
const ARM_FIRST_STAR_DISTANCE = 3.2;
const ARM_DISTANCE_PER_STAR = 1.28;

/** Stars shrink and dim along an arm; the floor keeps the last ones visible
 * at the small size beside an answer. */
const ARM_FIRST_STAR_RADIUS = 1.45;
const ARM_RADIUS_LOSS_PER_STAR = 0.15;
const ARM_SMALLEST_STAR_RADIUS = 0.5;
const ARM_BRIGHTNESS_LOSS_PER_STAR = 0.11;

/** A colour look has four colours, taken in turn along each arm. */
const LOOK_COLOUR_COUNT = 4;

function twoDecimals(length: number): string {
  return length.toFixed(2);
}

function armStarHtml(armIndex: number, starIndexInArm: number): string {
  const angleRadians = (armIndex * 2 * Math.PI) / ARM_COUNT + ARM_START_ANGLE_RADIANS + starIndexInArm * ARM_WINDING_RADIANS_PER_STAR;
  const distanceFromCore = ARM_FIRST_STAR_DISTANCE + starIndexInArm * ARM_DISTANCE_PER_STAR;
  const starRadius = Math.max(ARM_FIRST_STAR_RADIUS - starIndexInArm * ARM_RADIUS_LOSS_PER_STAR, ARM_SMALLEST_STAR_RADIUS);
  const brightness = 1 - starIndexInArm * ARM_BRIGHTNESS_LOSS_PER_STAR;
  const lookColour = (starIndexInArm % LOOK_COLOUR_COUNT) + 1;
  // Brightness is fill-opacity so the twinkle, which animates opacity,
  // multiplies with it instead of flattening the arm's fade.
  return (
    `<circle class="astro-galaxy__star" style="--star-order:${starIndexInArm};fill:var(--astro-colour-look-${lookColour},var(--astro-ink))" ` +
    `cx="${twoDecimals(GRID_CENTRE + distanceFromCore * Math.cos(angleRadians))}" ` +
    `cy="${twoDecimals(GRID_CENTRE + distanceFromCore * Math.sin(angleRadians))}" ` +
    `r="${twoDecimals(starRadius)}" fill-opacity="${twoDecimals(brightness)}"/>`
  );
}

function armsHtml(): string {
  const armStars: string[] = [];
  for (let armIndex = 0; armIndex < ARM_COUNT; armIndex += 1) {
    for (let starIndexInArm = 0; starIndexInArm < STARS_PER_ARM; starIndexInArm += 1) {
      armStars.push(armStarHtml(armIndex, starIndexInArm));
    }
  }
  return `<g class="astro-galaxy__arms">${armStars.join("")}</g>`;
}

/** The figure's markup. It is decoration: the words beside it say what Astro is doing. */
export function astroGalaxyHtml(): string {
  const core = `<circle class="astro-galaxy__core" cx="${GRID_CENTRE}" cy="${GRID_CENTRE}" r="${CORE_RADIUS}"/>`;
  return `<svg class="astro-galaxy" viewBox="0 0 ${GRID_SIZE} ${GRID_SIZE}" aria-hidden="true" focusable="false">${armsHtml()}${core}</svg>`;
}

/** How many stars the arms hold between them, for the tests that count them. */
export const ASTRO_GALAXY_ARM_STAR_COUNT = ARM_COUNT * STARS_PER_ARM;
