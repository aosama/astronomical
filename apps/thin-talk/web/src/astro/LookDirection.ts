/**
 * Which of Astro's pre-rendered turns faces a pointer.
 *
 * `pointerAcrossFraction` is the pointer's position across the window, 0 at the
 * left edge and 1 at the right; values outside that (a pointer beyond the
 * window during a drag) are clamped. Frame 0 looks furthest left and the last
 * frame furthest right, matching the order of the sprites in web/astro.
 */
export function lookFrameForPointer(pointerAcrossFraction: number, lookFrameCount: number): number {
  if (lookFrameCount <= 1 || !Number.isFinite(pointerAcrossFraction)) {
    return Math.max(0, Math.floor((lookFrameCount - 1) / 2));
  }
  const clampedFraction = Math.min(1, Math.max(0, pointerAcrossFraction));
  return Math.round(clampedFraction * (lookFrameCount - 1));
}
