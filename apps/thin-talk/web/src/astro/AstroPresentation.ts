import type { ComposerView } from "../ComposerView";
import type { PictureModelGuidance } from "../PictureModelGuidance";
import type { TranscriptPersona } from "../TranscriptPersona";
import type { TranscriptView } from "../TranscriptView";

/** The companion's name wherever the composer speaks of it. */
export const ASTRO_NAME = "Astro";
/** The message field's placeholder while Astro is shown. */
export const ASTRO_COMPOSER_PLACEHOLDER = `Ask ${ASTRO_NAME} anything`;

/** The parts of the window that change look when Astro is shown or hidden. */
export interface AstroPresentationTargets {
  readonly appRoot: HTMLElement;
  readonly transcript: TranscriptView;
  readonly composer: ComposerView;
  readonly pictureModelGuidance: PictureModelGuidance;
}

/**
 * Applies one presentation to the whole window: the root classes the Astro
 * stylesheets key on, the transcript's persona, the composer's words, and
 * where the drawing tool sends someone without an image model.
 */
export function applyAstroPresentation(targets: AstroPresentationTargets, persona: TranscriptPersona): void {
  targets.appRoot.classList.toggle("app--astro", persona.isAstroShown);
  targets.transcript.setPersona(persona);
  targets.pictureModelGuidance.setAstroShown(persona.isAstroShown);
  targets.composer.setIdlePlaceholder(persona.isAstroShown ? ASTRO_COMPOSER_PLACEHOLDER : targets.composer.markupIdlePlaceholder);
  targets.composer.setAssistantName(persona.isAstroShown ? ASTRO_NAME : null);
}
