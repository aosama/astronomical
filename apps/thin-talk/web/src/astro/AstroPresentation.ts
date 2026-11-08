import type { ComposerView } from "../ComposerView";
import type { PictureModelGuidance } from "../PictureModelGuidance";
import type { ComposerFriendlyNames } from "../ComposerFriendlyNames";
import type { TranscriptPersona } from "../TranscriptPersona";
import type { TranscriptView } from "../TranscriptView";
import { AstroStyle } from "./AstroStyle";

/** The message field's placeholder while Astro is shown. */
export const ASTRO_COMPOSER_PLACEHOLDER = "Ask Astro anything";

/** Plain words for the composer's pills in the Explorer style: the chat model
 * is Astro's brain and the image model its sketchbook. Pilots see the models'
 * own names, since teenagers often want to know which one they are using. */
export const EXPLORER_FRIENDLY_NAMES: ComposerFriendlyNames = {
  brain: "Astro's brain",
  sketchbook: "Sketchbook",
};

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
 * where the drawing tool sends someone without a picture model.
 * Explorer gets friendly pill names; Pilot keeps the models' own names.
 */
export function applyAstroPresentation(targets: AstroPresentationTargets, persona: TranscriptPersona): void {
  const isPilot = persona.isAstroShown && persona.style === AstroStyle.PILOT;
  targets.appRoot.classList.toggle("app--astro", persona.isAstroShown);
  targets.appRoot.classList.toggle("app--astro-pilot", isPilot);
  targets.transcript.setPersona(persona);
  targets.pictureModelGuidance.setAstroShown(persona.isAstroShown);
  const isExplorer = persona.isAstroShown && persona.style === AstroStyle.EXPLORER;
  targets.composer.setFriendlyNames(isExplorer ? EXPLORER_FRIENDLY_NAMES : null);
  targets.composer.setIdlePlaceholder(persona.isAstroShown ? ASTRO_COMPOSER_PLACEHOLDER : targets.composer.markupIdlePlaceholder);
}
