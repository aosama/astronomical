import { SkillCardKind } from "./SkillCard";

/** The skills' icons, drawn inline: the page may load images only from its own
 * bundle, and inline SVG follows the text colour around it. Each skill has one
 * glyph everywhere it appears: the composer's launchers, home's ideas and the
 * conversation's tool steps draw the same paths, so a picture never means two
 * things. */
export const SKILL_GLYPH_PATHS: Record<SkillCardKind, string> = {
  [SkillCardKind.CHAT]: "M4 5h16v10H9l-5 4z",
  [SkillCardKind.PICTURES]: "M20 4l-8.5 8.5M9.5 11.5l3 3M9.5 14.5C7 14.5 5 16 5 18.5V20h2.5c2.5 0 4-1.5 4-3.5",
  [SkillCardKind.LOOK]: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
  [SkillCardKind.READ]: "M6 3h9l4 4v14H6zM14 3v5h5M9 13h7M9 17h5",
  [SkillCardKind.VOICE]: "M12 4v10M8 8v6M16 8v6M4 11v2M20 11v2",
};
