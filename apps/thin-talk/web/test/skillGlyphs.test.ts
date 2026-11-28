import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { SkillCardKind } from "../src/astro/SkillCard";
import { SKILL_GLYPH_PATHS } from "../src/astro/SkillGlyphs";

/**
 * Each skill has one glyph: the composer's launcher for a skill must draw the
 * same picture as that skill's idea on Astro's home, or a picture would mean
 * "attach a photo" in one place and "make a picture" in another.
 */
function launcherGlyphPath(skill: string): string {
  const indexPath = process.env["THINTALK_INDEX_HTML"];
  assert.ok(indexPath, "THINTALK_INDEX_HTML must point at the real index.html");
  const pageMarkup = readFileSync(indexPath, "utf8");
  const launcherPattern = new RegExp(`data-skill="${skill}"[^>]*>\\s*<svg[^>]*><path d="([^"]+)"`);
  const launcherMatch = launcherPattern.exec(pageMarkup);
  assert.ok(launcherMatch, `the composer should have a ${skill} launcher drawn with one path`);
  return launcherMatch[1] as string;
}

test("should_draw_each_composer_launcher_with_its_skill_cards_glyph", () => {
  for (const skill of [SkillCardKind.PICTURES, SkillCardKind.LOOK, SkillCardKind.READ]) {
    assert.equal(launcherGlyphPath(skill), SKILL_GLYPH_PATHS[skill], `${skill} must look the same in both places`);
  }
});

test("should_give_every_skill_a_different_glyph", () => {
  const glyphPaths = Object.values(SKILL_GLYPH_PATHS);
  assert.equal(new Set(glyphPaths).size, glyphPaths.length);
});
