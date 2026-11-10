import assert from "node:assert/strict";
import { test } from "node:test";
import { mermaidThemeFor, prefersDarkAppearance, type AppearanceQueryHost } from "../src/DiagramTheme";

function hostWithAppearance(isDarkAppearance: boolean): AppearanceQueryHost & { askedMediaQueries: string[] } {
  const askedMediaQueries: string[] = [];
  return {
    askedMediaQueries,
    matchMedia: (mediaQuery: string) => {
      askedMediaQueries.push(mediaQuery);
      return { matches: isDarkAppearance };
    },
  };
}

test("should_draw_diagrams_with_the_dark_theme_on_a_mac_in_dark_appearance", () => {
  const darkHost = hostWithAppearance(true);
  assert.equal(mermaidThemeFor(darkHost), "dark");
  assert.deepEqual(darkHost.askedMediaQueries, ["(prefers-color-scheme: dark)"]);
});

test("should_not_draw_diagrams_with_the_dark_theme_on_a_mac_in_light_appearance", () => {
  assert.equal(mermaidThemeFor(hostWithAppearance(false)), "default");
  assert.equal(prefersDarkAppearance(hostWithAppearance(false)), false);
});

test("should_fall_back_to_the_light_theme_when_the_host_cannot_answer_media_queries", () => {
  assert.equal(mermaidThemeFor({}), "default");
});
