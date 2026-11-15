/** "2nd", "3rd", "11th": a place in the download line as people say it. */
export function formatPlaceInLine(place: number): string {
  const lastTwoDigits = place % 100;
  if (lastTwoDigits >= 11 && lastTwoDigits <= 13) {
    return `${place}th`;
  }
  const suffixByLastDigit: Record<number, string> = { 1: "st", 2: "nd", 3: "rd" };
  return `${place}${suffixByLastDigit[place % 10] ?? "th"}`;
}
