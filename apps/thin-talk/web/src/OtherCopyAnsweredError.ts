/** What the window says when the engine that answered belongs to another
 * install. Written for someone who installed Astronomical twice (the direct
 * download and the App Store version) and does not know the two share one
 * local address, so only one of them can run. */
export const OTHER_COPY_RUNNING_NOTICE =
  "Another copy of Astronomical is already running on this Mac, and only one can run at a time. Quit the other one, and this window carries on by itself.";

/**
 * The engine on this window's address answered, but it is not this app's
 * own: it reports another channel or keeps its files somewhere else.
 *
 * It is its own type because, unlike an answer that makes no sense, it can
 * stop being true while the window is open: the person quits the other copy
 * and this app's engine takes the address.
 */
export class OtherCopyAnsweredError extends Error {
  public constructor() {
    super(OTHER_COPY_RUNNING_NOTICE);
    this.name = "OtherCopyAnsweredError";
  }
}
