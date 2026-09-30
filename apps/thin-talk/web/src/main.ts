import { bootChatClient } from "./boot";

// The production entry point: the whole client is wired in boot.ts so the
// test harness can run the identical wiring against a fake host.
bootChatClient();
