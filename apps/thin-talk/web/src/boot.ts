import type { PaperSize } from "./export/PaperSize";
import { ImageModelDirectory } from "./ImageModelDirectory";
import { ImageModelSelection } from "./ImageModelSelection";
import { AppConfig } from "./AppConfig";
import { HostChannel } from "./HostChannel";
import { SessionBridge } from "./SessionBridge";
import { SessionStore } from "./SessionStore";
import { AgentTurnRunner } from "./AgentTurnRunner";
import { CurrentTimeTool } from "./CurrentTimeTool";
import { GenerateImageTool } from "./GenerateImageTool";
import { ReadChosenFileTool } from "./ReadChosenFileTool";
import { DescribeChosenImageTool } from "./DescribeChosenImageTool";
import { SearchConversationsTool } from "./SearchConversationsTool";
import { ToolCatalog } from "./ToolCatalog";
import { DaemonClient } from "./DaemonClient";
import { TrustBoundary } from "./TrustBoundary";
import { MathPipeline } from "./MathPipeline";
import { DiagramPipeline } from "./DiagramPipeline";
import { Renderer } from "./Renderer";
import { TranscriptView } from "./TranscriptView";
import { ComposerView } from "./ComposerView";
import { SidebarView } from "./SidebarView";
import { SidebarResizer } from "./SidebarResizer";
import { runConversationMenu } from "./ConversationMenuFlow";
import { ChatController } from "./ChatController";
import { AppShell } from "./AppShell";
import { WindowChrome } from "./WindowChrome";
import { refreshOnReturnToWindow } from "./WindowReturnRefresh";
import { HostCommandRouter } from "./HostCommandRouter";
import { hostCommandHandlers } from "./HostCommandWiring";
import { AstroCompanion } from "./astro/AstroCompanion";
import { AstroLibrary } from "./astro/AstroLibrary";
import { applyAstroPresentation } from "./astro/AstroPresentation";
import { PictureModelGuidance } from "./PictureModelGuidance";
import { ExportLauncher } from "./ExportLauncher";
import { MessagePicker } from "./MessagePicker";
import { BriefNotice } from "./BriefNotice";
import { ModelLoadNotice, isWaitingForFirstWords } from "./ModelLoadNotice";
import { SpeechClient } from "./speech/SpeechClient";
import { SpeechAudioOutput, createWebAudioContext } from "./speech/SpeechAudioOutput";
import { SpeechController } from "./speech/SpeechController";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`the page is missing its "${selector}" element`);
  }
  return element;
}

function showApplicationTitle(config: AppConfig): void {
  document.title = config.applicationTitle;
  const sidebarAppName = document.querySelector(".sidebar__app-name");
  if (sidebarAppName) {
    sidebarAppName.textContent = config.applicationTitle;
  }
}

/** Wires exporting: the sheet the export script draws in, the message picker
 * and the brief notice that says a file was saved. */
function wireExport(controller: ChatController, transcript: TranscriptView, bridge: SessionBridge, diagrams: DiagramPipeline, printedPaper: PaperSize): ExportLauncher {
  const picker = new MessagePicker(
    {
      appRoot: requireElement<HTMLElement>("#app"),
      transcriptElement: requireElement<HTMLElement>("#transcript"),
      pickBar: requireElement<HTMLElement>("#pick-bar"),
      pickCount: requireElement<HTMLElement>('[data-testid="pick-count"]'),
      doneButton: requireElement<HTMLButtonElement>('[data-testid="pick-done"]'),
      cancelButton: requireElement<HTMLButtonElement>('[data-testid="pick-cancel"]'),
    },
    transcript,
  );
  return new ExportLauncher({
    sheetElement: requireElement<HTMLElement>("#export-sheet"),
    printedPaper,
    controller,
    transcript,
    bridge,
    picker,
    notice: new BriefNotice(requireElement<HTMLElement>("#brief-notice")),
    drawDiagramForPaper: (diagramSource) => diagrams.drawForPaper(diagramSource),
  });
}

/**
 * Wires every client component to the page's elements and starts the shell.
 * Both the production entry (main.ts) and the test harness call this, so there
 * is exactly one wiring path and a test boot is the production boot.
 */
export function bootChatClient(): AppShell {
  const config = AppConfig.fromWindow(window.__thintalkConfig);
  showApplicationTitle(config);
  const windowChrome = new WindowChrome(requireElement<HTMLElement>("#app"));
  windowChrome.applyTitlebarInset(config.titlebarInset);
  const host = new HostChannel();
  const bridge = new SessionBridge((request) => host.postSessionCall(request));
  const store = new SessionStore(bridge);
  const client = new DaemonClient(config);

  const speechClient = new SpeechClient(config.daemonBaseURL);
  const speech = new SpeechController({
    output: new SpeechAudioOutput(createWebAudioContext),
    fetchSpokenPiece: (pieceRequest) => speechClient.fetchSpokenPiece(pieceRequest),
    listVoiceModels: () => speechClient.listVoiceModels(),
    openModelsPage: () => host.postAction("openExternal", "", config.modelsPageURL),
  });

  const trust = new TrustBoundary();
  const math = new MathPipeline(trust);
  const diagrams = new DiagramPipeline();
  const renderer = new Renderer({ trust, math, diagrams });

  const transcript = new TranscriptView({
    transcript: requireElement<HTMLElement>("#transcript"),
    emptyState: requireElement<HTMLElement>("#empty-state"),
    renderer,
    onAction: (action) => {
      if (action.kind === "copy") {
        // The host owns the pasteboard, so the text travels in the action detail.
        // A code block's Copy button brings its own text; a message's Copy
        // button asks for the whole rendered answer.
        const copiedText = action.detail.length > 0 ? action.detail : transcript.renderedText(action.messageID);
        host.postAction("copy", action.messageID, copiedText);
        return;
      }
      if (action.kind === "speakAnswer") {
        const spokenReply = transcript.latestMessages().find((message) => message.id === action.messageID);
        if (spokenReply !== undefined) {
          speech.press(spokenReply.id, spokenReply.markdown);
        }
        return;
      }
      if (action.kind === "exportAnswer") {
        void exportLauncher.open({ kind: "oneAnswer", messageID: action.messageID });
        return;
      }
      if (action.kind === "saveImage" || action.kind === "copyImage") {
        host.postAction(action.kind === "saveImage" ? "saveGeneratedImage" : "copyGeneratedImage", "", action.detail);
        return;
      }
      host.postAction("openExternal", "", action.detail);
    },
    onError: (detail) => host.postError(detail),
  });

  // One choice of image model and one list of installed image models, shared
  // by the controller (which draws the composer's model panel) and the image tool
  // (which draws pictures), so the two can never disagree.
  const imageModelDirectory = new ImageModelDirectory(client);
  const imageModelSelection = new ImageModelSelection();
  const pictureModelGuidance = new PictureModelGuidance();
  const composerField = requireElement<HTMLTextAreaElement>("#composer-input");
  const composer = new ComposerView({
    field: composerField,
    sendButton: requireElement<HTMLButtonElement>('[data-testid="composer-send"]'),
    stopButton: requireElement<HTMLButtonElement>('[data-testid="composer-stop"]'),
    retryButton: requireElement<HTMLButtonElement>('[data-testid="composer-retry"]'),
    modelControl: requireElement<HTMLButtonElement>('[data-testid="composer-model"]'),
    modelLabel: requireElement<HTMLElement>('[data-testid="composer-model-label"]'),
    modelPanel: requireElement<HTMLElement>('[data-testid="composer-model-menu"]'),
    banner: requireElement<HTMLElement>('[data-testid="composer-banner"]'),
    bannerMessage: requireElement<HTMLElement>('[data-testid="banner-message"]'),
    bannerNextAction: requireElement<HTMLElement>('[data-testid="banner-next-action"]'),
    onAction: (action) => {
      switch (action.kind) {
        case "send":
          // Silence at the press itself, not once the answer starts: the reply
          // being read scrolls away under the new question, taking its Stop
          // button out of sight.
          speech.stop();
          controller.send(action.detail);
          return;
        case "stop":
          void controller.stop();
          return;
        case "retry":
          controller.retry();
          return;
        case "selectModel":
          controller.selectModel(action.detail);
          return;
        case "selectImageModel":
          controller.selectImageModel(action.detail);
          return;
        case "openModelPanel":
          void controller.refreshImageModels();
          return;
        case "setTools":
          controller.setToolsEnabled(action.detail === "on");
          return;
        case "openModelsPage":
          host.postAction("openExternal", "", config.modelsPageURL);
          return;
      }
    },
  });

  const sidebar = new SidebarView({
    newChatButton: requireElement<HTMLButtonElement>('[data-testid="sidebar-new-chat"]'),
    sessionList: requireElement<HTMLElement>("#session-list"),
    onSelect: (id) => void controller.openSession(id),
    onNewChat: () => void controller.newChat(),
    onConversationMenu: (id) => void runConversationMenu(id, { bridge, sidebar, controller, exportLauncher }),
    onRename: (id, title) => void controller.renameSession(id, title),
  });

  const sidebarResizer = new SidebarResizer({
    sidebarElement: requireElement<HTMLElement>("#sidebar"),
    onResizeEnd: (widthPixels) => controller.setSidebarWidth(widthPixels),
  });

  const controller = new ChatController({
    client,
    streamer: new AgentTurnRunner(
      client,
      new ToolCatalog([
        new CurrentTimeTool(),
        new GenerateImageTool(client, bridge, imageModelDirectory, imageModelSelection, pictureModelGuidance),
        new ReadChosenFileTool(bridge),
        new DescribeChosenImageTool(client, bridge),
        new SearchConversationsTool(bridge),
      ]),
      (imageName) => bridge.loadGeneratedImage(imageName),
    ),
    bridge,
    store,
    imageModelDirectory,
    imageModelSelection,
  });
  const modelLoadNotice = new ModelLoadNotice({
    isModelLoading: () => client.isModelLoading(),
    transcriptElement: requireElement<HTMLElement>("#transcript"),
  });
  controller.subscribe((publishedState) => modelLoadNotice.setWaitingForFirstWords(isWaitingForFirstWords(publishedState)));
  // Buttons follow the speech state, and wait while an answer is written,
  // since the engine is busy; the controller decides when the conversation
  // moving on ends a reading.
  let latestIsGenerating = false;
  speech.subscribe((speechState) => transcript.showSpeech(speechState, latestIsGenerating));
  controller.subscribe((publishedState) => {
    latestIsGenerating = publishedState.composer.isStreaming;
    speech.followConversation({
      isGenerating: latestIsGenerating,
      activeSessionID: publishedState.activeSessionID,
      shownMessageIDs: new Set(publishedState.transcript.messages.map((message) => message.id)),
    });
    transcript.showSpeech(speech.state, latestIsGenerating);
  });
  const exportLauncher = wireExport(controller, transcript, bridge, diagrams, config.printedPaper);
  const shell = new AppShell({ controller, transcript, composer, sidebar, host, sidebarResizer });
  shell.start();
  const astro = new AstroCompanion({
    bridge,
    library: new AstroLibrary(client),
    getStartedOverlay: requireElement<HTMLElement>("#get-started"),
    settingsOverlay: requireElement<HTMLElement>("#settings"),
    emptyState: requireElement<HTMLElement>("#empty-state"),
    composerField,
    skillLaunchers: requireElement<HTMLElement>("#skill-launchers"),
    startNewChat: () => controller.newChat(),
    applyPresentation: (persona) => applyAstroPresentation({ appRoot: requireElement<HTMLElement>("#app"), transcript, composer, pictureModelGuidance }, persona),
    onModelInstalled: () => {
      void controller.refreshChatModels();
      void controller.refreshImageModels();
    },
    restartEngine: config.canRestartEngine ? () => host.requestEngineRestart() : null,
    // The Observatory is served by the same engine the window talks to.
    openModelsPage: () => host.postAction("openExternal", "", `${config.daemonBaseURL}/download-models`),
  });
  new HostCommandRouter(
    hostCommandHandlers({ controller, transcript, composerField, host, windowChrome, astro, exportLauncher }),
  ).install(window);
  refreshOnReturnToWindow(window, document, () => {
    void controller.refreshChatModels();
    void controller.refreshImageModels();
  });
  void controller.load();
  void astro.start();
  return shell;
}
