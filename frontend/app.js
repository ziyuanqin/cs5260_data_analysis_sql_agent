
// Keep session data stable across refresh/close to avoid cross-session file cleanup side effects.
window.addEventListener('DOMContentLoaded', () => {
  // 初始化页面状态
  loadPersistedState();

  // 确保一进来就是干净的，且根据 state.chatMode 隐藏/显示面板
  triggerGlobalCleanup();
});
// 前端主控脚本：负责会话状态管理、消息渲染、SSE 流式接收。
const STORAGE_KEY = "agent_ui_chat_sessions_v2";

const messageList = document.getElementById("messageList");
const historyList = document.getElementById("historyList");
const newChatBtn = document.getElementById("newChatBtn");
const composerForm = document.getElementById("composerForm");
const composerInput = document.getElementById("composerInput");
const mainPanel = document.getElementById("mainPanel");
const emptyState = document.getElementById("emptyState");
const emptyTitle = document.getElementById("emptyTitle");
const htmlPreviewPane = document.getElementById("htmlPreviewPane");
const htmlPreviewFrame = document.getElementById("htmlPreviewFrame");
const htmlPreviewCloseBtn = document.getElementById("htmlPreviewCloseBtn");
const artifactPreviewModal = document.getElementById("artifactPreviewModal");
const artifactPreviewTitle = document.getElementById("artifactPreviewTitle");
const artifactPreviewFrameModal = document.getElementById("artifactPreviewFrameModal");
const artifactPreviewText = document.getElementById("artifactPreviewText");
const artifactPreviewCloseBtn = document.getElementById("artifactPreviewCloseBtn");
const artifactPreviewDownloadBtn = document.getElementById("artifactPreviewDownloadBtn");
const taskStatusPane = document.getElementById("taskStatusPane");
const taskStatusList = document.getElementById("taskStatusList");
const taskStatusUsage = document.getElementById("taskStatusUsage");
const taskStatusToggleBtn = document.getElementById("taskStatusToggleBtn");
const taskRetryBtn = document.getElementById("taskRetryBtn");
const taskStopBtn = document.getElementById("taskStopBtn");
const taskClearBtn = document.getElementById("taskClearBtn");
const evidencePane = document.getElementById("evidencePane");
const evidenceList = document.getElementById("evidenceList");
const evidenceToggleBtn = document.getElementById("evidenceToggleBtn");
const taskChipBtn = document.getElementById("taskChipBtn");
const webSearchToggleBtn = document.getElementById("webSearchToggleBtn");
const webReadToggleBtn = document.getElementById("webReadToggleBtn");
const fileAccessToggleBtn = document.getElementById("fileAccessToggleBtn");
const generalCapabilityRow = document.getElementById("generalCapabilityRow");
const websiteBuilderBtn = document.getElementById("websiteBuilderBtn");
const slideBuilderBtn = document.getElementById("slideBuilderBtn");
const fileUploadBtn = document.getElementById("fileUploadBtn");
const fileUploadInput = document.getElementById("fileUploadInput");
const uploadStatus = document.getElementById("uploadStatus");
const edaAnalysisToggleBtn = document.getElementById("edaAnalysisToggleBtn");
const sqlAnalysisToggleBtn = document.getElementById("sqlAnalysisToggleBtn");
const modelSwitch = document.querySelector(".model-switch");

const GENERAL_MODELS = [
  { id: "deepseek", label: "DeepSeek" },
  { id: "openai", label: "OpenAI" },
  { id: "huggingface", label: "HuggingFace (Qwen)" },
];

function isValidGeneralModel(modelId) {
  return GENERAL_MODELS.some((item) => item.id === modelId);
}

function getGeneralModelLabel(modelId) {
  const hit = GENERAL_MODELS.find((item) => item.id === modelId);
  return hit ? hit.label : "DeepSeek";
}

//Database Connection
const dbConnectBtn = document.getElementById("dbConnectBtn");
const dbModal = document.getElementById("dbModal");
const closeModal = document.getElementById("closeModal");
const dbForm = document.getElementById("dbForm");
const dbDisconnectBtn = document.getElementById("dbDisconnectBtn");

// 可选后端配置：可在浏览器控制台设置 window.__CHAT_BACKEND_CONFIG__ 动态切换。
// 例如：window.__CHAT_BACKEND_CONFIG__ = { provider: "local_http", model: "qwen2.5:7b" }
const CHAT_BACKEND_CONFIG = window.__CHAT_BACKEND_CONFIG__ || {};
const SIDE_PANE_MIN_WIDTH = 1180;
let htmlPreviewDismissed = false;
let lastHtmlPreviewContent = "";
let currentHtmlPreviewSourceKey = "";
let dismissedHtmlPreviewSourceKey = "";
let mathTypesetTimer = null;
let activeStreamAbortController = null;
let activeArtifactPreview = null;

const state = {
  // 所有会话（按最近更新时间排序显示在左侧历史栏）
  conversations: [],
  // 当前正在查看/发送消息的会话 id
  activeConversationId: null,
  // 用于生成默认会话标题：新聊天 1/2/3...
  chatCounter: 1,
  // 防止并发提交（一次只允许一个流式请求）
  isStreaming: false,
  // UI 模式：点击专家模式按钮后切换欢迎语。
  chatMode: "general",
  generalModel: "deepseek",
  generalCapabilityPreset: "",
  webSearchEnabled: false,
  webReadEnabled: false,
  fileAccessEnabled: false,
  uploadedFiles: [],
  edaAnalysisEnabled: false,
  sqlAnalysisEnabled: false,
  taskStatusCollapsed: false,
  evidenceCollapsed: false,
  isEditingLastUser: false,
};

let modelMenuEl = null;

/**
 * 触发后端全量文件清理
 */
async function triggerGlobalCleanup() {
  try {
    const url = "/api/chat/cleanup";

    // 优先使用 fetch 发送请求（带上必要的 Headers）
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}) // 即便后端不需要参数，也传个空对象
    });

    // 清除前端相关状态
    state.uploadedFiles = [];
    state.file_name = "";

    // 如果有 UI 上的上传列表，记得同步
    if (typeof renderAll === "function") renderAll();

    console.log("--- [Cleanup] backend files cleared ---");
  } catch (err) {
    console.error("Cleanup failed:", err);
  }
}

function getGeneralModelText() {
  const selected = isValidGeneralModel(state.generalModel) ? state.generalModel : "deepseek";
  return `General Mode: ${getGeneralModelLabel(selected)}`;
}

function canShowSidePane() {
  return window.innerWidth > SIDE_PANE_MIN_WIDTH;
}

function closeModelMenu() {
  if (!modelMenuEl) return;
  modelMenuEl.hidden = true;
  if (modelSwitch) {
    modelSwitch.setAttribute("aria-expanded", "false");
  }
}

function openModelMenu() {
  if (!modelMenuEl || !modelSwitch) return;

  const rect = modelSwitch.getBoundingClientRect();
  modelMenuEl.style.left = `${rect.left}px`;
  modelMenuEl.style.top = `${rect.bottom + 8}px`;
  modelMenuEl.hidden = false;
  modelSwitch.setAttribute("aria-expanded", "true");
}

function syncModelMenuSelection() {
  if (!modelMenuEl) return;

  // Sync mode buttons
  const modeButtons = modelMenuEl.querySelectorAll("[data-mode]");
  for (const btn of modeButtons) {
    const mode = btn.getAttribute("data-mode") || "";
    const selected = mode === state.chatMode;
    btn.classList.toggle("active", selected);
    btn.setAttribute("aria-checked", String(selected));
  }

  // Sync model buttons (only shown in general mode)
  const items = modelMenuEl.querySelectorAll("[data-model-id]");
  for (const item of items) {
    const modelId = item.getAttribute("data-model-id") || "";
    const selected = modelId === state.generalModel;
    item.classList.toggle("active", selected);
    item.setAttribute("aria-checked", String(selected));
  }
}

function ensureModelMenu() {
  if (modelMenuEl || !modelSwitch) return;

  const menu = document.createElement("div");
  menu.className = "model-menu";
  menu.hidden = true;
  menu.setAttribute("role", "menu");
  menu.setAttribute("aria-label", "Mode and model selection");

  // Add mode selection (General / Expert)
  const modeLabel = document.createElement("div");
  modeLabel.className = "model-menu-section-label";
  modeLabel.textContent = "Mode";
  menu.appendChild(modeLabel);

  for (const mode of ["general", "expert"]) {
    const modeItem = document.createElement("button");
    modeItem.type = "button";
    modeItem.className = "model-menu-item";
    modeItem.setAttribute("role", "menuitemradio");
    modeItem.setAttribute("data-mode", mode);
    modeItem.textContent = mode === "expert" ? "🔬 Expert Mode" : "✨ General Mode";
    modeItem.addEventListener("click", () => {
      if (state.chatMode !== mode) {
        state.chatMode = mode;
        
        // Clear mode-specific states
        if (mode === "general") {
          state.edaAnalysisEnabled = false;
          state.sqlAnalysisEnabled = false;
        } else {
          state.generalCapabilityPreset = "";
        }
        
        renderAll();
        persistState();
      }
      closeModelMenu();
    });
    menu.appendChild(modeItem);
  }

  // Add divider
  const divider = document.createElement("hr");
  divider.className = "model-menu-divider";
  menu.appendChild(divider);

  // Add model selection (only for general mode)
  const modelLabel = document.createElement("div");
  modelLabel.className = "model-menu-section-label";
  modelLabel.textContent = "AI Model";
  menu.appendChild(modelLabel);

  for (const modelOption of GENERAL_MODELS) {
    const modelId = modelOption.id;
    const item = document.createElement("button");
    item.type = "button";
    item.className = "model-menu-item";
    item.setAttribute("role", "menuitemradio");
    item.setAttribute("data-model-id", modelId);
    item.textContent = modelOption.label;
    item.addEventListener("click", () => {
      state.generalModel = modelId;
      syncModeUi();
      persistState();
      closeModelMenu();
    });
    menu.appendChild(item);
  }

  document.body.appendChild(menu);
  modelMenuEl = menu;

  document.addEventListener("click", (e) => {
    if (!modelMenuEl || modelMenuEl.hidden) return;
    const target = e.target;
    if (!(target instanceof Node)) return;
    if (modelMenuEl.contains(target) || modelSwitch.contains(target)) return;
    closeModelMenu();
  });

  window.addEventListener("resize", () => {
    if (!modelMenuEl || modelMenuEl.hidden) return;
    closeModelMenu();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeModelMenu();
    }
  });

  syncModelMenuSelection();
}

function fileIdentity(file) {
  return `${file.name}__${file.size}__${file.lastModified}`;
}

function renderUploadedFiles() {
  if (!uploadStatus) return;

  if (!state.uploadedFiles.length) {
    uploadStatus.hidden = true;
    uploadStatus.innerHTML = "";
    return;
  }

  uploadStatus.hidden = false;
  uploadStatus.innerHTML = state.uploadedFiles
    .map(
      (item) =>
        `<span class="upload-file-item">${escapeHtml(item.name)}<button class="upload-file-remove" data-file-id="${escapeHtml(item.id)}" type="button" aria-label="Remove file">✕</button></span>`
    )
    .join("");
}

function getModeGreetingText() {
  return state.chatMode === "expert" ? "Hello, I'm Expert Mode" : "Hello, I'm General Mode";
}

function syncModeUi() {
  const isExpert = state.chatMode === "expert";
  if (!isValidGeneralModel(state.generalModel)) {
    state.generalModel = "deepseek";
  }

  const dbSidebarSection = document.querySelector(".sidebar-footer"); // 选中包含按钮的父容器
  if (dbSidebarSection) {
    // 只有专家模式才显示该区域
    dbSidebarSection.style.display = isExpert ? "flex" : "none";
  }

  if (isExpert) {
    closeModelMenu();
  }

  if (typeof updateDbUI === "function") {
    updateDbUI();
  }

  if (modelSwitch) {
    modelSwitch.textContent = isExpert ? "Expert Mode▾" : `${getGeneralModelText()}▾`;
    modelSwitch.setAttribute("aria-expanded", String(Boolean(modelMenuEl && !modelMenuEl.hidden)));
  }

  if (emptyTitle) {
    emptyTitle.textContent = getModeGreetingText();
  }

  if (taskChipBtn) {
    taskChipBtn.classList.toggle("active", isExpert);
    taskChipBtn.textContent = isExpert ? "★ Expert Mode" : "✶ Expert Mode";
    taskChipBtn.setAttribute("aria-pressed", String(isExpert));
    taskChipBtn.title = isExpert ? "Click to switch to General Mode" : "Click to switch to Expert Mode";
  }

  if (webSearchToggleBtn) {
    const isGeneral = !isExpert;
    webSearchToggleBtn.style.display = isGeneral ? "inline-flex" : "none";
    webSearchToggleBtn.classList.toggle("active", isGeneral && state.webSearchEnabled);
    webSearchToggleBtn.setAttribute("aria-pressed", String(isGeneral && state.webSearchEnabled));
    webSearchToggleBtn.title = state.webSearchEnabled ? "Click to disable Web Search" : "Click to enable Web Search";
  }

  if (generalCapabilityRow) {
    generalCapabilityRow.style.display = isExpert ? "none" : "flex";
  }

  if (websiteBuilderBtn) {
    const active = !isExpert && state.generalCapabilityPreset === "website_builder";
    websiteBuilderBtn.classList.toggle("active", active);
    websiteBuilderBtn.setAttribute("aria-pressed", String(active));
    websiteBuilderBtn.title = active ? "Click to cancel Website Builder preset" : "Click to enable Website Builder preset";
  }

  if (slideBuilderBtn) {
    const active = !isExpert && state.generalCapabilityPreset === "slide_builder";
    slideBuilderBtn.classList.toggle("active", active);
    slideBuilderBtn.setAttribute("aria-pressed", String(active));
    slideBuilderBtn.title = active ? "Click to cancel Slide Builder preset" : "Click to enable Slide Builder preset";
  }

  if (webReadToggleBtn) {
    const isGeneral = !isExpert;
    webReadToggleBtn.style.display = isGeneral ? "inline-flex" : "none";
    webReadToggleBtn.classList.toggle("active", isGeneral && state.webReadEnabled);
    webReadToggleBtn.setAttribute("aria-pressed", String(isGeneral && state.webReadEnabled));
    webReadToggleBtn.title = state.webReadEnabled ? "Click to disable Web Read" : "Click to enable Web Read";
  }

  if (fileAccessToggleBtn) {
    const isGeneral = !isExpert;
    fileAccessToggleBtn.style.display = isGeneral ? "inline-flex" : "none";
    fileAccessToggleBtn.classList.toggle("active", isGeneral && state.fileAccessEnabled);
    fileAccessToggleBtn.setAttribute("aria-pressed", String(isGeneral && state.fileAccessEnabled));
    fileAccessToggleBtn.title = state.fileAccessEnabled ? "Click to disable File Access" : "Click to enable File Access";
  }

  if (fileUploadBtn) {
    fileUploadBtn.style.display = isExpert ? "inline-flex" : "none";
    fileUploadBtn.classList.remove("active");
    fileUploadBtn.setAttribute("aria-pressed", "false");
    fileUploadBtn.title = "Click to upload files";
  }

  renderUploadedFiles();

  if (sqlAnalysisToggleBtn) {
    sqlAnalysisToggleBtn.style.display = isExpert ? "inline-flex" : "none";
    sqlAnalysisToggleBtn.classList.toggle("active", state.sqlAnalysisEnabled);
    sqlAnalysisToggleBtn.setAttribute("aria-pressed", String(state.sqlAnalysisEnabled));
    sqlAnalysisToggleBtn.title = state.sqlAnalysisEnabled ? "Click to disable SQL Analysis" : "Click to enable SQL Analysis";
  }

  if (edaAnalysisToggleBtn) {
    edaAnalysisToggleBtn.style.display = isExpert ? "inline-flex" : "none";
    edaAnalysisToggleBtn.classList.toggle("active", state.edaAnalysisEnabled);
    edaAnalysisToggleBtn.setAttribute("aria-pressed", String(state.edaAnalysisEnabled));
    edaAnalysisToggleBtn.title = state.edaAnalysisEnabled ? "Click to disable EDA Analysis" : "Click to enable EDA Analysis";
  }

  syncModelMenuSelection();
}

function createId() {
  // 优先使用浏览器原生 UUID，兼容时退化为时间戳随机串。
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function normalizeMessages(rawMessages) {
  if (!Array.isArray(rawMessages)) return [];
  return rawMessages
    .map((msg) => ({
      role: msg?.role === "user" ? "user" : "assistant",
      text: typeof msg?.text === "string" ? msg.text : "",
      runtime: normalizeRuntimeState(msg?.runtime),
      artifacts: normalizeArtifactItems(msg?.artifacts),
      edaReportThreadId: typeof msg?.edaReportThreadId === "string" ? msg.edaReportThreadId : null,
      edaMessageSource: msg?.edaMessageSource === "upload" || msg?.edaMessageSource === "chat" ? msg.edaMessageSource : null,
    }))
    .filter((msg) => msg.text.trim().length > 0 || msg.artifacts.length > 0)
    .filter((msg) => !(msg.role === "assistant" && isLegacyWelcomeText(msg.text)));
}

function normalizeTaskItems(rawItems) {
  if (!Array.isArray(rawItems)) return [];
  return rawItems
    .map((item, index) => {
      const title = typeof item?.title === "string" ? item.title.trim() : "";
      if (!title) return null;
      const statusRaw = typeof item?.status === "string" ? item.status.toLowerCase() : "pending";
      const status = statusRaw === "done" || statusRaw === "running" ? statusRaw : "pending";
      const idx = Number.isFinite(Number(item?.index)) ? Number(item.index) : index;
      return { index: idx, title, status };
    })
    .filter(Boolean);
}

function normalizeArtifactItems(rawItems) {
  if (!Array.isArray(rawItems)) return [];
  return rawItems
    .map((item) => {
      const relativePath = typeof item?.relative_path === "string" ? item.relative_path.trim() : "";
      const fallbackName = typeof item?.name === "string" ? item.name.trim() : "";
      const path = relativePath || fallbackName;
      if (!path) return null;
      return {
        name: fallbackName || path.split("/").pop() || path,
        relativePath: path,
        sizeBytes: Number.isFinite(Number(item?.size_bytes)) ? Number(item.size_bytes) : 0,
        previewable: Boolean(item?.previewable),
      };
    })
    .filter(Boolean);
}

function condenseAssistantTextForArtifacts(text, artifacts) {
  const raw = typeof text === "string" ? text : "";
  if (!raw.trim() || !Array.isArray(artifacts) || artifacts.length === 0) {
    return raw;
  }
  // When files are already generated, avoid dumping full source code in chat bubble.
  const replaced = raw.replace(/```[\s\S]*?```/g, "（代码内容已保存为文件，请使用下方文件卡片进行预览或下载。）");
  return replaced.replace(/\n{3,}/g, "\n\n").trim();
}

function createRuntimeState() {
  return {
    capability: "",
    stage: "",
    progressLogs: [],
    taskItems: [],
    artifacts: [],
  };
}

function normalizeRuntimeState(rawRuntime) {
  if (!rawRuntime || typeof rawRuntime !== "object") return createRuntimeState();
  const runtime = createRuntimeState();
  runtime.capability = typeof rawRuntime.capability === "string" ? rawRuntime.capability : "";
  runtime.stage = typeof rawRuntime.stage === "string" ? rawRuntime.stage : "";
  runtime.progressLogs = Array.isArray(rawRuntime.progressLogs)
    ? rawRuntime.progressLogs
        .map((line) => (typeof line === "string" ? line.trim() : ""))
        .filter((line) => line.length > 0)
        .slice(-50)
    : [];
  runtime.taskItems = normalizeTaskItems(rawRuntime.taskItems);
  runtime.artifacts = normalizeArtifactItems(rawRuntime.artifacts);
  return runtime;
}

function normalizeLastRequest(rawLastRequest) {
  if (!rawLastRequest || typeof rawLastRequest !== "object") return null;
  const text = typeof rawLastRequest.text === "string" ? rawLastRequest.text.trim() : "";
  if (!text) return null;

  const providerOptions =
    rawLastRequest.providerOptions && typeof rawLastRequest.providerOptions === "object"
      ? { ...rawLastRequest.providerOptions }
      : {};

  return {
    text,
    providerOptions,
    mode: rawLastRequest.mode === "expert" ? "expert" : "general",
    generalModel: isValidGeneralModel(rawLastRequest.generalModel)
      ? rawLastRequest.generalModel
      : "deepseek",
    timestamp: Number(rawLastRequest.timestamp || Date.now()),
  };
}

function ensureConversationArtifacts(convo) {
  if (!convo) return [];
  if (!Array.isArray(convo.artifacts)) {
    convo.artifacts = [];
  }
  return convo.artifacts;
}

function ensureMessageRuntime(message) {
  if (!message || typeof message !== "object") return createRuntimeState();
  if (!message.runtime || typeof message.runtime !== "object") {
    message.runtime = createRuntimeState();
  } else {
    message.runtime = normalizeRuntimeState(message.runtime);
  }
  return message.runtime;
}

function upsertArtifactItem(list, incomingItem) {
  const normalized = normalizeArtifactItems([incomingItem])[0];
  if (!normalized) return list;
  const target = Array.isArray(list) ? list : [];
  const existingIndex = target.findIndex((item) => item.relativePath === normalized.relativePath);
  if (existingIndex >= 0) {
    target[existingIndex] = { ...target[existingIndex], ...normalized };
  } else {
    target.push(normalized);
  }
  return target;
}

async function previewEdaReport(threadId) {
  if (typeof threadId !== "string" || !threadId.trim()) return;

  const safeId = encodeURIComponent(threadId.trim());
  const resp = await fetch(`/api/eda/report/${safeId}`);
  if (!resp.ok) {
    throw new Error(`Report fetch failed: HTTP ${resp.status}`);
  }

  const html = await resp.text();
  htmlPreviewDismissed = false;
  dismissedHtmlPreviewSourceKey = "";
  syncHtmlPreview({ html, sourceKey: `eda-report:${threadId}` });
}

function downloadEdaReport(threadId) {
  if (typeof threadId !== "string" || !threadId.trim()) return;
  const safeId = encodeURIComponent(threadId.trim());
  window.open(`/api/eda/report/${safeId}?download=true`, "_blank", "noopener");
}

function downloadEdaCsv(threadId) {
  if (typeof threadId !== "string" || !threadId.trim()) return;
  const safeId = encodeURIComponent(threadId.trim());
  window.open(`/api/eda/download-csv/${safeId}`, "_blank", "noopener");
}

function buildArtifactDownloadUrl(sessionId, artifactPath) {
  const safeSession = encodeURIComponent(String(sessionId || "").trim());
  const safeArtifact = String(artifactPath || "")
    .trim()
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
  return `/api/chat/artifacts/${safeSession}/download/${safeArtifact}`;
}

function closeArtifactPreviewModal() {
  if (!artifactPreviewModal) return;
  artifactPreviewModal.hidden = true;
  document.body.classList.remove("modal-open");
  if (artifactPreviewFrameModal) {
    artifactPreviewFrameModal.srcdoc = "";
    artifactPreviewFrameModal.hidden = true;
  }
  if (artifactPreviewText) {
    artifactPreviewText.textContent = "";
    artifactPreviewText.hidden = true;
  }
  activeArtifactPreview = null;
}

function showArtifactPreviewInModal({ sessionId, artifactPath, contentType, rawText }) {
  if (!artifactPreviewModal || !artifactPreviewTitle || !artifactPreviewFrameModal || !artifactPreviewText) {
    return;
  }

  const lowerPath = String(artifactPath || "").toLowerCase();
  const content = typeof rawText === "string" ? rawText : "";
  const likelyHtml =
    String(contentType || "").includes("text/html") ||
    lowerPath.endsWith(".html") ||
    lowerPath.endsWith(".htm") ||
    /<!doctype\s+html|<html[\s>]/i.test(content);

  artifactPreviewTitle.textContent = artifactPath || "Artifact Preview";
  artifactPreviewModal.hidden = false;
  document.body.classList.add("modal-open");

  if (likelyHtml) {
    const normalized = normalizeHtmlForPreview(content) || wrapHtmlFragment(content);
    artifactPreviewText.hidden = true;
    artifactPreviewText.textContent = "";
    artifactPreviewFrameModal.hidden = false;
    artifactPreviewFrameModal.srcdoc = normalized;
  } else {
    artifactPreviewFrameModal.hidden = true;
    artifactPreviewFrameModal.srcdoc = "";
    artifactPreviewText.hidden = false;
    artifactPreviewText.textContent = content;
  }

  activeArtifactPreview = {
    sessionId: String(sessionId || ""),
    artifactPath: String(artifactPath || ""),
  };
}

async function previewArtifact(sessionId, artifactPath) {
  const url = buildArtifactDownloadUrl(sessionId, artifactPath);
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Artifact preview failed: HTTP ${resp.status}`);
  }
  const contentType = (resp.headers.get("content-type") || "").toLowerCase();
  const rawText = await resp.text();
  showArtifactPreviewInModal({ sessionId, artifactPath, contentType, rawText });
}

function downloadArtifact(sessionId, artifactPath) {
  const url = buildArtifactDownloadUrl(sessionId, artifactPath);
  window.open(url, "_blank", "noopener");
}

async function refreshConversationArtifacts(conversationId, assistantMessage = null) {
  const safeSession = encodeURIComponent(String(conversationId || "").trim());
  if (!safeSession) return;
  const resp = await fetch(`/api/chat/artifacts/${safeSession}`);
  if (!resp.ok) return;
  const payload = await resp.json();
  const items = normalizeArtifactItems(payload?.items);
  const convo = state.conversations.find((item) => item.id === conversationId);
  if (convo) {
    convo.artifacts = items;
  }
  if (assistantMessage) {
    const runtime = ensureMessageRuntime(assistantMessage);
    runtime.artifacts = items;
    assistantMessage.artifacts = items;
  }
}

async function handleArtifactActionClick(target) {
  if (!(target instanceof HTMLElement)) return false;

  const openBtn = target.closest("[data-artifact-open='1']");
  if (openBtn) {
    const sessionId = openBtn.getAttribute("data-session-id") || "";
    const artifactPath = openBtn.getAttribute("data-artifact-path") || "";
    if (!sessionId || !artifactPath) return true;
    try {
      await previewArtifact(sessionId, artifactPath);
    } catch (err) {
      appendAssistantSystemMessage(`Artifact preview failed: ${err?.message || "unknown error"}`);
    }
    return true;
  }

  const previewBtn = target.closest("[data-artifact-preview='1']");
  if (previewBtn) {
    const sessionId = previewBtn.getAttribute("data-session-id") || "";
    const artifactPath = previewBtn.getAttribute("data-artifact-path") || "";
    if (!sessionId || !artifactPath) return true;
    try {
      await previewArtifact(sessionId, artifactPath);
    } catch (err) {
      appendAssistantSystemMessage(`Artifact preview failed: ${err?.message || "unknown error"}`);
    }
    return true;
  }

  const downloadBtn = target.closest("[data-artifact-download='1']");
  if (downloadBtn) {
    const sessionId = downloadBtn.getAttribute("data-session-id") || "";
    const artifactPath = downloadBtn.getAttribute("data-artifact-path") || "";
    if (!sessionId || !artifactPath) return true;
    downloadArtifact(sessionId, artifactPath);
    return true;
  }

  const bundleBtn = target.closest("[data-artifact-bundle='1']");
  if (bundleBtn) {
    const sessionId = bundleBtn.getAttribute("data-session-id") || "";
    if (!sessionId) return true;
    window.open(`/api/chat/artifacts/${encodeURIComponent(sessionId)}/bundle`, "_blank", "noopener");
    return true;
  }

  return false;
}

function isLegacyWelcomeText(text) {
  if (typeof text !== "string") return false;
  const normalized = text.replaceAll("～", "~").trim();
  return normalized === "Hello~You can start a new conversation";
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function parseMarkdownTable(line) {
  const stripped = line.trim();
  if (!stripped.startsWith("|")) return null;
  const cells = stripped
    .split("|")
    .map((cell) => cell.trim())
    .filter((cell, idx, arr) => !(idx === 0 && cell === "") && !(idx === arr.length - 1 && cell === ""));
  return cells.length ? cells : null;
}

function isTableSeparatorLine(line) {
  const stripped = line.trim();
  if (!stripped.startsWith("|")) return false;
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(stripped);
}

function isFenceStartLine(line) {
  return /^```/.test(line.trim());
}

function isHrLine(line) {
  return /^\s{0,3}([-*_])\s*\1\s*\1([\s*_\-]*)$/.test(line);
}

function isHeadingLine(line) {
  return /^\s{0,3}#{1,6}\s+/.test(line);
}

function isBlockquoteLine(line) {
  return /^\s{0,3}>\s?/.test(line);
}

function isUnorderedListLine(line) {
  return /^\s{0,3}[-*+]\s+/.test(line);
}

function isOrderedListLine(line) {
  return /^\s{0,3}\d+\.\s+/.test(line);
}

function getFenceLanguage(line) {
  const match = line.trim().match(/^```([a-zA-Z0-9_+-]*)/);
  return (match?.[1] || "").trim().toLowerCase();
}

function normalizeMathDelimiters(text) {
  if (typeof text !== "string" || text.length === 0) return "";

  return text
    .replaceAll("\\\\(", "\\(")
    .replaceAll("\\\\)", "\\)")
    .replaceAll("\\\\[", "\\[")
    .replaceAll("\\\\]", "\\]");
}

function normalizeCodeFencePayload(text) {
  if (typeof text !== "string" || text.length === 0) return "";

  let normalized = text;
  const hasFenceSignal = /```|\\`\\`\\`|&#96;&#96;&#96;|&grave;&grave;&grave;/.test(normalized);
  if (!hasFenceSignal) {
    return normalized;
  }

  // 某些后端会把换行和反引号二次转义，先做轻量还原以便识别代码块。
  if (!normalized.includes("\n") && normalized.includes("\\n")) {
    normalized = normalized.replaceAll("\\n", "\n");
  }

  normalized = normalized.replaceAll("\\t", "\t").replaceAll("\\`\\`\\`", "```").replaceAll("&#96;", "`").replaceAll("&grave;", "`");

  // 兼容流式输出中 "blockquote + fence" 的混合写法：
  // > ```sql
  // > SELECT ...
  // col2 ...
  // > ```
  // 这类内容在前端解析时容易断裂，先归一成标准 fenced code block。
  const lines = normalized.split(/\r?\n/);
  const repaired = [];
  let inQuotedFence = false;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] || "";
    const trimmed = line.trim();

    if (!inQuotedFence && /^>\s*```/.test(trimmed)) {
      repaired.push(trimmed.replace(/^>\s*/, ""));
      inQuotedFence = true;
      continue;
    }

    if (inQuotedFence && /^>\s*```\s*$/.test(trimmed)) {
      repaired.push("```");
      inQuotedFence = false;
      continue;
    }

    if (inQuotedFence) {
      repaired.push(line.replace(/^\s*>\s?/, ""));
      continue;
    }

    repaired.push(line);
  }

  normalized = repaired.join("\n");
  return normalized;
}

function queueMathTypeset() {
  if (!messageList) return;

  const mathJax = window.MathJax;
  if (!mathJax || typeof mathJax.typesetPromise !== "function") return;

  if (mathTypesetTimer) {
    clearTimeout(mathTypesetTimer);
  }

  mathTypesetTimer = setTimeout(() => {
    mathJax.typesetPromise([messageList]).catch(() => {
      // 忽略数学公式局部渲染失败，避免影响聊天流程。
    });
  }, 70);
}

function renderInlineMarkdown(text) {
  if (typeof text !== "string" || text.length === 0) return "";

  const codeTokens = [];
  let html = escapeHtml(text);

  // 先占位行内代码，避免被后续粗体/链接规则误处理。
  html = html.replace(/`([^`\n]+)`/g, (_, codeText) => {
    const token = `@@INLINE_CODE_${codeTokens.length}@@`;
    codeTokens.push(`<code>${codeText}</code>`);
    return token;
  });

  html = html.replace(
      /!\[([^\]]*)\]\((data:image\/[a-zA-Z]+;base64,[^)]+)\)/g,
      '<img src="$2" alt="$1" class="inline-image" />'
  );

  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  // 单星号是强调（斜体），与双星号加粗区分处理。
  html = html.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
  html = html.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  for (let i = 0; i < codeTokens.length; i += 1) {
    html = html.replace(`@@INLINE_CODE_${i}@@`, codeTokens[i]);
  }

  return html;
}

function isSpecialMarkdownBlockStart(lines, idx) {
  const line = lines[idx] || "";
  const nextLine = lines[idx + 1] || "";

  if (!line.trim()) return true;
  if (isFenceStartLine(line)) return true;
  if (isHrLine(line)) return true;
  if (isHeadingLine(line)) return true;
  if (isBlockquoteLine(line)) return true;
  if (isUnorderedListLine(line) || isOrderedListLine(line)) return true;

  const headerCells = parseMarkdownTable(line);
  if (headerCells && isTableSeparatorLine(nextLine)) return true;

  return false;
}

function renderAssistantMessageHtml(text) {
  // 轻量 Markdown 渲染：支持代码块、标题、列表、引用、表格和基础行内格式。
  const normalizedText = normalizeCodeFencePayload(normalizeMathDelimiters(text));
  const lines = normalizedText.split(/\r?\n/);
  const parts = [];
  let idx = 0;

  while (idx < lines.length) {
    const line = lines[idx] || "";

    if (!line.trim()) {
      idx += 1;
      continue;
    }

    if (isFenceStartLine(line)) {
      const trimmed = line.trim();
      const inlineFenceMatch = trimmed.match(/^```([a-zA-Z0-9_+-]*)\s+([\s\S]*?)```$/);
      if (inlineFenceMatch) {
        const inlineLang = (inlineFenceMatch[1] || "text").trim().toLowerCase() || "text";
        const inlineCode = inlineFenceMatch[2] || "";
        parts.push(
          `<div class="code-block"><div class="code-head">${escapeHtml(inlineLang)}</div><pre><code>${escapeHtml(inlineCode)}</code></pre></div>`
        );
        idx += 1;
        continue;
      }

      const lang = getFenceLanguage(line) || "text";
      const codeLines = [];
      idx += 1;

      while (idx < lines.length && !isFenceStartLine(lines[idx])) {
        codeLines.push(lines[idx]);
        idx += 1;
      }

      if (idx < lines.length && isFenceStartLine(lines[idx])) {
        idx += 1;
      }

      parts.push(
        `<div class="code-block"><div class="code-head">${escapeHtml(lang)}</div><pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre></div>`
      );
      continue;
    }

    if (isHrLine(line)) {
      parts.push("<hr />");
      idx += 1;
      continue;
    }

    if (isHeadingLine(line)) {
      const match = line.match(/^\s{0,3}(#{1,6})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const headingText = renderInlineMarkdown(match[2].trim());
        parts.push(`<h${level}>${headingText}</h${level}>`);
      }
      idx += 1;
      continue;
    }

    if (isBlockquoteLine(line)) {
      const quoteLines = [];
      while (idx < lines.length && isBlockquoteLine(lines[idx])) {
        quoteLines.push(lines[idx].replace(/^\s{0,3}>\s?/, ""));
        idx += 1;
      }
      const quoteHtml = quoteLines.map((item) => renderInlineMarkdown(item)).join("<br>");
      parts.push(`<blockquote>${quoteHtml}</blockquote>`);
      continue;
    }

    if (isUnorderedListLine(line)) {
      const items = [];
      while (idx < lines.length && isUnorderedListLine(lines[idx])) {
        items.push(lines[idx].replace(/^\s{0,3}[-*+]\s+/, ""));
        idx += 1;
      }
      parts.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (isOrderedListLine(line)) {
      const items = [];
      while (idx < lines.length && isOrderedListLine(lines[idx])) {
        items.push(lines[idx].replace(/^\s{0,3}\d+\.\s+/, ""));
        idx += 1;
      }
      parts.push(`<ol>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
      continue;
    }

    const headerCells = parseMarkdownTable(lines[idx]);
    const separatorLine = lines[idx + 1] || "";

    if (headerCells && isTableSeparatorLine(separatorLine)) {
      const rows = [];
      idx += 2;

      while (idx < lines.length) {
        const rowCells = parseMarkdownTable(lines[idx]);
        if (!rowCells) break;
        rows.push(rowCells);
        idx += 1;
      }

      const maxCols = Math.max(headerCells.length, ...rows.map((row) => row.length));
      const normalizedHeader = [...headerCells, ...Array(Math.max(0, maxCols - headerCells.length)).fill("")];
      const normalizedRows = rows.map((row) => [...row, ...Array(Math.max(0, maxCols - row.length)).fill("")]);

      const headHtml = normalizedHeader.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("");
      const bodyHtml = normalizedRows
        .map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`)
        .join("");

      parts.push(`<div class="table-wrap"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`);
      continue;
    }

    const textBuffer = [];
    while (idx < lines.length) {
      if (isSpecialMarkdownBlockStart(lines, idx)) break;
      textBuffer.push(lines[idx]);
      idx += 1;
    }

    const paragraph = textBuffer.join("\n").trim();
    if (paragraph) {
      const paragraphHtml = paragraph
        .split("\n")
        .map((item) => renderInlineMarkdown(item))
        .join("<br>");
      parts.push(`<p>${paragraphHtml}</p>`);
    }
  }

  return parts.join("") || `<p>${escapeHtml(normalizedText)}</p>`;
}

function formatBytes(size) {
  const value = Number(size || 0);
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function detectArtifactType(fileName) {
  const lower = String(fileName || "").toLowerCase();
  if (lower.endsWith(".html") || lower.endsWith(".htm")) return "Web";
  if (lower.endsWith(".md") || lower.endsWith(".txt")) return "Doc";
  if (lower.endsWith(".js") || lower.endsWith(".ts")) return "Script";
  if (lower.endsWith(".css")) return "Style";
  if (lower.endsWith(".json")) return "Data";
  return "File";
}

function artifactGlyph(fileName) {
  const lower = String(fileName || "").toLowerCase();
  if (lower.endsWith(".html") || lower.endsWith(".htm")) return "HTML";
  if (lower.endsWith(".md")) return "MD";
  if (lower.endsWith(".css")) return "CSS";
  if (lower.endsWith(".js")) return "JS";
  if (lower.endsWith(".ts")) return "TS";
  if (lower.endsWith(".json")) return "{}";
  if (lower.endsWith(".txt")) return "TXT";
  if (lower.endsWith(".csv")) return "CSV";
  if (lower.endsWith(".zip")) return "ZIP";
  return "FILE";
}

function artifactBaseName(filePath) {
  const normalized = String(filePath || "").replaceAll("\\", "/");
  const parts = normalized.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : normalized;
}

function renderArtifactIconTile(sessionId, item, compact = false, showName = false) {
  const relativePath = String(item?.relativePath || item?.name || "").trim();
  if (!relativePath) return "";
  const escapedSession = escapeHtml(sessionId);
  const escapedPath = escapeHtml(relativePath);
  const actionAttr = item?.previewable ? "data-artifact-open='1'" : "data-artifact-download='1'";
  const actionText = item?.previewable ? "Preview" : "Download";
  const title = escapeHtml(
    `${actionText}: ${relativePath}\n${detectArtifactType(relativePath)} · ${formatBytes(item?.sizeBytes)}`
  );
  const compactClass = compact ? "compact" : "";
  const buttonHtml = `
    <button
      class="artifact-icon-tile ${compactClass}"
      type="button"
      ${actionAttr}
      data-session-id="${escapedSession}"
      data-artifact-path="${escapedPath}"
      title="${title}"
      aria-label="${escapeHtml(`${actionText} ${relativePath}`)}"
    >
      <span class="artifact-icon-glyph">${escapeHtml(artifactGlyph(relativePath))}</span>
    </button>
  `;
  if (!showName) return buttonHtml;
  const name = escapeHtml(artifactBaseName(relativePath));
  return `
    <div class="artifact-file-card ${compactClass}">
      ${buttonHtml}
      <div class="artifact-file-name" title="${escapeHtml(relativePath)}">${name}</div>
    </div>
  `;
}

function renderRuntimeTaskItems(taskItems, isPending = false) {
  if (!Array.isArray(taskItems) || taskItems.length === 0) return "";
  const rows = taskItems
    .map((item, idx) => {
      const status = item.status || "pending";
      const symbol = status === "done" ? "✓" : status === "running" ? "●" : "○";
      const loadingClass = isPending && status === "running" ? "loading-scan" : "";
      return `
        <div class="runtime-task-item status-${escapeHtml(status)} ${loadingClass}">
          <span class="runtime-task-index">${idx + 1}</span>
          <span class="runtime-task-symbol">${symbol}</span>
          <span class="runtime-task-title">${escapeHtml(item.title)}</span>
        </div>
      `;
    })
    .join("");
  return `
    <div class="runtime-task-wrap">
      <div class="runtime-section-title">任务进度</div>
      <div class="runtime-task-list">${rows}</div>
    </div>
  `;
}

function renderRuntimeArtifacts(conversationId, artifacts) {
  if (!Array.isArray(artifacts) || artifacts.length === 0) return "";
  const escapedSession = escapeHtml(conversationId);
  const tiles = artifacts.map((item) => renderArtifactIconTile(conversationId, item, false, true)).join("");
  return `
    <div class="runtime-artifact-wrap">
      <div class="runtime-artifact-top">
        <div class="runtime-section-title">交付文件</div>
        <button class="runtime-artifact-btn ghost" type="button" data-artifact-bundle="1" data-session-id="${escapedSession}">下载全部 ZIP</button>
      </div>
      <div class="artifact-icon-grid">${tiles}</div>
    </div>
  `;
}

function renderAssistantArtifactCards(conversationId, artifacts) {
  if (!Array.isArray(artifacts) || artifacts.length === 0) return "";
  const escapedSession = escapeHtml(conversationId);
  const tiles = artifacts.map((item) => renderArtifactIconTile(conversationId, item, false, true)).join("");
  return `
    <section class="assistant-artifacts">
      <div class="assistant-artifacts-head">
        <span>交付文件</span>
        <button class="runtime-artifact-btn ghost" type="button" data-artifact-bundle="1" data-session-id="${escapedSession}">下载全部 ZIP</button>
      </div>
      <div class="assistant-artifact-grid">${tiles}</div>
    </section>
  `;
}

function renderAgentRuntimeCard(runtime, conversationId, usage, isPending = false) {
  const normalized = normalizeRuntimeState(runtime);
  const hasLogs = normalized.progressLogs.length > 0;
  const hasTaskItems = normalized.taskItems.length > 0;
  const hasArtifacts = normalized.artifacts.length > 0;
  const usageText = formatUsageText(usage);
  if (!hasLogs && !hasTaskItems && !hasArtifacts) return "";

  const stageText = normalized.stage ? formatStageLabel(normalized.stage) : "Agent Runtime";
  const capabilityText = normalized.capability ? ` · ${normalized.capability}` : "";
  const runningItem = normalized.taskItems.find((item) => item.status === "running");
  const summaryText = runningItem
    ? `正在执行：${runningItem.title}`
    : normalized.taskItems.length > 0
      ? "任务步骤已规划，持续执行中"
      : "任务执行中";
  const logLines = normalized.progressLogs.slice(-8);
  const logs = hasLogs
    ? logLines
        .map((line, index) => {
          const loadingClass = isPending && index === logLines.length - 1 ? "loading-scan" : "";
          return `<div class="runtime-log-line ${loadingClass}">${escapeHtml(line)}</div>`;
        })
        .join("")
    : `<div class="runtime-log-line muted">Running...</div>`;

  return `
    <section class="agent-runtime-card ${isPending ? "is-running" : ""}">
      <div class="runtime-head">
        <div class="runtime-brand">
          <span class="runtime-brand-name">manus</span>
          <span class="runtime-brand-lite">Lite</span>
        </div>
        ${usageText ? `<span class="runtime-usage">${escapeHtml(usageText)}</span>` : ""}
      </div>
      <div class="runtime-title">${escapeHtml(stageText + capabilityText)}</div>
      <div class="runtime-summary ${isPending ? "loading-scan" : ""}">${escapeHtml(summaryText)}</div>
      ${renderRuntimeTaskItems(normalized.taskItems, isPending)}
      <div class="runtime-log-wrap">${logs}</div>
      ${renderRuntimeArtifacts(conversationId, normalized.artifacts)}
    </section>
  `;
}

function buildPendingLabelFromMeta(meta) {
  const stage = String(meta?.stage || "").toLowerCase();
  const taskItems = normalizeTaskItems(meta?.task_items);
  const runningItem = taskItems.find((item) => item.status === "running");
  const stageMap = {
    router: "路由中",
    planner: "规划中",
    executor: "执行中",
    reviewer: "评审中",
    summarizer: "汇总中",
    tool: "工具处理中",
    artifact: "文件生成中",
    budget: "预算检查中",
  };
  const stageText = stageMap[stage] || "思考中";
  if (runningItem && runningItem.title) {
    return `思考中 · ${stageText}：${runningItem.title}`;
  }
  return `思考中 · ${stageText}`;
}

// 自动根据内容调整输入框高度
function autoResizeTextarea() {
  composerInput.style.height = "38px";
  composerInput.style.height = `${Math.min(composerInput.scrollHeight, 140)}px`;
}

// 工具：截断标题，避免历史栏过长
function shortTitle(text, max = 18) {
  if (!text) return "未命名聊天";
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

// 当前激活会话
function getActiveConversation() {
  return state.conversations.find((c) => c.id === state.activeConversationId) || null;
}

function findLastUserMessageIndex(convo) {
  if (!convo || !Array.isArray(convo.messages)) return -1;
  for (let idx = convo.messages.length - 1; idx >= 0; idx -= 1) {
    if (convo.messages[idx]?.role === "user") return idx;
  }
  return -1;
}

function cancelLastUserEditing() {
  state.isEditingLastUser = false;
}

function beginLastUserEditing() {
  if (state.isStreaming) return;
  const convo = getActiveConversation();
  const idx = findLastUserMessageIndex(convo);
  if (!convo || idx < 0) return;
  const message = convo.messages[idx];
  state.isEditingLastUser = true;
  composerInput.value = String(message?.text || "");
  autoResizeTextarea();
  composerInput.focus();
  renderMessages();
}

function isConversationEmpty(convo) {
  if (!convo || !Array.isArray(convo.messages)) return true;
  return convo.messages.length === 0;
}

function formatStageLabel(stage) {
  const normalized = String(stage || "").toLowerCase();
  const map = {
    router: "Router",
    planner: "Planner",
    executor: "Executor",
    reviewer: "Reviewer",
    summarizer: "Summarizer",
    tool: "Tool",
    artifact: "Artifact",
    budget: "Budget",
  };
  return map[normalized] || normalized || "Status";
}

function normalizeStatusMessage(stage, message) {
  const raw = String(message || "").trim();
  if (!raw) return "";
  const cleaned = raw.replace(/^\[[^\]]+\]\s*/u, "").trim();
  const normalizedStage = String(stage || "").toLowerCase();

  const tryParseJson = (text) => {
    const candidate = String(text || "").trim();
    if (!candidate || (!candidate.startsWith("{") && !candidate.startsWith("["))) return null;
    try {
      return JSON.parse(candidate);
    } catch {
      return null;
    }
  };

  const compactText = (text, maxChars = 240) => {
    const normalized = String(text || "").replace(/\s+/g, " ").trim();
    if (!normalized) return "";
    return normalized.length > maxChars ? `${normalized.slice(0, maxChars)}...` : normalized;
  };

  const humanizeToolAction = (obj) => {
    if (!obj || typeof obj !== "object") return "";
    const action = String(obj.action || "").toUpperCase();
    if (action !== "TOOL") return "";
    const tool = String(obj.tool || "").trim().toLowerCase();
    const input = obj.input;

    if (tool === "file_write") {
      const path = typeof input === "object" && input ? String(input.path || "").trim() : "";
      return path ? `已写入文件：${path}` : "已调用文件写入工具";
    }
    if (tool === "file_read") {
      const target = typeof input === "string" ? input.trim() : (typeof input === "object" && input ? String(input.path || input.file || "").trim() : "");
      return target ? `已读取文件：${target}` : "已调用文件读取工具";
    }
    if (tool === "web_search") {
      const query = typeof input === "string" ? input.trim() : "";
      return query ? `已执行网页搜索：${query}` : "已执行网页搜索";
    }
    if (tool === "web_read") {
      const url = typeof input === "string" ? input.trim() : "";
      return url ? `已读取网页内容：${url}` : "已读取网页内容";
    }
    return tool ? `已调用工具：${tool}` : "";
  };

  if (normalizedStage === "router") {
    const decisionMatch = cleaned.match(/decision=([A-Z]+)/i);
    const reasonMatch = cleaned.match(/reason=([^\s,]+)/i);
    const decision = decisionMatch ? decisionMatch[1].toUpperCase() : "";
    const reason = reasonMatch ? reasonMatch[1] : "";
    if (decision || reason) {
      return `Decision ${decision || "UNKNOWN"}${reason ? ` | ${reason}` : ""}`;
    }
    return cleaned;
  }

  if (normalizedStage === "planner") {
    const previewMatch = cleaned.match(/预览[:：]\s*([\s\S]+)$/u);
    if (!previewMatch) return cleaned;
    const beforePreview = cleaned.slice(0, previewMatch.index).trim().replace(/[；;:：]\s*$/, "");
    const previewRaw = previewMatch[1] || "";
    const items = previewRaw
      .split("|")
      .map((item) => item.trim())
      .filter(Boolean);
    const previewText = items.length ? items.map((item) => `- ${item}`).join("\n") : compactText(previewRaw, 400);
    return [beforePreview, "任务预览：", previewText].filter(Boolean).join("\n");
  }

  if (normalizedStage === "executor") {
    const normalized = cleaned
      .replace(/[|｜]/g, "\n")
      .replace(/[；;]\s*(?=(?:步骤|当前步骤|执行结果|结果)[:：])/g, "\n");
    const rawLines = normalized
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);

    let doneCount = "";
    const stepCandidates = [];
    const resultCandidates = [];

    for (const line of rawLines) {
      const plain = line.replace(/^\[[^\]]+\]\s*/u, "").trim();
      const doneMatch = plain.match(/^已执行步骤[:：]\s*(\d+)/u);
      if (doneMatch) {
        doneCount = doneMatch[1];
        continue;
      }

      const stepMatch = plain.match(/^(?:当前)?步骤[:：]\s*(.+)$/u);
      if (stepMatch) {
        stepCandidates.push(stepMatch[1].trim());
        continue;
      }

      const resultMatch = plain.match(/^(?:执行)?结果[:：]\s*(.+)$/u);
      if (resultMatch) {
        resultCandidates.push(resultMatch[1].trim());
        continue;
      }

      if (plain.startsWith("{") || plain.startsWith("[")) {
        resultCandidates.push(plain);
      }
    }

    const unique = (items) => {
      const seen = new Set();
      const out = [];
      for (const item of items) {
        const key = String(item || "").trim();
        if (!key || seen.has(key)) continue;
        seen.add(key);
        out.push(key);
      }
      return out;
    };

    let steps = unique(stepCandidates);
    const hasTextStep = steps.some((item) => !/^\d+$/.test(item));
    if (hasTextStep) {
      steps = steps.filter((item) => !/^\d+$/.test(item));
    }

    const humanResults = unique(
      resultCandidates.map((rawResult) => {
        const parsed = tryParseJson(rawResult);
        if (parsed) {
          const human = humanizeToolAction(parsed);
          if (human) return human;
        }
        return rawResult;
      })
    );

    const lines = [];
    if (doneCount) lines.push(`已执行步骤：${doneCount}`);
    if (steps.length > 0) lines.push(`当前步骤：${steps[0]}`);
    if (humanResults.length > 0) lines.push(`执行结果：${humanResults[0]}`);
    return lines.length ? lines.join("\n") : cleaned;
  }

  if (normalizedStage === "reviewer") {
    const passFail = cleaned.match(/\b(PASS|FAIL)\b/i);
    const retryMatch = cleaned.match(/retry[:=]\s*(\d+)/i);
    const retryCount = retryMatch ? Number(retryMatch[1]) : null;
    const cleanedNoRetry = cleaned.replace(/\(\s*retry[:=]\s*\d+\s*\)\s*$/i, "").trim();
    if (passFail) {
      const label = passFail[1].toUpperCase();
      const detail = cleanedNoRetry && cleanedNoRetry.toUpperCase() !== label ? cleanedNoRetry : "";
      if (label === "PASS") {
        const base = detail || "评审通过：该步骤已满足目标";
        return retryCount !== null ? `${base}（重试 ${retryCount} 次）` : base;
      }
      if (label === "FAIL") {
        const base = detail || "评审未通过：该步骤需要修订";
        return retryCount !== null ? `${base}（重试 ${retryCount} 次）` : base;
      }
    }
    const reviewLabel = passFail ? passFail[1].toUpperCase() : "REVIEW";
    const retryText = retryCount !== null ? `重试次数：${retryCount}` : "";
    return [reviewLabel, retryText, cleanedNoRetry || cleaned].filter(Boolean).join(" | ");
  }

  return cleaned;
}

function formatClockTime(ts) {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function parseExecutorDoneCount(message) {
  const text = String(message || "");
  const matchZh = text.match(/已执行步骤[:：]\s*(\d+)/u);
  if (matchZh) return Number(matchZh[1]);
  const matchEn = text.match(/executed\s*steps?\s*[:：]\s*(\d+)/iu);
  if (matchEn) return Number(matchEn[1]);
  return 0;
}

function formatUsageText(usage) {
  if (!usage || typeof usage !== "object") return "";
  const tokenUsed = Math.round(Number(usage.estimated_tokens || 0));
  const tokenBudget = Math.round(Number(usage.token_budget || 0));
  const costUsed = Number(usage.estimated_cost_usd || 0);
  const costBudget = Number(usage.cost_budget_usd || 0);
  if (!tokenBudget && !costBudget) return "";
  return `Tokens ${tokenUsed}/${tokenBudget} | Cost $${costUsed.toFixed(3)}/${costBudget.toFixed(2)}`;
}

function setTaskStatusCollapsed(collapsed, shouldPersist = true) {
  state.taskStatusCollapsed = Boolean(collapsed);
  if (taskStatusPane) {
    taskStatusPane.classList.toggle("collapsed", state.taskStatusCollapsed);
  }
  if (taskStatusToggleBtn) {
    taskStatusToggleBtn.textContent = state.taskStatusCollapsed ? "Expand" : "Collapse";
    taskStatusToggleBtn.setAttribute("aria-expanded", String(!state.taskStatusCollapsed));
  }
  if (shouldPersist) {
    persistState();
  }
  syncSidePaneLayout();
}

function setEvidenceCollapsed(collapsed, shouldPersist = true) {
  state.evidenceCollapsed = Boolean(collapsed);
  if (evidencePane) {
    evidencePane.classList.toggle("collapsed", state.evidenceCollapsed);
  }
  if (evidenceToggleBtn) {
    evidenceToggleBtn.textContent = state.evidenceCollapsed ? "Expand" : "Collapse";
    evidenceToggleBtn.setAttribute("aria-expanded", String(!state.evidenceCollapsed));
  }
  if (shouldPersist) {
    persistState();
  }
  syncSidePaneLayout();
}

function appendTaskStatusEvent(conversationId, payload) {
  const convo = state.conversations.find((c) => c.id === conversationId);
  if (!convo) return;

  const delta = typeof payload?.delta === "string" ? payload.delta.trim() : "";
  const meta = payload?.meta && typeof payload.meta === "object" ? payload.meta : {};
  const entry = {
    id: createId(),
    stage: meta.stage || "status",
    message: normalizeStatusMessage(meta.stage || "status", delta || meta.reason || ""),
    rawMessage: delta || meta.reason || "",
    model: meta.model || "",
    reason: meta.reason || "",
    stepIndex: Number.isFinite(Number(meta.step_index)) ? Number(meta.step_index) : null,
    totalSteps: Number.isFinite(Number(meta.total_steps)) ? Number(meta.total_steps) : null,
    retry: Number.isFinite(Number(meta.retry)) ? Number(meta.retry) : null,
    timestamp: Date.now(),
  };

  if (!Array.isArray(convo.taskEvents)) {
    convo.taskEvents = [];
  }
  convo.taskEvents.push(entry);
  if (convo.taskEvents.length > 200) {
    convo.taskEvents = convo.taskEvents.slice(-200);
  }
  const taskItems = normalizeTaskItems(meta.task_items);
  if (taskItems.length > 0) {
    convo.taskItems = taskItems;
  }

  convo.updatedAt = Date.now();
  persistState();
  renderTaskStatusPanel();
}

function setConversationUsage(conversationId, usage) {
  const convo = state.conversations.find((c) => c.id === conversationId);
  if (!convo) return;
  convo.usage = usage && typeof usage === "object" ? usage : null;
  persistState();
  renderTaskStatusPanel();
}

function renderTaskStatusPanel() {
  if (!taskStatusPane || !taskStatusList) return;
  if (state.chatMode !== "general" || !canShowSidePane()) {
    taskStatusPane.hidden = true;
    taskStatusList.innerHTML = "";
    if (taskStatusUsage) taskStatusUsage.textContent = "";
    updateTaskStatusActionButtons(false);
    syncSidePaneLayout();
    return;
  }
  const convo = getActiveConversation();
  const events = Array.isArray(convo?.taskEvents) ? convo.taskEvents : [];
  const taskItems = normalizeTaskItems(convo?.taskItems);
  const artifacts = normalizeArtifactItems(convo?.artifacts);
  const usage = convo?.usage || null;
  const hasContent =
    taskItems.length > 0 ||
    artifacts.length > 0 ||
    events.length > 0 ||
    Boolean(formatUsageText(usage)) ||
    state.isStreaming;

  taskStatusPane.hidden = !hasContent;
  if (!hasContent) {
    taskStatusList.innerHTML = "";
    if (taskStatusUsage) taskStatusUsage.textContent = "";
    setTaskStatusCollapsed(false, false);
    updateTaskStatusActionButtons(false);
    syncSidePaneLayout();
    return;
  }

  if (taskStatusUsage) {
    taskStatusUsage.textContent = formatUsageText(usage);
  }

  const latestEvent = events.length > 0 ? events[events.length - 1] : null;
  const latestStage = String(latestEvent?.stage || "").toLowerCase();
  const stageOrder = ["router", "planner", "executor", "reviewer", "summarizer"];
  const stageSeen = new Set(events.map((item) => String(item?.stage || "").toLowerCase()));

  const totalSteps = taskItems.length;
  let doneSteps = 0;
  for (const item of events) {
    if (String(item.stage || "").toLowerCase() === "executor") {
      doneSteps = Math.max(doneSteps, parseExecutorDoneCount(item.rawMessage || item.message));
    }
  }
  if (!doneSteps) {
    doneSteps = taskItems.filter((item) => item.status === "done").length;
  }
  if (totalSteps > 0 && doneSteps > totalSteps) {
    doneSteps = totalSteps;
  }
  const runningIndex =
    totalSteps > 0 && doneSteps < totalSteps && (latestStage === "executor" || latestStage === "reviewer")
      ? doneSteps
      : null;
  const runningItem = runningIndex !== null ? taskItems[runningIndex] : null;
  const progressPct = totalSteps > 0 ? Math.round((doneSteps / totalSteps) * 100) : 0;
  const summaryLabel = runningItem
    ? `Running: ${runningItem.title}`
    : doneSteps === totalSteps && totalSteps > 0
      ? "All tasks completed"
      : latestStage
        ? `${formatStageLabel(latestStage)} in progress`
        : "Ready";

  const stageFlow = `
    <div class="task-stage-flow">
      ${stageOrder
        .map((stage) => {
          const isActive = latestStage === stage;
          const isDone = stageSeen.has(stage) && !isActive;
          const cls = isActive ? "active" : isDone ? "done" : "pending";
          const marker = isActive ? "●" : isDone ? "✓" : "○";
          return `
            <div class="task-stage-pill ${cls}">
              <span class="task-stage-marker">${marker}</span>
              <span>${formatStageLabel(stage)}</span>
            </div>
          `;
        })
        .join("")}
    </div>
  `;

  const processEvents = events
    .filter((item) => ["router", "planner", "executor", "reviewer", "summarizer"].includes(String(item.stage || "").toLowerCase()))
    .slice(-4);
  const processList = processEvents.length
    ? `
      <div class="task-process-list">
        ${processEvents
          .map((item) => {
            const msg = String(item.message || "").trim();
            return `
              <div class="task-process-item stage-${escapeHtml(String(item.stage || "").toLowerCase())}">
                <span class="task-process-stage">${escapeHtml(formatStageLabel(item.stage || ""))}</span>
                <span class="task-process-text">${escapeHtml(msg)}</span>
              </div>
            `;
          })
          .join("")}
      </div>
    `
    : "";

  const summaryBlock = totalSteps > 0
    ? `
      <div class="task-status-summary">
        <div class="task-summary-top">
          <span><strong>${escapeHtml(summaryLabel)}</strong></span>
          <span>${doneSteps}/${totalSteps} steps</span>
        </div>
        ${stageFlow}
        <div class="task-progress-track">
          <div class="task-progress-fill" style="width: ${progressPct}%"></div>
        </div>
        ${processList}
      </div>
    `
    : state.isStreaming
      ? `
      <div class="task-status-summary">
        <div class="task-summary-top">
          <span><strong>Running...</strong></span>
          <span>--</span>
        </div>
      </div>
    `
    : "";

  const rows = taskItems
    .map((item, idx) => {
      let status = "pending";
      if (idx < doneSteps) {
        status = "done";
      } else if (runningIndex !== null && idx === runningIndex) {
        status = "running";
      } else if (item.status === "running" && runningIndex === null) {
        status = "running";
      }
      const symbol = status === "done" ? "✓" : status === "running" ? "●" : "○";
      const effectClass = status === "running" ? "shimmer pulse" : "";
      return `
        <div class="task-check-item status-${escapeHtml(status)} ${effectClass}">
          <span class="task-check-index">${idx + 1}</span>
          <span class="task-check-symbol">${symbol}</span>
          <span class="task-check-title">${escapeHtml(item.title || `Task ${idx + 1}`)}</span>
        </div>
      `;
    })
    .join("");

  const artifactBlock = artifacts.length
    ? `
      <div class="task-artifact-wrap">
        <div class="task-artifact-head">
          <span>Generated Files</span>
          <button class="runtime-artifact-btn ghost" type="button" data-artifact-bundle="1" data-session-id="${escapeHtml(convo?.id || "")}">Download ZIP</button>
        </div>
        <div class="task-artifact-grid artifact-icon-grid compact">${artifacts
          .map((item) => renderArtifactIconTile(convo?.id || "", item, true, true))
          .join("")}</div>
      </div>
    `
    : "";

  taskStatusList.innerHTML = `${summaryBlock}${rows}${artifactBlock}`;

  setTaskStatusCollapsed(state.taskStatusCollapsed, false);
  updateTaskStatusActionButtons(true);
  syncSidePaneLayout();
}

function hasRetryableRequest(convo) {
  const req = convo?.lastRequest;
  return Boolean(req && typeof req.text === "string" && req.text.trim());
}

function updateTaskStatusActionButtons(hasTaskContent) {
  const convo = getActiveConversation();
  const canRetry = !state.isStreaming && hasRetryableRequest(convo);
  const canStop = state.isStreaming && Boolean(activeStreamAbortController);

  if (taskRetryBtn) {
    taskRetryBtn.disabled = !canRetry;
    taskRetryBtn.setAttribute("aria-disabled", String(!canRetry));
  }
  if (taskStopBtn) {
    taskStopBtn.disabled = !canStop;
    taskStopBtn.setAttribute("aria-disabled", String(!canStop));
  }
  if (taskClearBtn) {
    taskClearBtn.disabled = !hasTaskContent;
    taskClearBtn.setAttribute("aria-disabled", String(!hasTaskContent));
  }
  if (taskStatusToggleBtn) {
    taskStatusToggleBtn.disabled = !hasTaskContent;
    taskStatusToggleBtn.setAttribute("aria-disabled", String(!hasTaskContent));
  }
}

function appendEvidenceItem(conversationId, item) {
  const convo = state.conversations.find((c) => c.id === conversationId);
  if (!convo || !item || typeof item !== "object") return;

  if (!Array.isArray(convo.evidenceItems)) {
    convo.evidenceItems = [];
  }
  convo.evidenceItems.push({
    id: createId(),
    stepIndex: Number.isFinite(Number(item.step_index)) ? Number(item.step_index) : null,
    stepTask: typeof item.step_task === "string" ? item.step_task : "",
    tool: typeof item.tool === "string" ? item.tool : "",
    source: typeof item.source === "string" ? item.source : "",
    status: typeof item.status === "string" ? item.status : "",
    reason: typeof item.reason === "string" ? item.reason : "",
    model: typeof item.model === "string" ? item.model : "",
    outputExcerpt: typeof item.output_excerpt === "string" ? item.output_excerpt : "",
    timestamp: Date.now(),
  });
  if (convo.evidenceItems.length > 200) {
    convo.evidenceItems = convo.evidenceItems.slice(-200);
  }
  convo.updatedAt = Date.now();
  persistState();
  renderEvidencePanel();
}

function renderEvidencePanel() {
  if (!evidencePane || !evidenceList) return;
  if (!canShowSidePane()) {
    evidencePane.hidden = true;
    evidenceList.innerHTML = "";
    setEvidenceCollapsed(false, false);
    if (evidenceToggleBtn) {
      evidenceToggleBtn.disabled = true;
      evidenceToggleBtn.setAttribute("aria-disabled", "true");
    }
    syncSidePaneLayout();
    return;
  }
  const convo = getActiveConversation();
  const items = Array.isArray(convo?.evidenceItems) ? convo.evidenceItems : [];
  const hasContent = items.length > 0;
  evidencePane.hidden = !hasContent;
  if (!hasContent) {
    evidenceList.innerHTML = "";
    setEvidenceCollapsed(false, false);
    if (evidenceToggleBtn) {
      evidenceToggleBtn.disabled = true;
      evidenceToggleBtn.setAttribute("aria-disabled", "true");
    }
    syncSidePaneLayout();
    return;
  }

  evidenceList.innerHTML = items
    .map((item) => {
      const stepLabel = item.stepIndex !== null ? `Step ${item.stepIndex + 1}` : "Step";
      const title = `${stepLabel} · ${item.tool || "tool"}`;
      const metaBits = [];
      if (item.status) metaBits.push(`status: ${escapeHtml(item.status)}`);
      if (item.model) metaBits.push(`model: ${escapeHtml(item.model)}`);
      if (item.reason) metaBits.push(`reason: ${escapeHtml(item.reason)}`);
      const metaText = metaBits.join(" | ");
      return `
        <div class="evidence-item">
          <div class="evidence-title-line">${escapeHtml(title)}</div>
          ${item.stepTask ? `<div class="evidence-meta">${escapeHtml(item.stepTask)}</div>` : ""}
          ${item.source ? `<div class="evidence-source">${escapeHtml(item.source)}</div>` : ""}
          ${metaText ? `<div class="evidence-meta">${metaText}</div>` : ""}
          ${item.outputExcerpt ? `<div class="evidence-excerpt">${escapeHtml(item.outputExcerpt)}</div>` : ""}
        </div>
      `;
    })
    .join("");

  if (evidenceToggleBtn) {
    evidenceToggleBtn.disabled = false;
    evidenceToggleBtn.setAttribute("aria-disabled", "false");
  }
  setEvidenceCollapsed(state.evidenceCollapsed, false);
  syncSidePaneLayout();
}

function syncSidePaneLayout() {
  if (!canShowSidePane()) {
    mainPanel.classList.remove("with-side-pane");
    return;
  }
  const hasTaskPane = taskStatusPane && !taskStatusPane.hidden;
  const hasHtmlPane = htmlPreviewPane && !htmlPreviewPane.hidden;
  const hasEvidencePane = evidencePane && !evidencePane.hidden;
  const showSidePane = Boolean(hasTaskPane || hasEvidencePane || hasHtmlPane);
  mainPanel.classList.toggle("with-side-pane", showSidePane);
}

function syncMainLayout() {
  const convo = getActiveConversation();
  const empty = isConversationEmpty(convo);
  mainPanel.classList.toggle("empty", empty);
  mainPanel.classList.toggle("has-messages", !empty);
  if (emptyState) {
    emptyState.style.display = empty ? "block" : "none";
  }
  syncSidePaneLayout();
}

function decodeHtmlEntities(text) {
  const textarea = document.createElement("textarea");
  textarea.innerHTML = text;
  return textarea.value;
}

function wrapHtmlFragment(fragment) {
  return `<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="UTF-8" />\n<meta name="viewport" content="width=device-width, initial-scale=1.0" />\n</head>\n<body>\n${fragment}\n</body>\n</html>`;
}

function normalizeHtmlForPreview(candidate) {
  if (typeof candidate !== "string") return null;
  const trimmed = decodeHtmlEntities(candidate).trim();
  if (!trimmed) return null;

  if (/<!doctype\s+html/i.test(trimmed) || /<html[\s>]/i.test(trimmed)) {
    return trimmed;
  }

  const hasBlockTag = /<(div|main|section|article|header|footer|table|form|ul|ol|p|h1|h2|h3|canvas|svg)(\s|>)/i.test(trimmed);
  const hasClosingTag = /<\/(div|main|section|article|header|footer|table|form|ul|ol|p|h1|h2|h3|canvas|svg)>/i.test(trimmed);
  if (hasBlockTag && hasClosingTag) {
    return wrapHtmlFragment(trimmed);
  }

  return null;
}

function extractHtmlFromText(text) {
  if (typeof text !== "string" || !text.trim()) return null;

  const fencedMatch = text.match(/```html\s*([\s\S]*?)```/i);
  if (fencedMatch && fencedMatch[1]) {
    const normalized = normalizeHtmlForPreview(fencedMatch[1]);
    if (normalized) return normalized;
  }

  const genericFenced = text.match(/```\s*([\s\S]*?)```/);
  if (genericFenced && genericFenced[1]) {
    const normalized = normalizeHtmlForPreview(genericFenced[1]);
    if (normalized) return normalized;
  }

  const fullDocMatch = text.match(/<!doctype\s+html[\s\S]*?<\/html>/i) || text.match(/<html[\s\S]*?<\/html>/i);
  if (fullDocMatch && fullDocMatch[0]) {
    const normalized = normalizeHtmlForPreview(fullDocMatch[0]);
    if (normalized) return normalized;
  }

  const fragmentStart = text.search(/<(div|main|section|article|header|footer|table|form|ul|ol|p|h1|h2|h3|canvas|svg)(\s|>)/i);
  if (fragmentStart >= 0) {
    const normalized = normalizeHtmlForPreview(text.slice(fragmentStart));
    if (normalized) return normalized;
  }

  return null;
}

function findLatestHtmlFromConversation(convo) {
  if (!convo || !Array.isArray(convo.messages)) return { html: null, sourceKey: "" };

  for (let i = convo.messages.length - 1; i >= 0; i -= 1) {
    const msg = convo.messages[i];
    if (msg?.role !== "assistant") continue;
    const html = extractHtmlFromText(msg.text);
    if (html) {
      return {
        html,
        sourceKey: `${convo.id}:${i}`,
      };
    }
  }

  return { html: null, sourceKey: "" };
}

function syncHtmlPreview(previewData) {
  if (!htmlPreviewPane || !htmlPreviewFrame) return;

  const htmlText = previewData?.html ?? null;
  const sourceKey = previewData?.sourceKey ?? "";
  const hasHtml = typeof htmlText === "string" && htmlText.trim().length > 0;
  const normalizedHtml = hasHtml ? htmlText.trim() : "";

  currentHtmlPreviewSourceKey = sourceKey;

  if (!hasHtml) {
    htmlPreviewDismissed = false;
    lastHtmlPreviewContent = "";
    dismissedHtmlPreviewSourceKey = "";
    currentHtmlPreviewSourceKey = "";
  } else if (normalizedHtml !== lastHtmlPreviewContent) {
    lastHtmlPreviewContent = normalizedHtml;
    if (sourceKey !== dismissedHtmlPreviewSourceKey) {
      htmlPreviewDismissed = false;
    }
  }

  const shouldShow = hasHtml && !htmlPreviewDismissed;
  htmlPreviewPane.hidden = !shouldShow;

  if (shouldShow) {
    htmlPreviewFrame.srcdoc = normalizedHtml;
  } else {
    htmlPreviewFrame.srcdoc = "";
  }
  syncSidePaneLayout();
}

function clearHtmlPreviewState() {
  htmlPreviewDismissed = false;
  lastHtmlPreviewContent = "";
  currentHtmlPreviewSourceKey = "";
  dismissedHtmlPreviewSourceKey = "";
  syncHtmlPreview({ html: null, sourceKey: "" });
}

async function resetConversationOnServer(conversationId) {
  if (!conversationId) return;

  try {
    await fetch("/api/chat/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: conversationId }),
    });
  } catch {
    // 忽略重置失败，不影响本地删除
  }
}

// 保存到本地：刷新页面后仍可恢复
function persistState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      conversations: state.conversations,
      activeConversationId: state.activeConversationId,
      chatCounter: state.chatCounter,
      chatMode: state.chatMode,
      generalModel: state.generalModel,
      webSearchEnabled: state.webSearchEnabled,
      webReadEnabled: state.webReadEnabled,
      fileAccessEnabled: state.fileAccessEnabled,
      edaAnalysisEnabled: state.edaAnalysisEnabled,
      sqlAnalysisEnabled: state.sqlAnalysisEnabled,
      isDbConnected: state.isDbConnected,
      taskStatusCollapsed: state.taskStatusCollapsed,
      evidenceCollapsed: state.evidenceCollapsed,
    })
  );
}

// 从本地恢复
function restoreState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return false;

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.conversations) || !parsed.conversations.length) return false;

    const normalizedConversations = parsed.conversations
      .map((convo, index) => {
        const messages = normalizeMessages(convo?.messages);
        const taskEvents = Array.isArray(convo?.taskEvents) ? convo.taskEvents : [];
        const taskItems = normalizeTaskItems(convo?.taskItems);
        const artifacts = normalizeArtifactItems(convo?.artifacts);
        const usage = convo?.usage && typeof convo.usage === "object" ? convo.usage : null;
        const evidenceItems = Array.isArray(convo?.evidenceItems) ? convo.evidenceItems : [];
        return {
          id: typeof convo?.id === "string" && convo.id ? convo.id : createId(),
          title:
            typeof convo?.title === "string" && convo.title.trim()
              ? convo.title
              : `New Chat ${index + 1}`,
          createdAt: Number(convo?.createdAt || Date.now()),
          updatedAt: Number(convo?.updatedAt || Date.now()),
          messages,
          taskEvents,
          taskItems,
          artifacts,
          usage,
          evidenceItems,
          lastRequest: normalizeLastRequest(convo?.lastRequest),
        };
      })
      .filter((convo) => convo.id);

    if (!normalizedConversations.length) return false;

    state.conversations = normalizedConversations;
    state.activeConversationId =
      normalizedConversations.find((c) => c.id === parsed.activeConversationId)?.id ||
      normalizedConversations[0].id;
    state.chatCounter = Number(parsed.chatCounter || normalizedConversations.length + 1);
    state.chatMode = parsed.chatMode === "expert" ? "expert" : "general";
    state.generalModel = isValidGeneralModel(parsed.generalModel) ? parsed.generalModel : "deepseek";
    state.webSearchEnabled = Boolean(parsed.webSearchEnabled);
    state.webReadEnabled = Boolean(parsed.webReadEnabled);
    state.fileAccessEnabled = Boolean(parsed.fileAccessEnabled);
    state.edaAnalysisEnabled = Boolean(parsed.edaAnalysisEnabled);
    state.sqlAnalysisEnabled = Boolean(parsed.sqlAnalysisEnabled);
    state.isDbConnected = Boolean(parsed.isDbConnected);
    state.taskStatusCollapsed = Boolean(parsed.taskStatusCollapsed);
    state.evidenceCollapsed = Boolean(parsed.evidenceCollapsed);
    if (state.chatMode !== "expert") {
      state.edaAnalysisEnabled = false;
      state.sqlAnalysisEnabled = false;
    }

    if (!Number.isFinite(state.chatCounter) || state.chatCounter < 1) {
      state.chatCounter = normalizedConversations.length + 1;
    }

    return true;
  } catch {
    return false;
  }
}

// 创建新会话
function createConversation() {
  cancelLastUserEditing();
  const id = createId();
  const convo = {
    id,
    title: `New Chat ${state.chatCounter}`,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
    taskEvents: [],
    taskItems: [],
    artifacts: [],
    evidenceItems: [],
    usage: null,
    lastRequest: null,
  };

  state.chatCounter += 1;
  state.conversations.unshift(convo);
  state.activeConversationId = id;
  clearHtmlPreviewState();

  persistState();
  renderAll();
}

// 切换会话
function switchConversation(conversationId) {
  if (!state.conversations.some((c) => c.id === conversationId)) return;
  cancelLastUserEditing();
  state.activeConversationId = conversationId;
  persistState();
  renderAll();
}

// 删除会话
function deleteConversation(conversationId) {
  const idx = state.conversations.findIndex((c) => c.id === conversationId);
  if (idx === -1) return;

  resetConversationOnServer(conversationId);

  state.conversations.splice(idx, 1);

  // 若删除的是当前会话，切到剩余第一条
  if (state.activeConversationId === conversationId) {
    cancelLastUserEditing();
    state.activeConversationId = state.conversations[0]?.id ?? null;
  }

  // 至少保留一个会话，避免界面空状态
  if (!state.conversations.length) {
    createConversation();
    return;
  }

  persistState();
  renderAll();
}

// 清除所有聊天记录
function clearAllHistory() {
  // 确认删除
  if (!confirm("确定要删除所有聊天记录吗？此操作无法撤销。")) {
    return;
  }

  // 清除所有会话对应的服务器端数据
  for (const convo of state.conversations) {
    resetConversationOnServer(convo.id);
  }

  // 重置state
  state.conversations = [];
  state.activeConversationId = null;
  state.chatCounter = 1;
  cancelLastUserEditing();

  // 创建一个新的空白会话
  createConversation();

  // 持久化并重新渲染
  persistState();
  renderAll();
}

// 渲染历史列表
function renderHistory() {
  historyList.innerHTML = "";

  for (const convo of state.conversations) {
    const row = document.createElement("div");
    row.className = "history-item-wrap";

    const btn = document.createElement("button");
    btn.className = `history-item${convo.id === state.activeConversationId ? " active" : ""}`;
    btn.textContent = convo.title;
    btn.title = convo.title;
    btn.addEventListener("click", () => switchConversation(convo.id));

    const delBtn = document.createElement("button");
    delBtn.className = "history-delete";
    delBtn.type = "button";
    delBtn.title = "删除聊天";
    delBtn.setAttribute("aria-label", "删除聊天");
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversation(convo.id);
    });

    row.appendChild(btn);
    row.appendChild(delBtn);
    historyList.appendChild(row);
  }
}

// 渲染当前会话消息
function renderMessages() {
  messageList.innerHTML = "";
  const convo = getActiveConversation();
  if (!convo) return;

  if (!Array.isArray(convo.messages)) {
    convo.messages = [];
  }

  if (isConversationEmpty(convo) && emptyState) {
    messageList.appendChild(emptyState);
    syncHtmlPreview({ html: null, sourceKey: "" });
    renderEvidencePanel();
    return;
  }

  const lastUserIndex = findLastUserMessageIndex(convo);
  for (let idx = 0; idx < convo.messages.length; idx += 1) {
    const msg = convo.messages[idx];
    const article = document.createElement("article");
    article.className = `msg ${msg.role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (msg.role === "assistant" && msg.pending) {
      bubble.classList.add("thinking");
      const pendingLabel = typeof msg.pendingLabel === "string" && msg.pendingLabel.trim() ? msg.pendingLabel.trim() : "In progress 🔍";
      bubble.innerHTML = `<span class="thinking-label loading-scan">${escapeHtml(pendingLabel)}</span>`;
    } else if (msg.role === "assistant") {
      bubble.classList.add("rich");
      const normalizedArtifacts = normalizeArtifactItems(msg.artifacts);
      const displayText = condenseAssistantTextForArtifacts(msg.text, normalizedArtifacts);
      const artifactHtml = renderAssistantArtifactCards(convo.id, normalizedArtifacts);
      bubble.innerHTML = `${renderAssistantMessageHtml(displayText)}${artifactHtml}`;
    } else {
      bubble.textContent = msg.text;
    }

    article.appendChild(bubble);
    if (msg.role === "user" && idx === lastUserIndex) {
      const editBtn = document.createElement("button");
      editBtn.className = "message-edit-btn";
      editBtn.type = "button";
      editBtn.textContent = state.isEditingLastUser ? "Cancel Edit" : "Edit";
      editBtn.disabled = state.isStreaming;
      editBtn.setAttribute("aria-label", "Edit last user message");
      editBtn.addEventListener("click", () => {
        if (state.isEditingLastUser) {
          cancelLastUserEditing();
          renderMessages();
          return;
        }
        beginLastUserEditing();
      });
      article.appendChild(editBtn);
    }

    if (msg.role === "assistant" && !msg.pending && typeof msg.edaReportThreadId === "string" && msg.edaReportThreadId.trim()) {
      const actions = document.createElement("div");
      actions.className = "eda-report-actions";
      const hasSuccessMark = typeof msg.text === "string" && msg.text.includes("✅");
      const source = msg.edaMessageSource;
      const isUploadMessage = source === "upload";
      const isChatMessage = source === "chat";
      const isEdaRefreshed = typeof msg.text === "string" && msg.text.toLowerCase().includes("eda refreshed");

      if (isUploadMessage || isEdaRefreshed) {
        const previewBtn = document.createElement("button");
        previewBtn.className = "eda-report-btn";
        previewBtn.type = "button";
        previewBtn.textContent = "Preview EDA report";
        previewBtn.addEventListener("click", async () => {
          try {
            await previewEdaReport(msg.edaReportThreadId);
          } catch (err) {
            appendAssistantSystemMessage(`EDA report preview failed: ${err?.message || "unknown error"}`);
          }
        });

        const downloadBtn = document.createElement("button");
        downloadBtn.className = "eda-report-btn";
        downloadBtn.type = "button";
        downloadBtn.textContent = "Download EDA report";
        downloadBtn.addEventListener("click", () => {
          downloadEdaReport(msg.edaReportThreadId);
        });

        actions.appendChild(previewBtn);
        actions.appendChild(downloadBtn);
      }

      if (isChatMessage && hasSuccessMark) {
        const downloadCsvBtn = document.createElement("button");
        downloadCsvBtn.className = "eda-report-btn";
        downloadCsvBtn.type = "button";
        downloadCsvBtn.textContent = "Download CSV";
        downloadCsvBtn.addEventListener("click", () => {
          downloadEdaCsv(msg.edaReportThreadId);
        });
        actions.appendChild(downloadCsvBtn);

        const sqlAnalysisBtn = document.createElement("button");
        sqlAnalysisBtn.className = "eda-report-btn"; // 维持样式一致
        sqlAnalysisBtn.type = "button";
        sqlAnalysisBtn.textContent = "Continue SQL";
        sqlAnalysisBtn.addEventListener("click", () => {
          state.sqlAnalysisEnabled = true;
          state.edaAnalysisEnabled = false;

          const messageContent = msg.text || "";
          const fileMatch = messageContent.match(/`(.+?\.csv)`/);
          if (messageContent.includes("✅ **Saved**") && fileMatch && fileMatch[1]) {
            const newFileName = fileMatch[1].split('/').pop();
            state.file_name = newFileName;
            state.uploadedFiles = [{
              id: "cleaned_" + Date.now(),
              name: newFileName,
              isCleaned: true
            }];
          }

          if (typeof syncModeUi === "function") syncModeUi();
          if (typeof persistState === "function") persistState();
          if (typeof renderAll === "function") renderAll();

          console.log("已切换至 SQL 分析模式");
        });
        actions.appendChild(sqlAnalysisBtn);
      }

      if (actions.childElementCount > 0) {
        article.appendChild(actions);
      }
    }

    messageList.appendChild(article);
  }

  messageList.scrollTop = messageList.scrollHeight;
  syncHtmlPreview(findLatestHtmlFromConversation(convo));
  queueMathTypeset();
}

// 统一刷新
function renderAll() {
  renderHistory();
  renderMessages();
  renderTaskStatusPanel();
  renderEvidencePanel();
  syncMainLayout();
  syncModeUi();
}

function initDbConnection() {

  const dbForm = document.getElementById("dbForm");

  if (!dbConnectBtn || !dbModal || !closeModal || !dbForm) return;

  dbConnectBtn.addEventListener("click", () => {
    dbModal.classList.add("active");
    dbModal.hidden = false;
  });

  closeModal.addEventListener("click", () => {
    dbModal.classList.remove("active");
    dbModal.hidden = true;
  });

  dbDisconnectBtn?.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to disconnect and clear the session cache？")) return;

    try {
      // 1. 断开物理连接
      const disconnectRes = await fetch("/api/db/disconnect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.activeConversationId })
      });

      // 2. 同时清理会话历史（可选，但强烈建议，防止字段名幻觉）
      await fetch("/api/chat/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.activeConversationId })
      });

      const data = await disconnectRes.json();
      if (data.ok) {
        alert("Disconnected and reset session.");
        state.isDbConnected = false;
        updateDbUI();
        // 如果你有消息列表，建议这里也清空一下 UI 上的对话记录
        // messages = []; renderMessages();
      }
    } catch (err) {
      console.error("Disconnect Error:", err);
    }
  });

  function updateDbUI() {
    const isExpert = state.chatMode === "expert";
    const dbSidebarSection = document.querySelector(".sidebar-footer");

    // 1. 如果不是专家模式，直接彻底隐藏
    if (!isExpert) {
      if (dbSidebarSection) dbSidebarSection.style.display = "none";
      return;
    }

    // 2. 如果是专家模式，显示容器并根据连接状态切换按钮
    if (dbSidebarSection) dbSidebarSection.style.display = "flex";

    if (state.isDbConnected) {
      dbConnectBtn.style.display = "none";
      dbDisconnectBtn.style.display = "flex";
    } else {
      dbConnectBtn.style.display = "flex";
      dbDisconnectBtn.style.display = "none";
    }
  }

  dbForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    // 1. 先从后端获取公钥 (也可以在页面加载时获取)
    const keyRes = await fetch("/api/db/public-key");
    const { public_key } = await keyRes.json();

    const formData = new FormData(dbForm);
    const rawInfo = Object.fromEntries(formData.entries());

    // 2. 使用 RSA 加密密码
    const encryptor = new JSEncrypt();
    encryptor.setPublicKey(public_key);
    const encryptedPassword = encryptor.encrypt(rawInfo.password);

    const payload = {
      session_id: state.activeConversationId,
      host: rawInfo.host,
      port: parseInt(rawInfo.port),
      user: rawInfo.user,
      password: encryptedPassword, // 传输的是密文
      database: rawInfo.database
    };

    // 调试用：在控制台打印发送的内容，方便你核对
    console.log("正在发送数据库连接请求:", payload);

    try {
      const response = await fetch("/api/db/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        alert("✅ Database connection succeeded！");
        state.isDbConnected = true; // 移动到这里：确保成功才置为 true
        updateDbUI();              // 移动到这里
        dbModal.hidden = true;
        dbModal.classList.remove("active");
        dbForm.reset();
      } else {
        state.isDbConnected = false; // 明确失败状态
        updateDbUI();
        const errDetail = await response.json();
        alert(`❌ Connection failed: ${JSON.stringify(errDetail.detail)}`);
      }
    } catch (err) {
      console.error("Network error:", err);
      alert("Network error, unable to connect to the server.");
    }
  });
}

// 3. 脚本入口：等待 DOM 加载完毕再启动
document.addEventListener("DOMContentLoaded", () => {
  closeArtifactPreviewModal();

  // Restore persisted state before binding interactive controls.
  const restored = restoreState();
  if (!restored) {
    createConversation();
  } else {
    renderAll();
  }

  ensureModelMenu();
  initDbConnection();
  autoResizeTextarea();

  window.addEventListener("resize", () => {
    renderTaskStatusPanel();
    renderEvidencePanel();
    syncSidePaneLayout();
  });
});

function appendAssistantSystemMessage(text, meta = {}) {
  if (typeof text !== "string" || !text.trim()) return;

  const convo = getActiveConversation();
  if (!convo) return;

  if (!Array.isArray(convo.messages)) {
    convo.messages = [];
  }

  convo.messages.push({
    role: "assistant",
    text: text.trim(),
    artifacts: [],
    edaReportThreadId: typeof meta.edaReportThreadId === "string" ? meta.edaReportThreadId : null,
    edaMessageSource: meta?.edaMessageSource === "upload" || meta?.edaMessageSource === "chat" ? meta.edaMessageSource : null,
  });
  convo.updatedAt = Date.now();
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  persistState();
  renderAll();
}

async function streamAssistantReply(conversationId, assistantMessage, userText, options = {}, requestMeta = {}) {
  const requestMode = requestMeta?.mode === "expert" ? "expert" : "general";
  const requestGeneralModel = isValidGeneralModel(requestMeta?.generalModel)
    ? requestMeta.generalModel
    : state.generalModel;

  const mergedProviderOptions = {
    ...(CHAT_BACKEND_CONFIG.provider_options || {}),
    ...options,
  };
  if (requestMode === "general") {
    mergedProviderOptions.general_model = requestGeneralModel;
  }

  // Use fetch stream + SSE framing to update the assistant message progressively.
  const resp = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: requestMeta?.abortSignal,
    body: JSON.stringify({
      session_id: conversationId,
      mode: requestMode,
      message: userText,
      provider: CHAT_BACKEND_CONFIG.provider || null,
      model: CHAT_BACKEND_CONFIG.model || null,
      provider_options: mergedProviderOptions,
      stream: true,
    }),
  });

  if (!resp.ok || !resp.body) {
    throw new Error(`Request failed: HTTP ${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  // 由于网络分片可能把一帧切断，buffer 用于拼接残缺数据。
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    // SSE 帧间隔是空行，因此按 \n\n 切帧。
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";

    for (const frame of frames) {
      const line = frame
        .split("\n")
        .find((item) => item.startsWith("data: "));
      if (!line) continue;

      let payload;
      try {
        payload = JSON.parse(line.slice(6));
      } catch {
        continue;
      }

      if (payload?.event === "token") {
        const tokenMeta = payload?.payload?.meta;
        if (tokenMeta?.kind === "status") {
          appendTaskStatusEvent(conversationId, payload?.payload || {});
          if (tokenMeta?.usage) {
            setConversationUsage(conversationId, tokenMeta.usage);
          }
          const runtime = ensureMessageRuntime(assistantMessage);
          runtime.stage = typeof tokenMeta.stage === "string" ? tokenMeta.stage : runtime.stage;
          runtime.capability = typeof tokenMeta.capability === "string" ? tokenMeta.capability : runtime.capability;
          const taskItems = normalizeTaskItems(tokenMeta.task_items);
          if (taskItems.length > 0) {
            runtime.taskItems = taskItems;
            const convo = state.conversations.find((item) => item.id === conversationId);
            if (convo) {
              convo.taskItems = taskItems;
            }
          }
          if (assistantMessage.pending) {
            assistantMessage.pendingLabel = buildPendingLabelFromMeta(tokenMeta);
            const active = getActiveConversation();
            if (active && active.id === conversationId) {
              renderMessages();
            }
          }
          continue;
        }
        if (tokenMeta?.kind === "progress") {
          if (tokenMeta?.usage) {
            setConversationUsage(conversationId, tokenMeta.usage);
          }
          const runtime = ensureMessageRuntime(assistantMessage);
          runtime.stage = typeof tokenMeta.stage === "string" ? tokenMeta.stage : runtime.stage;
          runtime.capability = typeof tokenMeta.capability === "string" ? tokenMeta.capability : runtime.capability;
          const taskItems = normalizeTaskItems(tokenMeta.task_items);
          if (taskItems.length > 0) {
            runtime.taskItems = taskItems;
            const convo = state.conversations.find((item) => item.id === conversationId);
            if (convo) {
              convo.taskItems = taskItems;
            }
          }
          const delta = typeof payload?.payload?.delta === "string" ? payload.payload.delta.trim() : "";
          if (delta) {
            runtime.progressLogs.push(delta);
            if (runtime.progressLogs.length > 50) {
              runtime.progressLogs = runtime.progressLogs.slice(-50);
            }
          }
          if (assistantMessage.pending) {
            assistantMessage.pendingLabel = buildPendingLabelFromMeta(tokenMeta);
            const active = getActiveConversation();
            if (active && active.id === conversationId) {
              renderMessages();
            }
          }
          continue;
        }
        if (tokenMeta?.kind === "evidence") {
          if (payload?.payload?.meta?.item) {
            appendEvidenceItem(conversationId, payload.payload.meta.item);
          }
          continue;
        }
        if (tokenMeta?.kind === "artifact") {
          const artifactItem = payload?.payload?.meta?.item;
          const convo = state.conversations.find((item) => item.id === conversationId);
          if (convo) {
            const conversationArtifacts = ensureConversationArtifacts(convo);
            upsertArtifactItem(conversationArtifacts, artifactItem);
          }
          const runtime = ensureMessageRuntime(assistantMessage);
          const runtimeArtifacts = Array.isArray(runtime.artifacts) ? runtime.artifacts : [];
          runtime.artifacts = upsertArtifactItem(runtimeArtifacts, artifactItem);
          assistantMessage.artifacts = upsertArtifactItem(normalizeArtifactItems(assistantMessage.artifacts), artifactItem);
          const active = getActiveConversation();
          if (active && active.id === conversationId) {
            renderMessages();
          }
          continue;
        }

        // token 事件为增量文本，持续追加到当前助手消息。
        const delta = payload?.payload?.delta;
        if (typeof delta === "string" && delta.length > 0) {
          assistantMessage.pending = false;
          assistantMessage.text += delta;
          const active = getActiveConversation();
          if (active && active.id === conversationId) {
            renderMessages();
          }
        }
        continue;
      }

      if (payload?.event === "final") {
        // final 事件是兜底完整文本，防止 token 丢失导致内容不完整。
        const finalText = payload?.payload?.text;
        if (typeof finalText === "string" && finalText.trim()) {
          assistantMessage.pending = false;
          if (!assistantMessage.text.trim()) {
            assistantMessage.text = finalText;
          }
          const active = getActiveConversation();
          if (active && active.id === conversationId) {
            renderMessages();
          }
        }
        if (payload?.payload?.usage) {
          setConversationUsage(conversationId, payload.payload.usage);
        }
        if (Array.isArray(payload?.payload?.artifacts)) {
          const normalizedArtifacts = normalizeArtifactItems(payload.payload.artifacts);
          const convo = state.conversations.find((item) => item.id === conversationId);
          if (convo) {
            convo.artifacts = normalizedArtifacts;
          }
          const runtime = ensureMessageRuntime(assistantMessage);
          runtime.artifacts = normalizedArtifacts;
          assistantMessage.artifacts = normalizedArtifacts;
        }
        if (typeof payload?.payload?.capability === "string") {
          const runtime = ensureMessageRuntime(assistantMessage);
          runtime.capability = payload.payload.capability;
        }
        continue;
      }

      if (payload?.event === "error") {
        // 统一将服务端错误抛出给 submitMessage 处理。
        throw new Error(payload?.payload?.message || "流式返回错误");
      }
    }
  }
}

// 发送消息：写入当前会话并通过后端流式获取回复
async function submitMessage(text, options = {}, uiOptions = {}) {
  const convo = getActiveConversation();
  if (!convo || state.isStreaming) return;

  const hideUserMessage = Boolean(uiOptions.hideUserMessage);
  const pendingLabel = typeof uiOptions.pendingLabel === "string" && uiOptions.pendingLabel.trim()
    ? uiOptions.pendingLabel
    : "Thinking · Routing";
  const requestMode = uiOptions.mode === "expert" ? "expert" : state.chatMode === "expert" ? "expert" : "general";
  const requestGeneralModel = isValidGeneralModel(uiOptions.generalModel)
    ? uiOptions.generalModel
    : state.generalModel;
  const providerOptions = options && typeof options === "object" ? { ...options } : {};

  if (!Array.isArray(convo.messages)) {
    convo.messages = [];
  }

  if (!hideUserMessage) {
    convo.messages.push({ role: "user", text });
  }
  const assistantMessage = {
    role: "assistant",
    text: "",
    pending: true,
    pendingLabel,
    runtime: createRuntimeState(),
    artifacts: [],
    edaReportThreadId: options?.eda_analysis ? convo.id : null,
    edaMessageSource: options?.eda_analysis ? "chat" : null,
  };
  convo.messages.push(assistantMessage);
  convo.updatedAt = Date.now();

  // 如果还是默认标题，用第一条用户消息更新标题
if (!hideUserMessage && /^New Chat\s\d+$/.test(convo.title)) {
    convo.title = shortTitle(text);
  }

  // 最近更新的会话放到顶部
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  convo.lastRequest = {
    text,
    providerOptions,
    mode: requestMode,
    generalModel: requestGeneralModel,
    timestamp: Date.now(),
  };

  const streamController = new AbortController();
  activeStreamAbortController = streamController;
  state.isStreaming = true;
  composerInput.disabled = true;
  persistState();
  renderAll();

  try {
    await streamAssistantReply(convo.id, assistantMessage, text, providerOptions, {
      mode: requestMode,
      generalModel: requestGeneralModel,
      abortSignal: streamController.signal,
    });
    if (requestMode === "general") {
      await refreshConversationArtifacts(convo.id, assistantMessage);
    }
    assistantMessage.pending = false;
    if (!assistantMessage.text.trim()) {
      assistantMessage.text = "模型未返回文本内容。";
    }

    const isExpertAnalysis = requestMode === "expert" && (state.edaAnalysisEnabled || state.sqlAnalysisEnabled);

    if (!isExpertAnalysis) {
      state.uploadedFiles = [];
    } else {
      const match = assistantMessage.text.match(/`(.+?\.csv)`/);
      if (match && match[1]) {
        state.file_name = match[1].split('/').pop();
      }
    }

    syncModeUi();
  } catch (err) {
    assistantMessage.pending = false;
    if (err?.name === "AbortError") {
      assistantMessage.text = "Generation stopped by user.";
    } else {
      assistantMessage.text = `Request failed: ${err?.message || "Unknown error"}`;
    }
  } finally {
    if (activeStreamAbortController === streamController) {
      activeStreamAbortController = null;
    }
    convo.updatedAt = Date.now();
    state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
    state.isStreaming = false;
    composerInput.disabled = false;
    persistState();
    renderAll();
    composerInput.focus();
  }
}

async function rerunWithEditedLastUser(editedText, options = {}) {
  const convo = getActiveConversation();
  if (!convo || state.isStreaming) return;

  const lastUserIndex = findLastUserMessageIndex(convo);
  if (lastUserIndex < 0) {
    state.isEditingLastUser = false;
    await submitMessage(editedText, options);
    return;
  }

  const response = await fetch("/api/chat/rewrite-last-user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: convo.id }),
  });
  if (!response.ok) {
    throw new Error(`编辑重跑失败：HTTP ${response.status}`);
  }

  convo.messages = convo.messages.slice(0, lastUserIndex);
  convo.taskEvents = [];
  convo.taskItems = [];
  convo.artifacts = [];
  convo.evidenceItems = [];
  convo.updatedAt = Date.now();
  state.isEditingLastUser = false;
  persistState();
  renderAll();

  await submitMessage(editedText, options, { pendingLabel: "思考中 · 已基于编辑内容重跑" });
}

// Enter 发送，Shift+Enter 换行
composerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    // 输入法候选确认阶段（中文/日文等）按 Enter：不提交
    // isComposing: 现代浏览器标准标志
    // keyCode 229: 输入法组合输入常见特殊码（兼容兜底）
    if (e.isComposing || e.keyCode === 229) {
      e.preventDefault();
      return;
    }

    e.preventDefault();
    composerForm.requestSubmit();
  }
});

// 输入时动态调节高度
composerInput.addEventListener("input", autoResizeTextarea);

// 新聊天按钮
newChatBtn.addEventListener("click", async () => {
  createConversation();
  triggerGlobalCleanup();
  // Keep frontend upload state isolated for the new conversation.
  state.uploadedFiles = [];
  state.file_name = "";
  persistState();
  renderAll();
});

// 清除所有聊天记录按钮
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
clearHistoryBtn?.addEventListener("click", clearAllHistory);

modelSwitch?.addEventListener("click", () => {
  ensureModelMenu();
  if (!modelMenuEl) return;

  if (modelMenuEl.hidden) {
    syncModelMenuSelection();
    openModelMenu();
  } else {
    closeModelMenu();
  }
});

// 点击后切换到专家模式，并更新欢迎语。
taskChipBtn?.addEventListener("click", () => {
  state.chatMode = state.chatMode === "expert" ? "general" : "expert";

  if (state.chatMode !== "expert") {
    state.edaAnalysisEnabled = false;
    state.sqlAnalysisEnabled = false;
  } else {
    state.generalCapabilityPreset = "";
  }

  renderAll();
  persistState();
});

function setGeneralCapabilityPreset(capability) {
  if (state.chatMode !== "general") {
    return;
  }
  if (state.generalCapabilityPreset === capability) {
    state.generalCapabilityPreset = "";
  } else {
    state.generalCapabilityPreset = capability;
  }
  renderAll();
  persistState();
}

websiteBuilderBtn?.addEventListener("click", () => {
  setGeneralCapabilityPreset("website_builder");
  composerInput.focus();
});

slideBuilderBtn?.addEventListener("click", () => {
  setGeneralCapabilityPreset("slide_builder");
  composerInput.focus();
});

webSearchToggleBtn?.addEventListener("click", () => {
  if (state.chatMode !== "general") {
    return;
  }
  state.webSearchEnabled = !state.webSearchEnabled;
  renderAll();
  persistState();
});

webReadToggleBtn?.addEventListener("click", () => {
  if (state.chatMode !== "general") {
    return;
  }
  state.webReadEnabled = !state.webReadEnabled;
  renderAll();
  persistState();
});

fileAccessToggleBtn?.addEventListener("click", () => {
  if (state.chatMode !== "general") {
    return;
  }
  state.fileAccessEnabled = !state.fileAccessEnabled;
  renderAll();
  persistState();
});

edaAnalysisToggleBtn?.addEventListener("click", () => {
  if (state.chatMode !== "expert") {
    alert("Please enable Expert Mode before using EDA Analysis.");
    return;
  }
  state.edaAnalysisEnabled = !state.edaAnalysisEnabled;
  renderAll();
  persistState();
});

fileUploadBtn?.addEventListener("click", () => {
  const canUpload = state.chatMode === "expert" && (state.edaAnalysisEnabled || state.sqlAnalysisEnabled);
  if (!canUpload) {
    alert("Please enable EDA or SQL analysis first before uploading files.");
    return;
  }
  fileUploadInput?.click();
});

fileUploadInput?.addEventListener("change", async () => {
  const selectedFiles = Array.from(fileUploadInput.files || []);
  if (!selectedFiles.length) return;

  const useEdaUpload = state.chatMode === "expert" && state.edaAnalysisEnabled;
  const uploadEdaSummaries = [];
  const uploadEdaFailures = [];
  let uploadEdaThreadId = "";

  const activeConvo = getActiveConversation();
  let edaPendingMessage = null;

  if (useEdaUpload && activeConvo) {
    if (!Array.isArray(activeConvo.messages)) {
      activeConvo.messages = [];
    }
    edaPendingMessage = {
      role: "assistant",
      text: "",
      pending: true,
      pendingLabel: "EDA analysis in progress 🔍",
      runtime: createRuntimeState(),
      artifacts: [],
    };
    activeConvo.messages.push(edaPendingMessage);
    activeConvo.updatedAt = Date.now();
    state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
    state.isStreaming = true;
    composerInput.disabled = true;
    persistState();
    renderAll();
  }

  const existingIds = new Set(state.uploadedFiles.map((item) => item.id));
  for (const file of selectedFiles) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const active = getActiveConversation();
      const sessionId = active?.id || "";
      const useEda = useEdaUpload;
      const query = new URLSearchParams();
      if (useEda) {
        query.set("use_eda", "true");
        if (sessionId) {
          query.set("session_id", sessionId);
        }
      }

      const uploadUrl = query.toString() ? `/api/upload?${query.toString()}` : "/api/upload";

      const resp = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) throw new Error("Upload failed");

      const result = await resp.json();
      const returnedName = typeof result?.file_name === "string" && result.file_name ? result.file_name : file.name;
      const id = fileIdentity(file);
      if (existingIds.has(id)) continue;

      state.uploadedFiles.push({ id, name: returnedName });
      existingIds.add(id);

      if (useEda && result && typeof result === "object") {
        const edaPayload = result.eda;
        const threadId = typeof result?.eda_thread_id === "string" && result.eda_thread_id ? result.eda_thread_id : sessionId;
        if (threadId) {
          uploadEdaThreadId = threadId;
        }
        const messageText = typeof edaPayload?.message === "string" ? edaPayload.message.trim() : "";

        if (messageText) {
          uploadEdaSummaries.push(`【${returnedName}】\n${messageText}`);
        } else {
          uploadEdaFailures.push(`【${returnedName}】Upload parsing failed, please try again.`);
        }
      }
    } catch (err) {
      console.error("Upload failed:", err);
      if (useEdaUpload) {
        uploadEdaFailures.push(`【${file.name}】Upload failed, please try again.`);
      }
    }
  }

  if (fileUploadInput) {
    fileUploadInput.value = "";
  }

  syncModeUi();

  if (useEdaUpload && edaPendingMessage) {
    edaPendingMessage.pending = false;
    edaPendingMessage.pendingLabel = "";

    if (uploadEdaSummaries.length > 0) {
      edaPendingMessage.text = uploadEdaSummaries.join("\n\n");
      if (uploadEdaThreadId) {
        edaPendingMessage.edaReportThreadId = uploadEdaThreadId;
      }
      edaPendingMessage.edaMessageSource = "upload";
    } else if (uploadEdaFailures.length > 0) {
      edaPendingMessage.text = uploadEdaFailures.join("\n");
    } else {
      edaPendingMessage.text = "upload失败或解析失败，请重试。";
    }

    if (activeConvo) {
      activeConvo.updatedAt = Date.now();
    }
    state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
    state.isStreaming = false;
    composerInput.disabled = false;
    persistState();
    renderAll();
    composerInput.focus();
  } else if (uploadEdaSummaries.length > 0) {
    appendAssistantSystemMessage(uploadEdaSummaries.join("\n\n"));
  }
});

uploadStatus?.addEventListener("click", (e) => {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return;

  const removeBtn = target.closest(".upload-file-remove");
  if (!removeBtn) return;

  const fileId = removeBtn.getAttribute("data-file-id");
  if (!fileId) return;

  state.uploadedFiles = state.uploadedFiles.filter((item) => item.id !== fileId);
  syncModeUi();
});

sqlAnalysisToggleBtn?.addEventListener("click", () => {
  if (state.chatMode !== "expert") {
    alert("Please enable Expert Mode before using SQL Analysis.");
    return;
  }
  state.sqlAnalysisEnabled = !state.sqlAnalysisEnabled;
  if (state.sqlAnalysisEnabled) {
    state.edaAnalysisEnabled = false;
  }
  renderAll();
  persistState();
});

htmlPreviewCloseBtn?.addEventListener("click", () => {
  htmlPreviewDismissed = true;
  dismissedHtmlPreviewSourceKey = currentHtmlPreviewSourceKey;
  syncHtmlPreview({ html: lastHtmlPreviewContent, sourceKey: currentHtmlPreviewSourceKey });
});

artifactPreviewCloseBtn?.addEventListener("click", () => {
  closeArtifactPreviewModal();
});

artifactPreviewModal?.addEventListener("click", (e) => {
  if (e.target === artifactPreviewModal) {
    closeArtifactPreviewModal();
  }
});

artifactPreviewDownloadBtn?.addEventListener("click", () => {
  if (!activeArtifactPreview) return;
  downloadArtifact(activeArtifactPreview.sessionId, activeArtifactPreview.artifactPath);
});

messageList?.addEventListener("click", async (e) => {
  await handleArtifactActionClick(e.target);
});

taskStatusPane?.addEventListener("click", async (e) => {
  await handleArtifactActionClick(e.target);
});

taskRetryBtn?.addEventListener("click", () => {
  if (state.isStreaming) return;
  const convo = getActiveConversation();
  const lastRequest = convo?.lastRequest;
  if (!lastRequest || typeof lastRequest.text !== "string" || !lastRequest.text.trim()) {
    return;
  }

  submitMessage(lastRequest.text, lastRequest.providerOptions || {}, {
    hideUserMessage: true,
    pendingLabel: "Retrying...",
    mode: lastRequest.mode,
    generalModel: lastRequest.generalModel,
  });
});

taskStopBtn?.addEventListener("click", () => {
  if (!activeStreamAbortController) return;
  activeStreamAbortController.abort();
});

taskClearBtn?.addEventListener("click", () => {
  const convo = getActiveConversation();
  if (!convo) return;
  convo.taskEvents = [];
  convo.taskItems = [];
  convo.usage = null;
  convo.updatedAt = Date.now();
  persistState();
  renderAll();
});

taskStatusToggleBtn?.addEventListener("click", () => {
  setTaskStatusCollapsed(!state.taskStatusCollapsed);
  renderTaskStatusPanel();
});

evidenceToggleBtn?.addEventListener("click", () => {
  setEvidenceCollapsed(!state.evidenceCollapsed);
  renderEvidencePanel();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && artifactPreviewModal && !artifactPreviewModal.hidden) {
    closeArtifactPreviewModal();
  }
});

// 提交消息
composerForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = composerInput.value.trim();
  if (!text || state.isStreaming) return;

  const isExpertMode = state.chatMode === "expert";
  const dynamicOptions = {
    eda_analysis: isExpertMode && !!state.edaAnalysisEnabled,
    sql_analysis: isExpertMode && !!state.sqlAnalysisEnabled,
    general_web_search: state.chatMode === "general" && !!state.webSearchEnabled,
    general_web_read: state.chatMode === "general" && !!state.webReadEnabled,
    general_file_access: state.chatMode === "general" && !!state.fileAccessEnabled,
    general_capability:
      state.chatMode === "general" &&
      (state.generalCapabilityPreset === "website_builder" || state.generalCapabilityPreset === "slide_builder")
        ? state.generalCapabilityPreset
        : null,
    file_name: state.uploadedFiles.map((item) => item.name).join(",") || null
  };

  try {
    const submittedText = text;
    composerInput.value = "";
    if (typeof autoResizeTextarea === "function") autoResizeTextarea();
    if (state.isEditingLastUser) {
      await rerunWithEditedLastUser(submittedText, dynamicOptions);
    } else {
      await submitMessage(submittedText, dynamicOptions);
    }
  } catch (err) {
    appendAssistantSystemMessage(`Request failed: ${err?.message || "Unknown error"}`);
  }

});

// Initialization is handled in DOMContentLoaded above.
