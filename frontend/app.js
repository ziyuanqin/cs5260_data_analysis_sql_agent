// 前端主控脚本：负责会话状态管理、消息渲染、SSE 流式接收。
const STORAGE_KEY = "agent_ui_chat_sessions_v1";

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
const taskChipBtn = document.getElementById("taskChipBtn");
const fileUploadBtn = document.getElementById("fileUploadBtn");
const fileUploadInput = document.getElementById("fileUploadInput");
const uploadStatus = document.getElementById("uploadStatus");
const sqlAnalysisToggleBtn = document.getElementById("sqlAnalysisToggleBtn");
const modelSwitch = document.querySelector(".model-switch");

// 可选后端配置：可在浏览器控制台设置 window.__CHAT_BACKEND_CONFIG__ 动态切换。
// 例如：window.__CHAT_BACKEND_CONFIG__ = { provider: "local_http", model: "qwen2.5:7b" }
const CHAT_BACKEND_CONFIG = window.__CHAT_BACKEND_CONFIG__ || {};
let htmlPreviewDismissed = false;
let lastHtmlPreviewContent = "";
let currentHtmlPreviewSourceKey = "";
let dismissedHtmlPreviewSourceKey = "";

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
  uploadedFileName: "",
  uploadedFile: null,
  sqlAnalysisEnabled: false,
};

function getModeGreetingText() {
  return state.chatMode === "expert" ? "你好，我是专家模式" : "你好，我是通用模式";
}

function syncModeUi() {
  const isExpert = state.chatMode === "expert";

  if (modelSwitch) {
    modelSwitch.textContent = isExpert ? "专家模式▾" : "通用模式▾";
  }

  if (emptyTitle) {
    emptyTitle.textContent = getModeGreetingText();
  }

  if (taskChipBtn) {
    taskChipBtn.classList.toggle("active", isExpert);
    taskChipBtn.textContent = isExpert ? "★ 专家模式" : "✶ 专家模式";
    taskChipBtn.setAttribute("aria-pressed", String(isExpert));
    taskChipBtn.title = isExpert ? "点击切回通用模式" : "点击切换到专家模式";
  }

  if (fileUploadBtn) {
    const hasUpload = Boolean(state.uploadedFileName);
    fileUploadBtn.classList.toggle("active", hasUpload);
    fileUploadBtn.setAttribute("aria-pressed", String(hasUpload));
    fileUploadBtn.title = hasUpload ? "点击清空已上传文件" : "点击上传文件";
  }

  if (uploadStatus) {
    const hasUpload = Boolean(state.uploadedFileName);
    uploadStatus.hidden = !hasUpload;
    uploadStatus.textContent = hasUpload ? `已上传: ${state.uploadedFileName}` : "";
  }

  if (sqlAnalysisToggleBtn) {
    sqlAnalysisToggleBtn.classList.toggle("active", state.sqlAnalysisEnabled);
    sqlAnalysisToggleBtn.setAttribute("aria-pressed", String(state.sqlAnalysisEnabled));
    sqlAnalysisToggleBtn.title = state.sqlAnalysisEnabled ? "点击关闭SQL分析" : "点击开启SQL分析";
  }
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
    }))
    .filter((msg) => msg.text.trim().length > 0)
    .filter((msg) => !(msg.role === "assistant" && isLegacyWelcomeText(msg.text)));
}

function isLegacyWelcomeText(text) {
  if (typeof text !== "string") return false;
  const normalized = text.replaceAll("～", "~").trim();
  return normalized === "你好~可以开始新对话了";
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

function renderAssistantMessageHtml(text) {
  // 把助手纯文本按“普通段落 + Markdown 表格”转换成 HTML。
  const lines = text.split(/\r?\n/);
  const parts = [];
  let idx = 0;

  while (idx < lines.length) {
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

      const headHtml = normalizedHeader.map((cell) => `<th>${escapeHtml(cell)}</th>`).join("");
      const bodyHtml = normalizedRows
        .map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`)
        .join("");

      parts.push(`<div class="table-wrap"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`);
      continue;
    }

    const textBuffer = [];
    while (idx < lines.length) {
      const maybeHeader = parseMarkdownTable(lines[idx]);
      const maybeSeparator = lines[idx + 1] || "";
      if (maybeHeader && isTableSeparatorLine(maybeSeparator)) break;
      textBuffer.push(lines[idx]);
      idx += 1;
    }

    const paragraph = textBuffer.join("\n").trim();
    if (paragraph) {
      parts.push(`<p>${escapeHtml(paragraph).replaceAll("\n", "<br>")}</p>`);
    }
  }

  return parts.join("") || `<p>${escapeHtml(text)}</p>`;
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

function isConversationEmpty(convo) {
  if (!convo || !Array.isArray(convo.messages)) return true;
  return convo.messages.length === 0;
}

function syncMainLayout() {
  const convo = getActiveConversation();
  const empty = isConversationEmpty(convo);
  mainPanel.classList.toggle("empty", empty);
  mainPanel.classList.toggle("has-messages", !empty);
  if (emptyState) {
    emptyState.style.display = empty ? "block" : "none";
  }
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
  mainPanel.classList.toggle("with-html-preview", shouldShow);

  if (shouldShow) {
    htmlPreviewFrame.srcdoc = normalizedHtml;
  } else {
    htmlPreviewFrame.srcdoc = "";
  }
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
      sqlAnalysisEnabled: state.sqlAnalysisEnabled,
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
        return {
          id: typeof convo?.id === "string" && convo.id ? convo.id : createId(),
          title:
            typeof convo?.title === "string" && convo.title.trim()
              ? convo.title
              : `新聊天 ${index + 1}`,
          createdAt: Number(convo?.createdAt || Date.now()),
          updatedAt: Number(convo?.updatedAt || Date.now()),
          messages,
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
    state.sqlAnalysisEnabled = Boolean(parsed.sqlAnalysisEnabled);

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
  const id = createId();
  const convo = {
    id,
    title: `新聊天 ${state.chatCounter}`,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
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
    return;
  }

  for (const msg of convo.messages) {
    const article = document.createElement("article");
    article.className = `msg ${msg.role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (msg.role === "assistant") {
      bubble.classList.add("rich");
      bubble.innerHTML = renderAssistantMessageHtml(msg.text);
    } else {
      bubble.textContent = msg.text;
    }

    article.appendChild(bubble);
    messageList.appendChild(article);
  }

  messageList.scrollTop = messageList.scrollHeight;
  syncHtmlPreview(findLatestHtmlFromConversation(convo));
}

// 统一刷新
function renderAll() {
  renderHistory();
  renderMessages();
  syncMainLayout();
  syncModeUi();
}

async function streamAssistantReply(conversationId, assistantMessage, userText, options = {}) {
  // 通过 fetch 获取 ReadableStream，按 SSE 帧实时读取 token。
  const resp = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: conversationId,
      mode: state.chatMode,
      message: userText,
      provider: CHAT_BACKEND_CONFIG.provider || null,
      model: CHAT_BACKEND_CONFIG.model || null,
      provider_options: {
        ...(CHAT_BACKEND_CONFIG.provider_options || {}),
        ...options
      },
      stream: true,
    }),
  });

  if (!resp.ok || !resp.body) {
    throw new Error(`请求失败：HTTP ${resp.status}`);
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
        // token 事件为增量文本，持续追加到当前助手消息。
        const delta = payload?.payload?.delta;
        if (typeof delta === "string" && delta.length > 0) {
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
          if (!assistantMessage.text.trim()) {
            assistantMessage.text = finalText;
          }
          const active = getActiveConversation();
          if (active && active.id === conversationId) {
            renderMessages();
          }
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
async function submitMessage(text, options = {}) {
  const convo = getActiveConversation();
  if (!convo || state.isStreaming) return;

  if (!Array.isArray(convo.messages)) {
    convo.messages = [];
  }

  convo.messages.push({ role: "user", text });
  const assistantMessage = { role: "assistant", text: "" };
  convo.messages.push(assistantMessage);
  convo.updatedAt = Date.now();

  // 如果还是默认标题，用第一条用户消息更新标题
  if (/^新聊天\s\d+$/.test(convo.title)) {
    convo.title = shortTitle(text);
  }

  // 最近更新的会话放到顶部
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);

  state.isStreaming = true;
  composerInput.disabled = true;
  persistState();
  renderAll();

  try {
    await streamAssistantReply(convo.id, assistantMessage, text, options);
    if (!assistantMessage.text.trim()) {
      assistantMessage.text = "模型未返回文本内容。";
    }
  } catch (err) {
    assistantMessage.text = `请求失败：${err?.message || "未知错误"}`;
  } finally {
    convo.updatedAt = Date.now();
    state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
    state.isStreaming = false;
    composerInput.disabled = false;
    persistState();
    renderAll();
    composerInput.focus();
  }
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
newChatBtn.addEventListener("click", createConversation);

// 点击后切换到专家模式，并更新欢迎语。
taskChipBtn?.addEventListener("click", () => {
  state.chatMode = state.chatMode === "expert" ? "general" : "expert";
  if (state.chatMode !== "expert") {
    state.sqlAnalysisEnabled = false;
  }
  syncModeUi();
  persistState();
});

fileUploadBtn?.addEventListener("click", () => {
  if (state.uploadedFileName) {
    state.uploadedFileName = "";
    state.uploadedFile = null;
    if (fileUploadInput) {
      fileUploadInput.value = "";
    }
    syncModeUi();
    return;
  }

  fileUploadInput?.click();
});

fileUploadInput?.addEventListener("change", async () => {
  const file = fileUploadInput.files?.[0] || null;
  if (!file) return;

  // --- 新增：发送文件到后端 ---
  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) throw new Error("上传失败");

    const result = await resp.json();
    console.log("Upload response:", result);

    // 适配后端返回的 file_name 字段
    if (result && result.file_name) {
      state.uploadedFile = file;
      state.uploadedFileName = result.file_name;
      syncModeUi();
      alert("上传成功！");
    } else {
      throw new Error("后端返回格式不正确");
    }
  } catch (err) {
    console.error("上传明细:", err);
    alert("文件上传失败: " + err.message);
  }
});

sqlAnalysisToggleBtn?.addEventListener("click", () => {
  if (state.chatMode !== "expert") {
    alert("请先开启『专家模式』后再使用 SQL 分析功能。");
    return;
  }
  state.sqlAnalysisEnabled = !state.sqlAnalysisEnabled;
  syncModeUi();
  persistState();
});

htmlPreviewCloseBtn?.addEventListener("click", () => {
  htmlPreviewDismissed = true;
  dismissedHtmlPreviewSourceKey = currentHtmlPreviewSourceKey;
  syncHtmlPreview({ html: lastHtmlPreviewContent, sourceKey: currentHtmlPreviewSourceKey });
});

// 提交消息
composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = composerInput.value.trim();
  if (!text || state.isStreaming) return;

  const dynamicOptions = {
    sql_analysis: !!state.sqlAnalysisEnabled,
    file_name: state.uploadedFileName || null
  };

  submitMessage(text, dynamicOptions);
  composerInput.value = "";
  if (typeof autoResizeTextarea === 'function') autoResizeTextarea();

});

// 初始化：优先恢复本地历史，否则新建一个会话
clearHtmlPreviewState();
if (!restoreState()) {
  createConversation();
} else {
  renderAll();
}

autoResizeTextarea();
