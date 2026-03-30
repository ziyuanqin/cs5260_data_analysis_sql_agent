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
const edaAnalysisToggleBtn = document.getElementById("edaAnalysisToggleBtn");
const sqlAnalysisToggleBtn = document.getElementById("sqlAnalysisToggleBtn");
const modelSwitch = document.querySelector(".model-switch");

const GENERAL_MODELS = ["deepseek", "openai"];

//Database Connection
const dbConnectBtn = document.getElementById("dbConnectBtn");
const dbModal = document.getElementById("dbModal");
const closeModal = document.getElementById("closeModal");
const dbForm = document.getElementById("dbForm");
const dbDisconnectBtn = document.getElementById("dbDisconnectBtn");

// 可选后端配置：可在浏览器控制台设置 window.__CHAT_BACKEND_CONFIG__ 动态切换。
// 例如：window.__CHAT_BACKEND_CONFIG__ = { provider: "local_http", model: "qwen2.5:7b" }
const CHAT_BACKEND_CONFIG = window.__CHAT_BACKEND_CONFIG__ || {};
let htmlPreviewDismissed = false;
let lastHtmlPreviewContent = "";
let currentHtmlPreviewSourceKey = "";
let dismissedHtmlPreviewSourceKey = "";
let mathTypesetTimer = null;

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
  uploadedFiles: [],
  edaAnalysisEnabled: false,
  sqlAnalysisEnabled: false,
};

let modelMenuEl = null;

function getGeneralModelText() {
  const selected = GENERAL_MODELS.includes(state.generalModel) ? state.generalModel : "deepseek";
  return `通用模式:${selected}`;
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
  menu.setAttribute("aria-label", "通用模式模型选择");

  for (const modelId of GENERAL_MODELS) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "model-menu-item";
    item.setAttribute("role", "menuitemradio");
    item.setAttribute("data-model-id", modelId);
    item.textContent = modelId;
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
        `<span class="upload-file-item">${escapeHtml(item.name)}<button class="upload-file-remove" data-file-id="${escapeHtml(item.id)}" type="button" aria-label="删除文件">✕</button></span>`
    )
    .join("");
}

function getModeGreetingText() {
  return state.chatMode === "expert" ? "你好，我是专家模式" : "你好，我是通用模式";
}

function syncModeUi() {
  const isExpert = state.chatMode === "expert";

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
    modelSwitch.textContent = isExpert ? "专家模式▾" : `${getGeneralModelText()}▾`;
    modelSwitch.setAttribute("aria-expanded", String(Boolean(modelMenuEl && !modelMenuEl.hidden)));
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
    fileUploadBtn.classList.remove("active");
    fileUploadBtn.setAttribute("aria-pressed", "false");
    fileUploadBtn.title = "点击上传文件";
  }

  renderUploadedFiles();

  if (sqlAnalysisToggleBtn) {
    sqlAnalysisToggleBtn.classList.toggle("active", state.sqlAnalysisEnabled);
    sqlAnalysisToggleBtn.setAttribute("aria-pressed", String(state.sqlAnalysisEnabled));
    sqlAnalysisToggleBtn.title = state.sqlAnalysisEnabled ? "点击关闭SQL分析" : "点击开启SQL分析";
  }

  if (edaAnalysisToggleBtn) {
    edaAnalysisToggleBtn.classList.toggle("active", state.edaAnalysisEnabled);
    edaAnalysisToggleBtn.setAttribute("aria-pressed", String(state.edaAnalysisEnabled));
    edaAnalysisToggleBtn.title = state.edaAnalysisEnabled ? "点击关闭EDA分析" : "点击开启EDA分析";
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
      edaReportThreadId: typeof msg?.edaReportThreadId === "string" ? msg.edaReportThreadId : null,
      edaMessageSource: msg?.edaMessageSource === "upload" || msg?.edaMessageSource === "chat" ? msg.edaMessageSource : null,
    }))
    .filter((msg) => msg.text.trim().length > 0)
    .filter((msg) => !(msg.role === "assistant" && isLegacyWelcomeText(msg.text)));
}

async function previewEdaReport(threadId) {
  if (typeof threadId !== "string" || !threadId.trim()) return;

  const safeId = encodeURIComponent(threadId.trim());
  const resp = await fetch(`/api/eda/report/${safeId}`);
  if (!resp.ok) {
    throw new Error(`报告获取失败：HTTP ${resp.status}`);
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
      generalModel: state.generalModel,
      edaAnalysisEnabled: state.edaAnalysisEnabled,
      sqlAnalysisEnabled: state.sqlAnalysisEnabled,
      isDbConnected: state.isDbConnected,
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
    state.generalModel = GENERAL_MODELS.includes(parsed.generalModel) ? parsed.generalModel : "deepseek";
    state.edaAnalysisEnabled = Boolean(parsed.edaAnalysisEnabled);
    state.sqlAnalysisEnabled = Boolean(parsed.sqlAnalysisEnabled);
    state.isDbConnected = Boolean(parsed.isDbConnected);

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
    return;
  }

  for (const msg of convo.messages) {
    const article = document.createElement("article");
    article.className = `msg ${msg.role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (msg.role === "assistant" && msg.pending) {
      bubble.classList.add("thinking");
      const pendingLabel = typeof msg.pendingLabel === "string" && msg.pendingLabel.trim() ? msg.pendingLabel.trim() : "分析中 🔍";
      bubble.innerHTML = `<span class="thinking-label">${escapeHtml(pendingLabel)}</span>`;
    } else if (msg.role === "assistant") {
      bubble.classList.add("rich");
      bubble.innerHTML = renderAssistantMessageHtml(msg.text);
    } else {
      bubble.textContent = msg.text;
    }

    article.appendChild(bubble);

    if (msg.role === "assistant" && !msg.pending && typeof msg.edaReportThreadId === "string" && msg.edaReportThreadId.trim()) {
      const actions = document.createElement("div");
      actions.className = "eda-report-actions";
      const hasSuccessMark = typeof msg.text === "string" && msg.text.includes("✅");
      const source = msg.edaMessageSource;
      const isUploadMessage = source === "upload";
      const isChatMessage = source === "chat";

      if (isUploadMessage) {
        const previewBtn = document.createElement("button");
        previewBtn.className = "eda-report-btn";
        previewBtn.type = "button";
        previewBtn.textContent = "预览EDA报告";
        previewBtn.addEventListener("click", async () => {
          try {
            await previewEdaReport(msg.edaReportThreadId);
          } catch (err) {
            appendAssistantSystemMessage(`EDA报告预览失败：${err?.message || "未知错误"}`);
          }
        });

        const downloadBtn = document.createElement("button");
        downloadBtn.className = "eda-report-btn";
        downloadBtn.type = "button";
        downloadBtn.textContent = "下载EDA报告";
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
        downloadCsvBtn.textContent = "下载CSV";
        downloadCsvBtn.addEventListener("click", () => {
          downloadEdaCsv(msg.edaReportThreadId);
        });
        actions.appendChild(downloadCsvBtn);
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
    if (!confirm("确定要断开连接并清除会话缓存吗？")) return;

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
        alert("已断开连接并重置会话");
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

    // 1. 强制获取当前的 Session ID
    const currentSessionId = state.activeConversationId;
    if (!currentSessionId) {
      alert("错误：未找到有效的会话ID，请刷新页面重试。");
      return;
    }

    const formData = new FormData(dbForm);
    const rawInfo = Object.fromEntries(formData.entries());

    // 2. 显式构建 Payload，确保字段名与后端 Pydantic 模型完全一致
    const payload = {
      session_id: currentSessionId,
      host: rawInfo.host,
      port: parseInt(rawInfo.port) || 3306, // 强制转为整数
      user: rawInfo.user,
      password: rawInfo.password,
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
        alert("✅ 数据库连接成功！");
        state.isDbConnected = true; // 移动到这里：确保成功才置为 true
        updateDbUI();              // 移动到这里
        dbModal.hidden = true;
        dbModal.classList.remove("active");
        dbForm.reset();
      } else {
        state.isDbConnected = false; // 明确失败状态
        updateDbUI();
        const errDetail = await response.json();
        alert(`❌ 连接失败: ${JSON.stringify(errDetail.detail)}`);
      }
    } catch (err) {
      console.error("Network error:", err);
      alert("网络错误，无法连接服务器。");
    }
  });
}

// 3. 脚本入口：等待 DOM 加载完毕再启动
document.addEventListener("DOMContentLoaded", () => {
  // 恢复状态
  const restored = restoreState();
  if (!restored) {
    createConversation();
  } else {
    renderAll();
  }

  // 初始化模型菜单和数据库功能
  ensureModelMenu();
  initDbConnection();
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
    edaReportThreadId: typeof meta.edaReportThreadId === "string" ? meta.edaReportThreadId : null,
    edaMessageSource: meta?.edaMessageSource === "upload" || meta?.edaMessageSource === "chat" ? meta.edaMessageSource : null,
  });
  convo.updatedAt = Date.now();
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  persistState();
  renderAll();
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
  const pendingLabel = typeof uiOptions.pendingLabel === "string" ? uiOptions.pendingLabel : "";

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
    edaReportThreadId: options?.eda_analysis ? convo.id : null,
    edaMessageSource: options?.eda_analysis ? "chat" : null,
  };
  convo.messages.push(assistantMessage);
  convo.updatedAt = Date.now();

  // 如果还是默认标题，用第一条用户消息更新标题
  if (!hideUserMessage && /^新聊天\s\d+$/.test(convo.title)) {
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
    assistantMessage.pending = false;
    if (!assistantMessage.text.trim()) {
      assistantMessage.text = "模型未返回文本内容。";
    }

    // 本轮消息发送成功后，清空已上传文件标签，避免影响下一轮输入。
    state.uploadedFiles = [];
    syncModeUi();
  } catch (err) {
    assistantMessage.pending = false;
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

// 清除所有聊天记录按钮
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
clearHistoryBtn?.addEventListener("click", clearAllHistory);

modelSwitch?.addEventListener("click", () => {
  if (state.chatMode !== "general") {
    closeModelMenu();
    return;
  }

  ensureModelMenu();
  if (!modelMenuEl) return;

  if (modelMenuEl.hidden) {
    syncModelMenuSelection();
    openModelMenu();
  } else {
    closeModelMenu();
  }

  syncModeUi();
});

// 点击后切换到专家模式，并更新欢迎语。
taskChipBtn?.addEventListener("click", () => {
  state.chatMode = state.chatMode === "expert" ? "general" : "expert";
  if (state.chatMode !== "expert") {
    state.edaAnalysisEnabled = false;
    state.sqlAnalysisEnabled = false;
  }
  syncModeUi();
  persistState();
});

edaAnalysisToggleBtn?.addEventListener("click", () => {
  if (state.chatMode !== "expert") {
    alert("请先开启『专家模式』后再使用 EDA 分析功能。");
    return;
  }
  state.edaAnalysisEnabled = !state.edaAnalysisEnabled;
  syncModeUi();
  persistState();
});

fileUploadBtn?.addEventListener("click", () => {
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
    edaPendingMessage = { role: "assistant", text: "", pending: true, pendingLabel: "EDA分析中 🔍" };
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

      if (!resp.ok) throw new Error("上传失败");

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
          uploadEdaFailures.push(`【${returnedName}】upload解析失败，请重试。`);
        }
      }
    } catch (err) {
      console.error("上传失败:", err);
      if (useEdaUpload) {
        uploadEdaFailures.push(`【${file.name}】upload失败，请重试。`);
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
    alert("请先开启『专家模式』后再使用 SQL 分析功能。");
    return;
  }
  state.sqlAnalysisEnabled = !state.sqlAnalysisEnabled;
  if (state.sqlAnalysisEnabled) {
    state.edaAnalysisEnabled = false;
  }
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
    eda_analysis: !!state.edaAnalysisEnabled,
    sql_analysis: !!state.sqlAnalysisEnabled,
    file_name: state.uploadedFiles.map((item) => item.name).join(",") || null
  };

  submitMessage(text, dynamicOptions);
  composerInput.value = "";
  if (typeof autoResizeTextarea === 'function') autoResizeTextarea();

});

// 初始化：优先恢复本地历史，否则新建一个会话
clearHtmlPreviewState();
ensureModelMenu();
if (!restoreState()) {
  createConversation();
} else {
  renderAll();
}

autoResizeTextarea();
