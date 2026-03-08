// UI 交互脚本：支持多会话创建、会话历史保存、点击切换会话
const STORAGE_KEY = "agent_ui_chat_sessions_v1";

const messageList = document.getElementById("messageList");
const historyList = document.getElementById("historyList");
const newChatBtn = document.getElementById("newChatBtn");
const composerForm = document.getElementById("composerForm");
const composerInput = document.getElementById("composerInput");

const state = {
  conversations: [],
  activeConversationId: null,
  chatCounter: 1,
  isStreaming: false,
};

function createId() {
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
    .filter((msg) => msg.text.trim().length > 0);
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

// 保存到本地：刷新页面后仍可恢复
function persistState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      conversations: state.conversations,
      activeConversationId: state.activeConversationId,
      chatCounter: state.chatCounter,
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
          messages: messages.length
            ? messages
            : [{ role: "assistant", text: "你好~可以开始新对话了" }],
        };
      })
      .filter((convo) => convo.id);

    if (!normalizedConversations.length) return false;

    state.conversations = normalizedConversations;
    state.activeConversationId =
      normalizedConversations.find((c) => c.id === parsed.activeConversationId)?.id ||
      normalizedConversations[0].id;
    state.chatCounter = Number(parsed.chatCounter || normalizedConversations.length + 1);

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
    messages: [{ role: "assistant", text: "你好~可以开始新对话了" }],
  };

  state.chatCounter += 1;
  state.conversations.unshift(convo);
  state.activeConversationId = id;

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
    convo.messages = [{ role: "assistant", text: "你好~可以开始新对话了" }];
  }

  for (const msg of convo.messages) {
    const article = document.createElement("article");
    article.className = `msg ${msg.role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = msg.text;

    article.appendChild(bubble);
    messageList.appendChild(article);
  }

  messageList.scrollTop = messageList.scrollHeight;
}

// 统一刷新
function renderAll() {
  renderHistory();
  renderMessages();
}

async function streamAssistantReply(conversationId, assistantMessage, userText) {
  const resp = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: conversationId,
      mode: "general",
      message: userText,
      stream: true,
    }),
  });

  if (!resp.ok || !resp.body) {
    throw new Error(`请求失败：HTTP ${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
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

      if (payload?.event === "error") {
        throw new Error(payload?.payload?.message || "流式返回错误");
      }
    }
  }
}

// 发送消息：写入当前会话并通过后端流式获取回复
async function submitMessage(text) {
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
    await streamAssistantReply(convo.id, assistantMessage, text);
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

// 提交消息
composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = composerInput.value.trim();
  if (!text) return;

  submitMessage(text);
  composerInput.value = "";
  autoResizeTextarea();
});

// 初始化：优先恢复本地历史，否则新建一个会话
if (!restoreState()) {
  createConversation();
} else {
  renderAll();
}

autoResizeTextarea();
