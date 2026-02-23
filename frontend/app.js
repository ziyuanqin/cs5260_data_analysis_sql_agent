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
};

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

    state.conversations = parsed.conversations;
    state.activeConversationId = parsed.activeConversationId || parsed.conversations[0].id;
    state.chatCounter = Number(parsed.chatCounter || parsed.conversations.length + 1);
    return true;
  } catch {
    return false;
  }
}

// 创建新会话
function createConversation() {
  const id = crypto.randomUUID();
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

// 渲染历史列表
function renderHistory() {
  historyList.innerHTML = "";

  for (const convo of state.conversations) {
    const btn = document.createElement("button");
    btn.className = `history-item${convo.id === state.activeConversationId ? " active" : ""}`;
    btn.textContent = convo.title;
    btn.title = convo.title;
    btn.addEventListener("click", () => switchConversation(convo.id));
    historyList.appendChild(btn);
  }
}

// 渲染当前会话消息
function renderMessages() {
  messageList.innerHTML = "";
  const convo = getActiveConversation();
  if (!convo) return;

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

// 发送消息：写入当前会话并模拟助手回复
function submitMessage(text) {
  const convo = getActiveConversation();
  if (!convo) return;

  convo.messages.push({ role: "user", text });
  convo.messages.push({ role: "assistant", text: "已收到。当前为 UI 样式版，后续会接入真实模型与流式能力。" });
  convo.updatedAt = Date.now();

  // 如果还是默认标题，用第一条用户消息更新标题
  if (/^新聊天\s\d+$/.test(convo.title)) {
    convo.title = shortTitle(text);
  }

  // 最近更新的会话放到顶部
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);

  persistState();
  renderAll();
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
