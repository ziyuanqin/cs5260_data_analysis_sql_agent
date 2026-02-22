// 轻量 UI 演示脚本：仅保留输入框交互，不接后端
const messageList = document.getElementById("messageList");
const composerForm = document.getElementById("composerForm");
const composerInput = document.getElementById("composerInput");

// 自动根据内容调整输入框高度
function autoResizeTextarea() {
  composerInput.style.height = "38px";
  composerInput.style.height = `${Math.min(composerInput.scrollHeight, 140)}px`;
}

// 创建用户消息（右侧气泡）
function appendUserMessage(text) {
  const article = document.createElement("article");
  article.className = "msg user";
  article.innerHTML = `<div class="bubble"></div>`;
  article.querySelector(".bubble").textContent = text;
  messageList.appendChild(article);
}

// 创建助手消息（左侧大段文本）
function appendAssistantMessage(text) {
  const article = document.createElement("article");
  article.className = "msg assistant";

  const wrapper = document.createElement("div");
  wrapper.className = "content";

  const p = document.createElement("p");
  p.textContent = text;
  wrapper.appendChild(p);

  article.appendChild(wrapper);
  messageList.appendChild(article);
}

// 滚动到底部
function scrollToBottom() {
  messageList.scrollTop = messageList.scrollHeight;
}

// Enter 发送，Shift+Enter 换行
composerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composerForm.requestSubmit();
  }
});

// 输入时动态调节高度
composerInput.addEventListener("input", autoResizeTextarea);

// 提交消息：先本地展示，功能接入后再替换为真实请求
composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = composerInput.value.trim();
  if (!text) return;

  appendUserMessage(text);
  appendAssistantMessage("已收到。当前为 UI 样式版，后续会接入真实模型与流式能力。");

  composerInput.value = "";
  autoResizeTextarea();
  scrollToBottom();
});

autoResizeTextarea();
