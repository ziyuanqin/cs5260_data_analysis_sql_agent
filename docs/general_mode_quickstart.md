# General Mode Quickstart

This document explains what `general` mode does and how to use it in the web UI.

## 1. What General Mode provides

- Intent routing:
  - `DIRECT` for simple Q&A
  - `PLAN` for multi-step tasks
- Agent stages:
  - Router -> Planner -> Executor -> Reviewer -> Summarizer
- Basic tools:
  - `file_read` (read local file inside workspace)
  - `web_read` (read text from a public URL)
- Model routing:
  - Select model alias from UI
  - Backend maps alias to provider/model
- Session memory:
  - Conversation and usage are persisted in SQLite

## 2. How to use in UI

1. Keep mode as `General Mode` (top bar).
2. Choose model from top dropdown:
   - `DeepSeek`
   - `OpenAI`
   - `HuggingFace (Qwen)`
3. Send a request:
   - Short prompt -> direct answer
   - Complex prompt -> planner flow
4. Watch side panels:
   - `Task Status`: stage timeline + step progress + token/cost
   - `Evidence`: tool usage, source, and output excerpt

## 3. Triggering tool usage

General mode tools are planner/executor-driven. You can prompt with explicit intent:

- `Read README.md and summarize key setup steps.`
- `Read https://example.com and extract the main points.`

Evidence cards will appear when a tool is used.

## 4. Notes and limits

- Frontend allows one active streaming request per browser tab.
- Use multiple tabs for true parallel chat sessions.
- `file_read` is restricted to project workspace files.
- `web_read` supports http/https URLs only.

## 5. Config and keys

- General mode backend config is read from:
  - `app_config.json` (local, ignored)
  - `app_config.example.json` (template, committed)
- If your repository does not contain `app_config.json`, create it from the example and fill your keys locally.
