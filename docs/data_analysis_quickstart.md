# Data Analysis Quickstart

This quickstart explains how to use the `expert` data-analysis workflow, where to configure API keys, and how it differs from `general` mode.

## 1. What `expert` mode includes

- EDA upload pipeline:
  - Upload file
  - Type inference and optional cleaning
  - Automated EDA report generation (HTML)
- EDA follow-up chat:
  - Cleaning instructions (for example: `drop duplicates`, `fill NaN`)
  - Custom EDA requests (for example: histogram, correlation)
  - `rerun eda` to regenerate report
- SQL analysis:
  - Database connection flow (MySQL)
  - SQL generation and analysis response

## 2. Where to configure API keys

There are currently two key/config paths in the project:

- `general` mode (new unified chat service):
  - Uses `app_config.json`
  - Keys: `openai_api_key`, `huggingface_api_key`
- `expert` data analysis backend (`backend/data_analysis/agent.py`):
  - Supports both `.env` and `app_config.json`
  - Priority: non-empty env value first, then fallback to `app_config.json`
  - Keys:
    - env: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`
    - app_config: `openai_api_key`, `deepseek_api_key`

Recommended practice:

- You can now keep only `app_config.json` keys and leave `.env` empty.
- If env keys are present and non-empty, they override `app_config.json`.

## 3. How to use Expert EDA in UI

1. Switch to `Expert Mode`.
2. Enable `EDA Analysis`.
3. Click `Upload Files` and upload CSV/Excel.
4. Wait for EDA completion message.
5. Use follow-up prompts:
   - `drop duplicates`
   - `fill NaN in column_x with 0`
   - `show histogram of age`
   - `rerun eda`

## 4. How to use Expert SQL in UI

1. Switch to `Expert Mode`.
2. Enable `SQL Analysis`.
3. Optional: connect DB from `Database Connection`.
4. Ask natural-language SQL tasks.

## 5. General vs Expert boundary

- `general` mode:
  - Planner/Executor/Reviewer/Summarizer
  - Task status and evidence panel
  - Tool calls (`file_read`, `web_read`)
- `expert` mode:
  - Data analysis and SQL workflow
  - EDA/SQL-specific UI controls

Keep these boundaries to avoid feature collisions.

## 6. Session behavior and concurrency notes

- The backend supports multiple sessions (`session_id`) in parallel.
- Frontend currently allows one active streaming request per browser tab.
- Use multiple tabs/windows for true concurrent conversations.
