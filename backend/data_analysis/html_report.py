import pandas as pd
from backend.data_analysis.shared import BG, CARD, ACCENT, WARN, DANGER, SUCCESS, TEXT, MUTED


def render_eda_html(
    eda_report:      dict,
    schema_info:     dict,
    table_name:      str,
    type_suggestions: dict,
    cleaning_log:    list,
) -> str:
    """Build the full HTML EDA report. Charts are embedded as base64 PNGs."""

    ov     = eda_report.get("overview", {})
    mv     = eda_report.get("missing_values", {})
    ns     = eda_report.get("numeric_stats", {})
    cs     = eda_report.get("categorical_stats", {})
    dq     = eda_report.get("data_quality", {})
    corr   = eda_report.get("correlations", {})
    charts = eda_report.get("charts", {})

    # ── Inner helpers ─────────────────────────────────────────

    def badge(txt, color):
        return (f'<span style="background:{color};color:#fff;padding:2px 10px;'
                f'border-radius:999px;font-size:12px;font-weight:600">{txt}</span>')

    def stat_card(label, value, color="#6366f1"):
        return (f'<div style="background:#0f172a;border-radius:10px;padding:14px 18px;'
                f'text-align:center;flex:1;min-width:120px">'
                f'<div style="color:#64748b;font-size:11px;margin-bottom:6px">{label}</div>'
                f'<div style="color:{color};font-size:22px;font-weight:800">{value}</div></div>')

    def section(title, content, accent=""):
        border = f"border-left:3px solid {accent};" if accent else ""
        return (f'<div style="background:#1e293b;border-radius:12px;padding:22px;'
                f'margin-bottom:18px;{border}">'
                f'<h2 style="color:#94a3b8;font-size:11px;font-weight:700;letter-spacing:1px;'
                f'text-transform:uppercase;margin:0 0 16px">{title}</h2>'
                f'{content}</div>')

    def chart_img(key):
        b64 = charts.get(key, "")
        if not b64:
            return ""
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;border-radius:10px;margin:8px 0" alt="{key}">')

    def table(headers, rows, col_colors=None):
        th = "padding:8px 14px;background:#0f172a;color:#64748b;font-size:11px;text-transform:uppercase;text-align:left"
        head = "".join(f"<th style='{th}'>{h}</th>" for h in headers)
        body = ""
        for row in rows:
            tds = ""
            for i, cell in enumerate(row):
                color = col_colors[i] if col_colors and i < len(col_colors) else "#e2e8f0"
                tds += (f'<td style="padding:8px 14px;border-bottom:1px solid #0f172a;'
                        f'color:{color};font-size:13px">{cell}</td>')
            body += f"<tr>{tds}</tr>"
        return (f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse">'
                f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')

    # ── Overview ──────────────────────────────────────────────
    ov_missing_color = WARN if ov.get("total_missing", 0) > 0 else SUCCESS
    ov_dup_color     = WARN if ov.get("duplicate_rows", 0) > 0 else SUCCESS
    ov_html = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap">'
        + stat_card("Rows",           f"{ov.get('rows', 0):,}")
        + stat_card("Columns",         ov.get("columns", 0))
        + stat_card("Missing Cells",  f"{ov.get('total_missing', 0):,}", ov_missing_color)
        + stat_card("Duplicate Rows",  ov.get("duplicate_rows", 0),      ov_dup_color)
        + stat_card("Memory (KB)",     ov.get("memory_usage_kb", 0),     "#94a3b8")
        + "</div>"
    )

    # ── Type suggestions ──────────────────────────────────────
    type_rows = [
        [col,
         info.get("current", "?"),
         info.get("suggested", "?"),
         badge(info.get("confidence", "?"),
               SUCCESS if info.get("confidence") == "high" else WARN),
         info.get("reason", "")]
        for col, info in (type_suggestions or {}).items()
    ]
    type_html = (
        table(["Column", "Current", "Suggested", "Confidence", "Reason"], type_rows)
        if type_rows
        else '<p style="color:#64748b;font-size:14px">No dtype changes suggested.</p>'
    )

    # ── Missing values ────────────────────────────────────────
    mv_rows = [
        [col, info["count"], f'{info["pct"]}%']
        for col, info in mv.get("columns_with_missing", {}).items()
    ]
    mv_html = (
        table(["Column", "Missing Count", "Missing %"], mv_rows, col_colors=[TEXT, WARN, WARN])
        if mv_rows
        else '<p style="color:#22c55e;font-size:14px">✓ No missing values.</p>'
    ) + chart_img("missing_bar")

    # ── Numeric stats ─────────────────────────────────────────
    ns_rows = [
        [col, s["mean"], s["median"], s["std"], s["min"], s["max"],
         s["skewness"],
         badge(str(s["outliers_iqr"]), DANGER if s["outliers_iqr"] > 0 else SUCCESS)]
        for col, s in ns.items()
    ]
    ns_html = (
        table(["Column", "Mean", "Median", "Std", "Min", "Max", "Skewness", "Outliers"],
              ns_rows, col_colors=[TEXT]*8)
        if ns_rows
        else '<p style="color:#64748b;font-size:14px">No numeric columns.</p>'
    ) + chart_img("histograms")

    # ── Categorical stats ─────────────────────────────────────
    cs_rows = [
        [col, s["unique_count"], s.get("mode", "?"), f'{s.get("mode_frequency_pct", 0)}%']
        for col, s in cs.items()
    ]
    cs_html = (
        table(["Column", "Unique", "Mode", "Mode %"], cs_rows)
        if cs_rows
        else '<p style="color:#64748b;font-size:14px">No categorical columns.</p>'
    ) + chart_img("categorical_bars")

    # ── Data quality ──────────────────────────────────────────
    issues = dq.get("issues", [])
    dq_html = (
        '<ul style="padding-left:20px;margin:0">'
        + "".join(f'<li style="color:#fbbf24;margin-bottom:6px;font-size:14px">{i}</li>'
                  for i in issues)
        + "</ul>"
        if issues
        else '<p style="color:#22c55e;font-size:14px">✓ No data quality issues detected.</p>'
    )

    # ── Correlations ──────────────────────────────────────────
    high      = corr.get("high_correlations", [])
    corr_html = chart_img("correlation_heatmap")
    if high:
        corr_html += '<div style="margin-top:14px">'
        for c in high:
            col_str    = SUCCESS if c["correlation"] > 0 else DANGER
            pill_style = 'background:#1e40af;color:#e2e8f0;padding:3px 10px;border-radius:999px;font-size:12px'
            corr_html += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
                f'<span style="{pill_style}">{c["col1"]}</span>'
                f'<span style="color:#64748b">↔</span>'
                f'<span style="{pill_style}">{c["col2"]}</span>'
                f'<strong style="color:{col_str}">r = {c["correlation"]}</strong></div>'
            )
        corr_html += "</div>"
    else:
        corr_html += '<p style="color:#64748b;font-size:14px">No strong correlations (|r| > 0.7).</p>'

    # ── Cleaning log ──────────────────────────────────────────
    cl      = cleaning_log or []
    cl_html = (
        '<ol style="padding-left:20px;margin:0">'
        + "".join(f'<li style="color:#22c55e;margin-bottom:4px;font-size:13px">{e}</li>'
                  for e in cl)
        + "</ol>"
        if cl
        else '<p style="color:#64748b;font-size:13px">No cleaning operations applied yet.</p>'
    )

    # ── Schema table ──────────────────────────────────────────
    schema_cols = schema_info.get("columns", {})
    sc_rows     = [
        [col, info.get("dtype","?"), info.get("null_count", 0),
         f'{info.get("null_pct", 0)}%', info.get("unique_count", "?")]
        for col, info in schema_cols.items()
    ]
    sc_html = table(["Column", "Dtype", "Nulls", "Null %", "Unique"], sc_rows,
                    col_colors=[TEXT, ACCENT, TEXT, TEXT, TEXT])

    # ── Assemble ──────────────────────────────────────────────
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EDA Report — {table_name}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f172a;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;padding:32px 20px;line-height:1.6}}
  h1{{font-size:26px;font-weight:800;color:#f1f5f9;margin-bottom:6px}}
  .subtitle{{color:#64748b;font-size:13px;margin-bottom:28px}}
  a{{color:#6366f1}}
</style>
</head>
<body>
<h1>📊 EDA Report — <code style="color:#6366f1">{table_name}</code></h1>
<p class="subtitle">Generated by Data Analysis Agent · {ts}</p>

{section("Dataset Overview",              ov_html)}
{section("Column Schema",                 sc_html)}
{section("⚠️ AI-Suggested Type Conversions", type_html, WARN)}
{section("Missing Values",                mv_html, WARN if mv_rows else SUCCESS)}
{section("Numeric Statistics",            ns_html)}
{section("Categorical Statistics",        cs_html)}
{section("Data Quality",                  dq_html, DANGER if issues else SUCCESS)}
{section("Correlations",                  corr_html)}
{section("Cleaning Log",                  cl_html, SUCCESS if cl else "")}
</body>
</html>"""
