"""
processors.py — Core data-processing classes.

  DatasetUnderstanding  — load CSV and build schema dict
  TypeInferencer        — heuristic + LLM dtype suggestions
  AutomatedEDA          — stats, correlations, default charts
  DataCleaningEngine    — NL-driven cleaning ops via LLM
  CustomEDAEngine       — bar / box / hist / scatter / regression / k-means
"""

import json
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.messages import HumanMessage

from backend.data_analysis.shared import (
    log,
    BG, CARD, ACCENT, WARN, DANGER, SUCCESS, TEXT, MUTED,
    _fig_to_b64, _dark_fig, _dark_fig_multi,
)


class DatasetUnderstanding:

    def load_and_inspect(self, file_path: str) -> dict:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return {"error": str(e), "df": None, "schema": None}
        return {"df": df, "schema": self._inspect(df), "error": None}

    def _inspect(self, df: pd.DataFrame) -> dict:
        schema = {}
        for col in df.columns:
            s = df[col]
            info = {
                "dtype":        str(s.dtype),
                "null_count":   int(s.isnull().sum()),
                "null_pct":     round(s.isnull().mean() * 100, 2),
                "unique_count": int(s.nunique()),
                "sample":       [str(v) for v in s.dropna().head(3).tolist()],
            }
            if pd.api.types.is_numeric_dtype(s):
                d = s.describe()
                info["distribution"] = {
                    k: round(float(d[k2]), 4) for k, k2 in
                    [("min","min"),("max","max"),("mean","mean"),("std","std"),
                     ("q25","25%"),("q75","75%")]
                }
                info["median"] = round(float(s.median()), 4)
            else:
                vc = s.value_counts().head(5)
                info["top_values"] = {str(k): int(v) for k, v in vc.items()}
            schema[col] = info

        return {
            "row_count":      len(df),
            "col_count":      len(df.columns),
            "columns":        schema,
            "column_names":   list(df.columns),
            "dtypes_summary": df.dtypes.astype(str).to_dict(),
        }


class TypeInferencer:
    """
    Heuristic dtype detection; LLM is only called for ≤5 ambiguous
    columns to keep token usage minimal.
    """

    def infer(self, df: pd.DataFrame, llm) -> dict:
        """Return {col: {current, suggested, reason, confidence}}."""
        suggestions: dict = {}
        ambiguous:   list = []

        log.info("[TypeInferencer] Inspecting %d columns: %s",
                 len(df.columns), list(df.columns))

        for col in df.columns:
            s   = df[col]
            cur = str(s.dtype)

            # Both "object" and pandas StringDtype should be treated as text
            is_text = s.dtype == object or isinstance(s.dtype, pd.StringDtype)

            if is_text:
                log.info("  col='%s' dtype=%s — checking for better type", col, cur)

                # text → numeric?
                converted    = pd.to_numeric(s, errors="coerce")
                success_pct  = converted.notna().mean()
                if success_pct > 0.9:
                    log.info("    → suggests float64 (%.0f%% numeric)", success_pct * 100)
                    suggestions[col] = {"current": cur, "suggested": "float64",
                                        "reason": f"{success_pct*100:.0f}% values are numeric",
                                        "confidence": "high"}
                    continue

                # text → datetime?
                try:
                    pd.to_datetime(s.dropna().head(20), infer_datetime_format=True)
                    log.info("    → suggests datetime64")
                    suggestions[col] = {"current": cur, "suggested": "datetime64",
                                        "reason": "values look like dates", "confidence": "high"}
                    continue
                except Exception:
                    pass

                # text → bool?
                unique_lower = {str(v).strip().lower() for v in s.dropna().unique()}
                if unique_lower <= {"true","false","yes","no","1","0","t","f"}:
                    log.info("    → suggests bool")
                    suggestions[col] = {"current": cur, "suggested": "bool",
                                        "reason": "binary text values", "confidence": "high"}
                    continue

                # Queue low-cardinality text for LLM
                if s.nunique() < 20:
                    log.info("    → ambiguous, queuing for LLM (nunique=%d)", s.nunique())
                    ambiguous.append({"col": col, "dtype": cur,
                                      "sample": [str(v) for v in s.dropna().head(5).tolist()],
                                      "unique_count": int(s.nunique())})
                else:
                    log.info("    → high-cardinality text, skipping")

            elif s.dtype in (np.float64, np.float32):
                non_null = s.dropna()
                if (len(non_null) > 0
                        and non_null.apply(lambda x: x == int(x)).all()
                        and non_null.min() >= 0):
                    log.info("  col='%s' dtype=%s → suggests int64", col, cur)
                    suggestions[col] = {"current": cur, "suggested": "int64",
                                        "reason": "all values are whole numbers",
                                        "confidence": "high"}
            else:
                log.info("  col='%s' dtype=%s — no change needed", col, cur)

        # # LLM batch call for ambiguous columns only
        # if ambiguous:
        #     log.info("[TypeInferencer] Sending %d ambiguous col(s) to LLM", len(ambiguous))
        #     prompt = (
        #         "For each column below, suggest the best pandas dtype from: "
        #         "category, str, int64, float64, bool, datetime64. "
        #         'Reply ONLY with compact JSON: {"col_name": "dtype", ...}\n\n'
        #         + json.dumps(ambiguous[:5])
        #     )
        #     resp = llm.invoke([HumanMessage(content=prompt)])
        #     try:
        #         for col, dtype in json.loads(resp.content.strip()).items():
        #             if col in df.columns and col not in suggestions:
        #                 log.info("    LLM suggests col='%s' → %s", col, dtype)
        #                 suggestions[col] = {"current": str(df[col].dtype),
        #                                     "suggested": dtype,
        #                                     "reason": "LLM inference from sample values",
        #                                     "confidence": "low"}
        #     except Exception:
        #         log.warning("[TypeInferencer] Failed to parse LLM response: %s",
        #                     resp.content[:100])

        log.info("[TypeInferencer] Done — %d suggestion(s): %s", len(suggestions),
                 {c: v["suggested"] for c, v in suggestions.items()})
        return suggestions

    def apply_suggestion(self, df: pd.DataFrame, col: str, dtype: str) -> tuple:
        """Apply a single dtype conversion. Returns (new_df, success, message)."""
        try:
            df = df.copy()
            if dtype == "datetime64":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "bool":
                mapping = {"true":True,"false":False,"yes":True,"no":False,
                           "1":True,"0":False,"t":True,"f":False}
                df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)
            elif dtype == "category":
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(dtype, errors="ignore")
            return df, True, f"Column `{col}` converted to `{dtype}`."
        except Exception as e:
            return df, False, f"Conversion failed for `{col}`: {e}"


class AutomatedEDA:

    def run(self, df: pd.DataFrame) -> dict:
        return {
            "overview":          self._overview(df),
            "missing_values":    self._missing_values(df),
            "numeric_stats":     self._numeric_stats(df),
            "categorical_stats": self._categorical_stats(df),
            "data_quality":      self._data_quality(df),
            "correlations":      self._correlations(df),
            "charts":            self._default_charts(df),
        }

    def _overview(self, df):
        return {
            "rows":             len(df),
            "columns":          len(df.columns),
            "total_cells":      len(df) * len(df.columns),
            "total_missing":    int(df.isnull().sum().sum()),
            "duplicate_rows":   int(df.duplicated().sum()),
            "memory_usage_kb":  round(df.memory_usage(deep=True).sum() / 1024, 2),
        }

    def _missing_values(self, df):
        missing = df.isnull().sum()
        pct     = (missing / len(df) * 100).round(2)
        wm      = missing[missing > 0]
        return {
            "columns_with_missing": {
                c: {"count": int(missing[c]), "pct": float(pct[c])} for c in wm.index
            },
            "complete_columns":  int((missing == 0).sum()),
            "high_missing_cols": list(pct[pct > 50].index),
        }

    def _numeric_stats(self, df):
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr     = q3 - q1
            stats[col] = {
                "mean":         round(float(s.mean()), 4),
                "median":       round(float(s.median()), 4),
                "std":          round(float(s.std()), 4),
                "min":          round(float(s.min()), 4),
                "max":          round(float(s.max()), 4),
                "skewness":     round(float(s.skew()), 4),
                "kurtosis":     round(float(s.kurtosis()), 4),
                "outliers_iqr": int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()),
            }
        return stats

    def _categorical_stats(self, df):
        stats = {}
        for col in df.select_dtypes(include=["object","category"]).columns:
            s  = df[col].dropna()
            vc = s.value_counts()
            stats[col] = {
                "unique_count":        int(s.nunique()),
                "top_5_values":        {str(k): int(v) for k, v in vc.head(5).items()},
                "mode":                str(vc.index[0]) if len(vc) else None,
                "mode_frequency_pct":  round(float(vc.iloc[0] / len(s) * 100), 2) if len(vc) else 0,
            }
        return stats

    def _data_quality(self, df):
        issues = []
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(f"Column '{col}' is constant.")
        for col in df.columns:
            if df[col].nunique() / max(len(df), 1) > 0.9:
                issues.append(f"Column '{col}' has very high cardinality (possible ID column).")
        dups = df.duplicated().sum()
        if dups:
            issues.append(f"{dups} duplicate rows detected.")
        for col in df.columns:
            pct = df[col].isnull().mean() * 100
            if pct > 50:
                issues.append(f"Column '{col}' is missing {pct:.1f}% of values.")
        return {"issues": issues, "issue_count": len(issues)}

    def _correlations(self, df):
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return {"high_correlations": [], "correlation_matrix": {}}
        corr = num_df.corr().round(3)
        high = [
            {"col1": corr.columns[i], "col2": corr.columns[j],
             "correlation": float(corr.iloc[i, j])}
            for i in range(len(corr.columns))
            for j in range(i + 1, len(corr.columns))
            if abs(corr.iloc[i, j]) > 0.7
        ]
        return {"high_correlations": high, "correlation_matrix": corr.to_dict()}

    def _default_charts(self, df) -> dict:
        charts = {}

        # 1. Missing values bar
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            fig, ax = _dark_fig((8, max(3, len(missing) * 0.45)))
            pct  = (missing / len(df) * 100).round(1)
            bars = ax.barh(pct.index.tolist(), pct.values, color=WARN, edgecolor="none")
            for bar, val in zip(bars, pct.values):
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", color=TEXT, fontsize=9)
            ax.set_xlabel("Missing (%)")
            ax.set_title("Missing Values by Column")
            ax.set_xlim(0, max(pct.values) * 1.15)
            charts["missing_bar"] = _fig_to_b64(fig)

        # 2. Correlation heatmap
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr    = num_df.corr()
            n       = len(corr)
            figsize = (max(6, n * 0.85), max(5, n * 0.75))
            fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
            ax.set_facecolor(BG)
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(corr, ax=ax, cmap=cmap, annot=True, fmt=".2f",
                        linewidths=0.5, linecolor="#334155",
                        cbar_kws={"shrink": 0.8},
                        annot_kws={"size": 9, "color": TEXT})
            ax.set_title("Correlation Heatmap", color=TEXT, pad=12)
            ax.tick_params(colors=MUTED, labelrotation=45)
            plt.setp(ax.get_xticklabels(), ha="right")
            fig.patch.set_facecolor(BG)
            charts["correlation_heatmap"] = _fig_to_b64(fig)

        # 3. Histograms grid
        num_cols = num_df.columns.tolist()
        if num_cols:
            ncols = min(3, len(num_cols))
            nrows = (len(num_cols) + ncols - 1) // ncols
            fig, axes = _dark_fig_multi(nrows, ncols, (ncols * 4, nrows * 3.2))
            for i, col in enumerate(num_cols):
                ax   = axes[i]
                data = df[col].dropna()
                ax.hist(data, bins=min(30, max(5, len(data) // 10)),
                        color=ACCENT, edgecolor=BG, alpha=0.85)
                ax.set_title(col, fontsize=10)
                ax.set_ylabel("Count")
                ax.axvline(data.mean(), color=WARN, linestyle="--", linewidth=1.2,
                           label=f"μ={data.mean():.2f}")
                ax.legend(fontsize=7, facecolor=CARD, labelcolor=MUTED)
            for j in range(len(num_cols), len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Numeric Distributions", color=TEXT, fontsize=13, y=1.01)
            plt.tight_layout()
            charts["histograms"] = _fig_to_b64(fig)

        # 4. Categorical top-5 bars
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        if cat_cols:
            ncols = min(2, len(cat_cols))
            nrows = (len(cat_cols) + ncols - 1) // ncols
            fig, axes = _dark_fig_multi(nrows, ncols, (ncols * 5, nrows * 3.2))
            for i, col in enumerate(cat_cols):
                ax     = axes[i]
                vc     = df[col].value_counts().head(5)
                colors = plt.cm.get_cmap("coolwarm", len(vc))
                ax.bar(range(len(vc)), vc.values,
                       color=[colors(j) for j in range(len(vc))], edgecolor=BG)
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels([str(x)[:12] for x in vc.index],
                                   rotation=30, ha="right", fontsize=9)
                ax.set_title(col, fontsize=10)
                ax.set_ylabel("Count")
            for j in range(len(cat_cols), len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Categorical Distributions (Top 5)", color=TEXT, fontsize=13, y=1.01)
            plt.tight_layout()
            charts["categorical_bars"] = _fig_to_b64(fig)

        return charts


class DataCleaningEngine:
    """
    Interprets natural-language cleaning requests via LLM and
    dispatches to typed operations.

    Supported ops: drop_col, drop_duplicates, drop_rows_where,
                   convert_dtype, rename_col, fill_nan.
    """

    def apply(self, df: pd.DataFrame, schema_info: dict,
              user_request: str, llm) -> dict:

        prompt = f"""You are a data cleaning assistant. Dataset columns and dtypes:
{json.dumps(schema_info["dtypes_summary"], indent=2)}

User request: "{user_request}"

Reply ONLY with compact JSON matching ONE schema:

Drop column:      {{"op":"drop_col","cols":["col1","col2"]}}
Drop duplicates:  {{"op":"drop_duplicates","subset":null}}
Drop rows (cond): {{"op":"drop_rows_where","col":"<col>","operator":"<|>|==|!=|>=|<=","value":<val>}}
Convert dtype:    {{"op":"convert_dtype","col":"<col>","dtype":"int64|float64|str|bool|datetime64|category"}}
Rename column:    {{"op":"rename_col","old":"<col>","new":"<col>"}}
Fill NaN:         {{"op":"fill_nan","col":"<col>","value":"<val_or_mean_or_median_or_mode>"}}
Unsupported:      {{"op":"unsupported","message":"<str>"}}
"""
        resp = llm.invoke([HumanMessage(content=prompt)])
        try:
            action = json.loads(resp.content.strip())
        except Exception:
            return {"error": f"Could not parse cleaning action: {resp.content[:200]}",
                    "df": df, "log": None}

        op = action.get("op", "")
        try:
            return self._dispatch(df, action)
        except Exception:
            return {"error": f"Cleaning op `{op}` failed:\n{traceback.format_exc(limit=3)}",
                    "df": df, "log": None}

    def _dispatch(self, df, a) -> dict:
        op = a["op"]
        df = df.copy()

        if op == "drop_col":
            # Safety check: only drop columns that actually exist
            cols = [c for c in a.get("cols", []) if c in df.columns]
            if not cols:
                return {"df": df, "log": "No valid columns to drop", "reply": "⚠️ No matching columns found."}
            df.drop(columns=cols, inplace=True)
            msg = f"Dropped column(s): {', '.join(cols)}"
            return {"df": df, "log": msg, "reply": f"✅ {msg}"}

        if op == "drop_duplicates":
            before = len(df)
            df.drop_duplicates(subset=a.get("subset") or None, inplace=True)
            msg = f"Dropped {before - len(df)} duplicate rows"
            return {"df": df, "log": msg, "reply": f"✅ {msg}. {len(df)} rows remain."}

        if op == "drop_rows_where":
            col, op_str, val = a["col"], a["operator"], a["value"]
            # Basic validation to prevent crashing on missing columns
            if col not in df.columns:
                return {"df": df, "log": None, "reply": f"⚠️ Column `{col}` not found."}
            
            ops = {
                "<":  df[col] < val,  ">":  df[col] > val,
                "==": df[col] == val, "!=": df[col] != val,
                ">=": df[col] >= val, "<=": df[col] <= val
            }
            mask = ops[op_str]
            before = len(df)
            df = df[~mask]
            msg = f"Dropped {before - len(df)} rows where {col} {op_str} {val}"
            return {"df": df, "log": msg, "reply": f"✅ {msg}. {len(df)} rows remain."}

        if op == "convert_dtype":
            col, dtype = a["col"], a["dtype"]
            # Note: Ensure TypeInferencer is defined in your environment
            df, ok, msg = TypeInferencer().apply_suggestion(df, col, dtype)
            return {"df": df, "log": msg, "reply": ("✅ " if ok else "⚠️ ") + msg}

        if op == "rename_col":
            old, new = a["old"], a["new"]
            if old not in df.columns:
                 return {"df": df, "log": None, "reply": f"⚠️ Column `{old}` not found."}
            df.rename(columns={old: new}, inplace=True)
            msg = f"Renamed column `{old}` → `{new}`"
            return {"df": df, "log": msg, "reply": f"✅ {msg}"}

        if op == "fill_nan":
            col, val = a["col"], a["value"]
            if col not in df.columns:
                 return {"df": df, "log": None, "reply": f"⚠️ Column `{col}` not found."}
                 
            if val == "mean": fill = df[col].mean()
            elif val == "median": fill = df[col].median()
            elif val == "mode": fill = df[col].mode().iloc[0] if not df[col].mode().empty else None
            else:
                # Simple numeric check helper
                try:
                    fill = float(val) if "." in str(val) else int(val)
                except:
                    fill = val
                    
            count = int(df[col].isnull().sum())
            df[col] = df[col].fillna(fill) # Avoid inplace warning in newer pandas
            msg = f"Filled {count} NaN in `{col}` with {fill}"
            return {"df": df, "log": msg, "reply": f"✅ {msg}"}

        if op == "unsupported":
            return {"df": df, "log": None,
                    "reply": f"⚠️ {a.get('message','Unsupported cleaning operation.')}\n\n"
                             "Supported: drop columns, drop duplicates, drop rows by condition, "
                             "convert dtype, rename column, fill NaN."}

        return {"df": df, "log": None, "reply": f"⚠️ Unknown op: {op}"}


def _is_numeric_str(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


class CustomEDAEngine:
    """Produces on-demand charts: bar | box | histogram | scatter |
    linear_regression | kmeans — driven by a single NL request."""

    def run(self, df, schema_info, user_request, llm) -> dict:
        prompt = f"""Dataset columns:
{json.dumps(schema_info["dtypes_summary"], indent=2)}

User: "{user_request}"

Reply ONLY with JSON matching one schema:
Bar:        {{"action":"bar_plot","x_col":"<col>","y_col":"<col|null>","agg":"count|mean|sum","title":"<str>"}}
Box:        {{"action":"box_plot","y_col":"<col>","x_col":"<col|null>","title":"<str>"}}
Histogram:  {{"action":"histogram","col":"<col>","bins":20,"title":"<str>"}}
Scatter:    {{"action":"scatter","x_col":"<col>","y_col":"<col>","hue_col":"<col|null>","title":"<str>"}}
Regression: {{"action":"linear_regression","x_col":"<col>","y_col":"<col>"}}
KMeans:     {{"action":"kmeans","cols":["<col1>","<col2>"],"k":3}}
Unsupported:{{"action":"unsupported","message":"<str>"}}"""

        resp = llm.invoke([HumanMessage(content=prompt)])
        try:
            action = json.loads(resp.content.strip())
        except Exception:
            return {"error": f"Could not parse action: {resp.content[:200]}"}

        act = action.get("action", "")
        try:
            dispatch = {
                "bar_plot":          self._bar,
                "box_plot":          self._box,
                "histogram":         self._hist,
                "scatter":           self._scatter,
                "linear_regression": lambda d, a: self._regression(d, a, llm),
                "kmeans":            self._kmeans,
            }
            if act in dispatch:
                return dispatch[act](df, action)
            if act == "unsupported":
                return {"type": "unsupported", "text_result": action.get("message")}
            return {"error": f"Unknown action: {act}"}
        except Exception:
            return {"error": traceback.format_exc(limit=3)}

    def _bar(self, df, a):
        x, y, agg = a["x_col"], a.get("y_col"), a.get("agg", "count")
        fig, ax   = _dark_fig()
        if agg == "count" or not y:
            d = df[x].value_counts().head(15)
            ax.bar(d.index.astype(str), d.values, color=ACCENT)
        else:
            fn = {"mean": "mean", "sum": "sum"}.get(agg, "mean")
            d  = df.groupby(x)[y].agg(fn).head(15)
            ax.bar(d.index.astype(str), d.values, color=ACCENT)
            ax.set_ylabel(f"{agg}({y})")
        ax.set_xlabel(x)
        ax.set_title(a.get("title", "Bar Plot"))
        plt.xticks(rotation=40, ha="right")
        return {"type": "bar_plot", "plot_b64": _fig_to_b64(fig)}

    def _box(self, df, a):
        y, x  = a["y_col"], a.get("x_col")
        fig, ax = _dark_fig()
        if x:
            cats = df[x].dropna().unique()[:8]
            data = [df[df[x] == c][y].dropna().values for c in cats]
            bp   = ax.boxplot(data, patch_artist=True, labels=[str(c) for c in cats])
            cmap = plt.cm.get_cmap("coolwarm", len(cats))
            for patch, i in zip(bp["boxes"], range(len(cats))):
                patch.set_facecolor(cmap(i))
            plt.xticks(rotation=30, ha="right")
            ax.set_xlabel(x)
        else:
            bp = ax.boxplot(df[y].dropna().values, patch_artist=True)
            bp["boxes"][0].set_facecolor(ACCENT)
        ax.set_ylabel(y)
        ax.set_title(a.get("title", "Box Plot"))
        return {"type": "box_plot", "plot_b64": _fig_to_b64(fig)}

    def _hist(self, df, a):
        col, bins = a["col"], int(a.get("bins", 20))
        data      = df[col].dropna()
        fig, ax   = _dark_fig()
        ax.hist(data, bins=bins, color=ACCENT, edgecolor=BG, alpha=0.85)
        ax.axvline(data.mean(),   color=WARN,    linestyle="--", linewidth=1.5,
                   label=f"Mean={data.mean():.2f}")
        ax.axvline(data.median(), color=SUCCESS,  linestyle=":",  linewidth=1.5,
                   label=f"Median={data.median():.2f}")
        ax.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(a.get("title", f"Histogram: {col}"))
        return {"type": "histogram", "plot_b64": _fig_to_b64(fig)}

    def _scatter(self, df, a):
        x, y, hue = a["x_col"], a["y_col"], a.get("hue_col")
        fig, ax   = _dark_fig()
        if hue and hue in df.columns:
            cats = df[hue].dropna().unique()
            cmap = plt.cm.get_cmap("tab10", len(cats))
            for i, cat in enumerate(cats):
                mask = df[hue] == cat
                ax.scatter(df.loc[mask, x], df.loc[mask, y],
                           label=str(cat), color=cmap(i), alpha=0.7, s=25)
            ax.legend(title=hue, fontsize=8, facecolor=CARD, labelcolor=TEXT)
        else:
            ax.scatter(df[x], df[y], color=ACCENT, alpha=0.6, s=25)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(a.get("title", "Scatter"))
        return {"type": "scatter", "plot_b64": _fig_to_b64(fig)}

    def _regression(self, df, a, llm):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        x_col, y_col = a["x_col"], a["y_col"]
        clean        = df[[x_col, y_col]].dropna()
        X            = clean[[x_col]].values
        y            = clean[y_col].values
        model        = LinearRegression().fit(X, y)
        y_pred       = model.predict(X)
        r2           = r2_score(y, y_pred)
        rmse         = float(mean_squared_error(y, y_pred) ** 0.5)
        fig, ax      = _dark_fig()
        ax.scatter(X, y, color=ACCENT, alpha=0.5, s=20, label="Actual")
        ax.plot(X, y_pred, color=WARN, linewidth=2, label=f"Fit R²={r2:.3f}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Linear Regression: {x_col} → {y_col}")
        ax.legend(facecolor=CARD, labelcolor=TEXT)
        stats = {"coef": round(float(model.coef_[0]), 6),
                 "intercept": round(float(model.intercept_), 6),
                 "r2": round(r2, 4), "rmse": round(rmse, 4), "n": len(clean)}
        interp = llm.invoke([HumanMessage(
            content=f"Interpret this regression in 2 sentences for a non-technical audience:\n"
                    + json.dumps(stats)
        )])
        return {"type": "linear_regression", "plot_b64": _fig_to_b64(fig),
                "stats": stats, "text_result": interp.content}

    def _kmeans(self, df, a):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        cols, k = a["cols"], int(a.get("k", 3))
        clean   = df[cols].dropna().copy()
        labels  = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(
            StandardScaler().fit_transform(clean))
        clean["cluster"] = labels
        fig, ax = _dark_fig()
        cmap    = plt.cm.get_cmap("tab10", k)
        for c in range(k):
            mask = labels == c
            ax.scatter(clean.iloc[mask, 0], clean.iloc[mask, 1],
                       color=cmap(c), label=f"Cluster {c}", alpha=0.7, s=25)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1] if len(cols) > 1 else cols[0])
        ax.set_title(f"K-Means (k={k})")
        ax.legend(facecolor=CARD, labelcolor=TEXT)
        sizes = {int(c): int((labels == c).sum()) for c in range(k)}
        return {"type": "kmeans", "plot_b64": _fig_to_b64(fig),
                "k": k, "cluster_sizes": sizes,
                "text_result": "K-Means complete. Sizes: "
                               + ", ".join(f"C{c}:{s}" for c, s in sizes.items())}
