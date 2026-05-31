"""
modules/visualization.py — CA Intelligence Suite
Phase 7: Visualization Engine

All Plotly chart functions extracted from app.py. Each chart:
    1. Creates a Plotly figure
    2. Renders it via st.plotly_chart()
    3. Appends a contextual CA interpretation below the chart

This module has zero business logic — it only transforms already-computed
financial data into visual representations.
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import CAT_COLORS, MONO_SEQ, PLOT_LAYOUT
from utils.helpers import inr_format

# Convenience alias
inr = inr_format


# 
# Internal helper — chart interpretation banner
# 
def _render_interpretation(what: str, meaning: str) -> None:
    """Render a two-line interpretive callout below each chart."""
    st.markdown(
        f"""
        <div class="chart-interp">
            <b>What this chart shows:</b> {what}<br>
            <b>What it means:</b> {meaning}
        </div>
        """,
        unsafe_allow_html=True,
    )


# 
# Charts
# 
def chart_income_vs_expense(summary) -> None:
    """
    Side-by-side bar chart: Total Revenue vs Total Expenses.

    Args:
        summary: FinancialSummary from the rule engine.
    """
    fig = go.Figure(go.Bar(
        x=["Total Revenue", "Total Expenses"],
        y=[summary.total_income, summary.total_expense],
        marker_color=["#3b82f6", "#1e40af"],
        text=[inr(summary.total_income), inr(summary.total_expense)],
        textposition="outside",
        textfont=dict(size=11),
        width=0.4,
    ))
    fig.update_layout(
        title="Income vs Expenses",
        **PLOT_LAYOUT,
        xaxis_title="",
        yaxis_title="Amount (₹)",
        showlegend=False,
        yaxis=dict(tickformat=",.0f"),
    )
    st.plotly_chart(fig, use_container_width=True)
    _render_interpretation(
        "A side-by-side comparison of total revenue earned versus total expenses incurred.",
        f"Revenue ({inr(summary.total_income)}) "
        f"{'exceeds' if summary.total_income > summary.total_expense else 'is below'} "
        f"total expenses ({inr(summary.total_expense)}), "
        f"resulting in a net {'profit' if summary.net_profit_loss >= 0 else 'loss'} of "
        f"{inr(abs(summary.net_profit_loss))}. "
        f"Expense-to-income ratio: {summary.total_expense / max(summary.total_income, 1)*100:.1f}%.",
    )


def chart_expense_distribution_pie(df: pd.DataFrame, cat_col: str, summary) -> None:
    """
    Donut pie chart: Expense breakdown by sub-category (top 6).

    Args:
        df:      Full transactions DataFrame.
        cat_col: Category column name.
        summary: FinancialSummary.
    """
    if "Sub_Category" not in df.columns or "Amount" not in df.columns:
        st.info("Sub-category data not available for expense distribution.")
        return

    exp_df = df[df[cat_col] == "Expense"]
    if exp_df.empty:
        return

    top = exp_df.groupby("Sub_Category")["Amount"].sum().sort_values(ascending=False).head(6)
    fig = px.pie(
        values=top.values,
        names=top.index,
        title="Expense Category Distribution",
        color_discrete_sequence=MONO_SEQ[:6],
        hole=0.42,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#161b22", width=2)),
    )
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    top1     = top.index[0] if len(top) > 0 else "N/A"
    top1_pct = top.iloc[0] / max(summary.total_expense, 1) * 100 if len(top) > 0 else 0
    _render_interpretation(
        "Proportional breakdown of total expenses by sub-category for the period.",
        f"'{top1}' is the largest expense driver at {top1_pct:.1f}% of total expenses. "
        "Identifying the top cost heads enables targeted cost control measures.",
    )


def chart_category_bar(df: pd.DataFrame, cat_col: str = "Category") -> None:
    """
    Grouped bar chart: Total amount per category (Income, Expense, Asset, Liability).

    Args:
        df:      Transactions DataFrame.
        cat_col: Category column name.
    """
    if "Amount" not in df.columns or cat_col not in df.columns:
        return
    data = df.groupby(cat_col)["Amount"].sum().reset_index()
    data.columns = ["Category", "Total Amount"]
    data["Color"] = data["Category"].map(CAT_COLORS)

    fig = go.Figure(go.Bar(
        x=data["Category"],
        y=data["Total Amount"],
        marker_color=data["Color"].tolist(),
        text=[inr(v) for v in data["Total Amount"]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title="Category-wise Total Amount",
        **PLOT_LAYOUT,
        xaxis_title="Category",
        yaxis_title="Amount (₹)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    _render_interpretation(
        "Total monetary amount grouped by the four primary financial categories.",
        "A balanced chart with Income > Expense and Assets > Liabilities indicates healthy financials. "
        "A dominant 'Expense' bar relative to 'Income' signals a profitability concern.",
    )


def chart_category_pie(df: pd.DataFrame, cat_col: str = "Category") -> None:
    """
    Donut pie chart: Transaction value share by category.

    Args:
        df:      Transactions DataFrame.
        cat_col: Category column name.
    """
    if "Amount" not in df.columns or cat_col not in df.columns:
        return
    data = df.groupby(cat_col)["Amount"].sum().reset_index()
    fig  = px.pie(
        data,
        values="Amount",
        names=cat_col,
        color=cat_col,
        color_discrete_map=CAT_COLORS,
        hole=0.45,
        title="Transaction Distribution by Category",
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#161b22", width=2)),
    )
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    _render_interpretation(
        "Share of total transaction value across all financial categories.",
        "Expense categories dominating > 60% of total volume may indicate financial imbalance. "
        "Income should ideally represent the largest share for a profitable business.",
    )


def chart_monthly_trend(df: pd.DataFrame, cat_col: str = "Category") -> None:
    """
    Line chart: Month-on-month Income vs Expense trend.

    Args:
        df:      Transactions DataFrame with a 'Date' column.
        cat_col: Category column name.
    """
    if "Date" not in df.columns or "Amount" not in df.columns:
        return

    df2          = df.copy()
    df2["Date"]  = pd.to_datetime(df2["Date"], errors="coerce")
    df2          = df2.dropna(subset=["Date"])
    df2["Month"] = df2["Date"].dt.to_period("M").astype(str)

    cat_filter = [c for c in ["Income", "Expense"] if c in df2[cat_col].unique()]
    if not cat_filter:
        return

    monthly = (
        df2[df2[cat_col].isin(cat_filter)]
        .groupby(["Month", cat_col])["Amount"]
        .sum()
        .reset_index()
    )
    fig = px.line(
        monthly,
        x="Month",
        y="Amount",
        color=cat_col,
        color_discrete_map=CAT_COLORS,
        markers=True,
        title="Monthly Income vs Expense Trend",
    )
    fig.update_traces(line_width=2.5)
    fig.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-45, yaxis_title="Amount (₹)")
    st.plotly_chart(fig, use_container_width=True)
    _render_interpretation(
        "Month-on-month movement of revenue and expenditure over the selected period.",
        "A consistent gap between the Income and Expense lines indicates stable profitability. "
        "Converging or crossing lines indicate deteriorating margins or seasonal volatility.",
    )


def chart_subcategory_expense(df: pd.DataFrame, cat_col: str = "Category") -> None:
    """
    Horizontal bar chart: Top expense sub-categories by amount.

    Args:
        df:      Transactions DataFrame.
        cat_col: Category column name.
    """
    if "Sub_Category" not in df.columns or "Amount" not in df.columns:
        return
    exp = df[df[cat_col] == "Expense"]
    if exp.empty:
        return

    data = (
        exp.groupby("Sub_Category")["Amount"]
        .sum()
        .reset_index()
        .sort_values("Amount", ascending=True)
        .tail(12)
    )
    fig = go.Figure(go.Bar(
        x=data["Amount"],
        y=data["Sub_Category"],
        orientation="h",
        marker=dict(
            color=data["Amount"],
            colorscale=[[0, "#0d1b36"], [0.5, "#2563eb"], [1, "#93c5fd"]],
            showscale=False,
        ),
        text=[inr(v) for v in data["Amount"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top Expense Sub-categories",
        **PLOT_LAYOUT,
        xaxis_title="Amount (₹)",
        yaxis_title="",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
    _render_interpretation(
        "Horizontal bar chart ranking the top expense sub-categories by total amount.",
        "The longest bars represent the biggest cash drains. "
        "These are the primary targets for cost rationalisation and budget cap implementation.",
    )


def chart_payment_mode(df: pd.DataFrame) -> None:
    """
    Bar chart: Total transaction value by payment mode.

    Args:
        df: Transactions DataFrame with 'Payment_Mode' and 'Amount' columns.
    """
    if "Payment_Mode" not in df.columns or "Amount" not in df.columns:
        return
    data = df.groupby("Payment_Mode")["Amount"].sum().reset_index()
    fig  = px.bar(
        data,
        x="Payment_Mode",
        y="Amount",
        color="Payment_Mode",
        color_discrete_sequence=MONO_SEQ[:5],
        title="Amount by Payment Mode",
    )
    fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                      xaxis_title="Payment Mode", yaxis_title="Amount (₹)")
    st.plotly_chart(fig, use_container_width=True)
    _render_interpretation(
        "Total transaction value broken down by the mode of payment used.",
        "A high proportion of cash transactions creates compliance risk under Sections 40A(3) and 269ST. "
        "Digital payment modes provide a verifiable audit trail and are preferred.",
    )


def chart_gst_breakdown(df: pd.DataFrame, cat_col: str = "Category") -> None:
    """
    Donut pie chart: GST amount distribution by tax rate slab.

    Args:
        df:      Transactions DataFrame.
        cat_col: Category column name.
    """
    if "GST_Percentage" not in df.columns:
        return
    gst_df = df[df["GST_Percentage"] > 0]
    if gst_df.empty:
        return
    data = gst_df.groupby("GST_Percentage")["GST_Amount"].sum().reset_index()
    fig  = px.pie(
        data,
        values="GST_Amount",
        names="GST_Percentage",
        title="GST Amount by Rate",
        color_discrete_sequence=MONO_SEQ[:4],
        hole=0.4,
    )
    fig.update_traces(textinfo="percent+label", texttemplate="%{label}%<br>₹%{value:,.0f}")
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)



# ─────────────────────────────────────────────────────────────────────────────
# Advanced Interactive Charts (Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

def chart_sankey_flow(df: pd.DataFrame, cat_col: str = "Category") -> go.Figure:
    """
    Sankey Diagram — Money flow: Income Sources → Category → Sub-Category → GST / Net.

    Insight: Reveals exactly where money enters the business and how it fans out
    across expense heads and tax burden. A thick GST branch signals high tax
    outgo relative to revenue — an immediate flag for CA review.
    """
    import numpy as np

    if not {"Amount", "GST_Amount", "Sub_Category"}.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(title="Sankey: insufficient columns", **PLOT_LAYOUT)
        return fig

    # ── colour map ───────────────────────────────────────────────────────────
    CAT_NODE_COLOR = {
        "Income":    "rgba(16,185,129,0.85)",    # green
        "Expense":   "rgba(239,68,68,0.85)",     # red
        "Asset":     "rgba(37,99,235,0.85)",     # blue
        "Liability": "rgba(139,92,246,0.85)",    # purple
        "GST":       "rgba(245,158,11,0.85)",    # amber
        "Net":       "rgba(96,165,250,0.85)",    # light-blue
    }

    def _inr_hover(v: float) -> str:
        """Indian number format for hover text."""
        try:
            v = float(v)
            sign = "-" if v < 0 else ""
            v = abs(v)
            s = f"{v:,.0f}"
            # Convert Western grouping → Indian grouping
            parts = s.split(",")
            if len(parts) > 1:
                last = parts[-1]
                rest = "".join(parts[:-1])
                # re-group with Indian convention
                rest_rev = rest[::-1]
                groups = [rest_rev[:2]] + [rest_rev[i:i+2] for i in range(2, len(rest_rev), 2)]
                rest = ",".join(g[::-1] for g in groups)[::-1]
                s = rest + "," + last
            return f"₹\u200b{sign}{s}"
        except Exception:
            return f"₹{v}"

    # ── build node list ──────────────────────────────────────────────────────
    categories = [c for c in [cat_col, "Sub_Category"] if c in df.columns]

    grp = df.groupby(categories)[["Amount", "GST_Amount"]].sum().reset_index()
    grp["Net_Amount"] = (grp["Amount"] - grp["GST_Amount"]).clip(lower=0)

    # Node labels (unique)
    cat_nodes   = list(df[cat_col].dropna().unique())
    sub_nodes   = list(df["Sub_Category"].dropna().unique())
    end_nodes   = ["GST Total", "Net Total"]

    all_labels  = cat_nodes + sub_nodes + end_nodes
    idx         = {lbl: i for i, lbl in enumerate(all_labels)}

    # Node colours
    def _node_color(lbl):
        for k, v in CAT_NODE_COLOR.items():
            if lbl.startswith(k) or lbl == k:
                return v
        if "GST" in lbl:
            return CAT_NODE_COLOR["GST"]
        if "Net" in lbl:
            return CAT_NODE_COLOR["Net"]
        return "rgba(100,120,160,0.7)"

    node_colors = [_node_color(l) for l in all_labels]

    # ── build link list ──────────────────────────────────────────────────────
    sources, targets, values, link_labels = [], [], [], []

    # Category → Sub_Category
    cat_sub = df.groupby([cat_col, "Sub_Category"])["Amount"].sum().reset_index()
    for _, row in cat_sub.iterrows():
        cat, sub, amt = row[cat_col], row["Sub_Category"], row["Amount"]
        if cat in idx and sub in idx and amt > 0:
            sources.append(idx[cat])
            targets.append(idx[sub])
            values.append(float(amt))
            link_labels.append(f"{sub}<br>{_inr_hover(amt)}")

    # Sub_Category → GST Total / Net Total
    sub_gst = df.groupby("Sub_Category")[["Amount", "GST_Amount"]].sum().reset_index()
    for _, row in sub_gst.iterrows():
        sub = row["Sub_Category"]
        gst = float(row["GST_Amount"])
        net = max(0.0, float(row["Amount"]) - gst)
        if sub in idx:
            if gst > 0:
                sources.append(idx[sub])
                targets.append(idx["GST Total"])
                values.append(gst)
                link_labels.append(f"GST on {sub}<br>{_inr_hover(gst)}")
            if net > 0:
                sources.append(idx[sub])
                targets.append(idx["Net Total"])
                values.append(net)
                link_labels.append(f"Net from {sub}<br>{_inr_hover(net)}")

    if not values:
        fig = go.Figure()
        fig.update_layout(title="Sankey: no data to render", **PLOT_LAYOUT)
        return fig

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=all_labels,
            color=node_colors,
            pad=20,
            thickness=22,
            line=dict(color="#30363d", width=0.5),
            hovertemplate="%{label}<br>Total: %{value:,.0f}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color="rgba(100,150,255,0.18)",
            hovertemplate="%{label}<extra></extra>",
        ),
    ))
    fig.update_layout(
        title="Money Flow: Income Sources → Categories → Sub-Categories → GST / Net",
        height=540,
        **PLOT_LAYOUT,
    )
    return fig


def chart_network_graph(df: pd.DataFrame, cat_col: str = "Category") -> go.Figure:
    """
    Network Graph — Category hub → Sub-Category satellites → Payment Mode outer ring.

    Insight: Makes clustered spending patterns immediately visible. Sub-categories
    huddled around a dominant Category node reveal concentration risk. Payment-mode
    edges that bypass digital channels surface Section 40A(3) cash-payment exposure.

    Built entirely with Plotly Scatter traces; no external graph libraries required.
    """
    import math
    import numpy as np

    required = {cat_col, "Sub_Category", "Payment_Mode", "Amount"}
    if not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(title="Network Graph: insufficient columns", **PLOT_LAYOUT)
        return fig

    CAT_HEX = {
        "Income":    "#10b981",
        "Expense":   "#ef4444",
        "Asset":     "#2563eb",
        "Liability": "#8b5cf6",
    }
    PAYMENT_COLOR = "#f59e0b"
    SUB_ALPHA     = "rgba(96,165,250,0.85)"

    # ── aggregate ────────────────────────────────────────────────────────────
    cat_totals = df.groupby(cat_col)["Amount"].sum()
    sub_totals = df.groupby("Sub_Category")["Amount"].sum()
    pm_totals  = df.groupby("Payment_Mode")["Amount"].sum()

    cats = list(cat_totals.index)
    subs = list(sub_totals.index)
    pms  = list(pm_totals.index)

    n_cat = len(cats)
    n_sub = len(subs)
    n_pm  = len(pms)

    def _polar(r: float, angle_deg: float):
        a = math.radians(angle_deg)
        return r * math.cos(a), r * math.sin(a)

    # Positions ── three concentric rings
    cat_pos  = {c: _polar(0.0 if n_cat == 1 else 1.5, i * 360 / max(n_cat, 1))
                for i, c in enumerate(cats)}
    sub_pos  = {s: _polar(3.2, i * 360 / max(n_sub, 1))
                for i, s in enumerate(subs)}
    pm_pos   = {p: _polar(5.2, i * 360 / max(n_pm, 1))
                for i, p in enumerate(pms)}

    # Node sizes (log-scaled for readability)
    def _scale(val, lo=12, hi=55):
        vals = list(val.values)
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: (lo + hi) / 2 for k in val.index}
        return {k: lo + (hi - lo) * (v - mn) / (mx - mn) for k, v in val.items()}

    cat_sizes = _scale(cat_totals)
    sub_sizes = _scale(sub_totals, lo=8, hi=36)
    pm_sizes  = _scale(pm_totals,  lo=8, hi=30)

    # ── edges: Category ↔ Sub_Category ──────────────────────────────────────
    cat_sub_vol = df.groupby([cat_col, "Sub_Category"])["Amount"].sum()

    edge_x, edge_y = [], []
    for (cat, sub), vol in cat_sub_vol.items():
        if cat in cat_pos and sub in sub_pos:
            x0, y0 = cat_pos[cat]
            x1, y1 = sub_pos[sub]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    # ── edges: Sub_Category ↔ Payment_Mode ──────────────────────────────────
    sub_pm_vol = df.groupby(["Sub_Category", "Payment_Mode"])["Amount"].sum()

    edge2_x, edge2_y = [], []
    for (sub, pm), vol in sub_pm_vol.items():
        if sub in sub_pos and pm in pm_pos:
            x0, y0 = sub_pos[sub]
            x1, y1 = pm_pos[pm]
            edge2_x += [x0, x1, None]
            edge2_y += [y0, y1, None]

    fig = go.Figure()

    # Edge traces
    for ex, ey, col, name in [
        (edge_x,  edge_y,  "rgba(96,165,250,0.25)",  "Category–SubCat"),
        (edge2_x, edge2_y, "rgba(245,158,11,0.18)",  "SubCat–PayMode"),
    ]:
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines", name=name,
            line=dict(color=col, width=1.5),
            hoverinfo="skip", showlegend=False,
        ))

    # Category nodes
    cx = [cat_pos[c][0] for c in cats]
    cy = [cat_pos[c][1] for c in cats]
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers+text",
        name="Category",
        marker=dict(
            size=[cat_sizes[c] for c in cats],
            color=[CAT_HEX.get(c, "#60a5fa") for c in cats],
            line=dict(color="#e6edf3", width=2),
            symbol="circle",
        ),
        text=cats,
        textposition="top center",
        textfont=dict(size=12, color="#e6edf3", family="Inter"),
        customdata=[f"{inr(cat_totals[c])}" for c in cats],
        hovertemplate="<b>%{text}</b><br>Total: %{customdata}<extra></extra>",
    ))

    # Sub-category nodes
    sx = [sub_pos[s][0] for s in subs]
    sy = [sub_pos[s][1] for s in subs]
    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode="markers+text",
        name="Sub-Category",
        marker=dict(
            size=[sub_sizes[s] for s in subs],
            color=SUB_ALPHA,
            line=dict(color="#30363d", width=1),
        ),
        text=subs,
        textposition="middle right",
        textfont=dict(size=9, color="#8b949e"),
        customdata=[f"{inr(sub_totals[s])}" for s in subs],
        hovertemplate="<b>%{text}</b><br>Total: %{customdata}<extra></extra>",
    ))

    # Payment mode nodes
    px_ = [pm_pos[p][0] for p in pms]
    py_ = [pm_pos[p][1] for p in pms]
    fig.add_trace(go.Scatter(
        x=px_, y=py_, mode="markers+text",
        name="Payment Mode",
        marker=dict(
            size=[pm_sizes[p] for p in pms],
            color=PAYMENT_COLOR,
            opacity=0.80,
            symbol="diamond",
            line=dict(color="#30363d", width=1),
        ),
        text=pms,
        textposition="bottom center",
        textfont=dict(size=9, color="#f59e0b"),
        customdata=[f"{inr(pm_totals[p])}" for p in pms],
        hovertemplate="<b>%{text}</b><br>Volume: %{customdata}<extra></extra>",
    ))

    fig.update_layout(
        title="Transaction Network: Categories → Sub-Categories → Payment Modes",
        height=620,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        legend=dict(
            font=dict(color="#e6edf3", size=11),
            bgcolor="rgba(28,34,48,0.8)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        **PLOT_LAYOUT,
    )
    return fig


def chart_sunburst_hierarchy(df: pd.DataFrame, cat_col: str = "Category") -> go.Figure:
    """
    Sunburst Chart — Category → Sub-Category → Payment_Mode hierarchy.

    Insight: The outer rings magnify sub-category and payment-mode granularity.
    Sectors coloured by GST-ratio intensity (amber = high GST burden) let a CA
    immediately spot which expense heads carry the heaviest indirect-tax load
    — critical for ITC reconciliation and GST audit planning.
    """
    required = {cat_col, "Sub_Category", "Payment_Mode", "Amount", "GST_Amount"}
    if not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(title="Sunburst: insufficient columns", **PLOT_LAYOUT)
        return fig

    agg = (
        df.groupby([cat_col, "Sub_Category", "Payment_Mode"])[["Amount", "GST_Amount"]]
        .sum()
        .reset_index()
    )
    agg = agg[agg["Amount"] > 0].copy()

    # GST ratio 0-1 for colour intensity
    agg["gst_ratio"] = (agg["GST_Amount"] / agg["Amount"].replace(0, float("nan"))).fillna(0).clip(0, 1)

    # Hover text with Indian INR + percentage
    total_all = agg["Amount"].sum()
    agg["pct"] = (agg["Amount"] / total_all * 100).round(2)

    def _hover(row):
        return (
            f"<b>{row[cat_col]} › {row['Sub_Category']} › {row['Payment_Mode']}</b><br>"
            f"Amount: {inr(row['Amount'])}<br>"
            f"GST: {inr(row['GST_Amount'])} ({row['gst_ratio']*100:.1f}%)<br>"
            f"Share of Total: {row['pct']:.2f}%"
        )

    agg["hover"] = agg.apply(_hover, axis=1)

    # Colour intensity: map gst_ratio to an amber-to-red scale
    # Use numeric colour values (0 = low GST, 1 = high GST)
    colors = agg["gst_ratio"].tolist()

    fig = go.Figure(go.Sunburst(
        ids=[
            f"{r[cat_col]}|{r['Sub_Category']}|{r['Payment_Mode']}"
            for _, r in agg.iterrows()
        ],
        labels=[r["Payment_Mode"] for _, r in agg.iterrows()],
        parents=[
            f"{r[cat_col]}|{r['Sub_Category']}"
            for _, r in agg.iterrows()
        ],
        values=agg["Amount"].tolist(),
        customdata=agg[["hover", "gst_ratio"]].values,
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(
            colors=colors,
            colorscale=[
                [0.0,  "#1e3a5f"],
                [0.33, "#2563eb"],
                [0.66, "#f59e0b"],
                [1.0,  "#ef4444"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="GST Ratio", font=dict(color="#8b949e", size=11)),
                tickfont=dict(color="#8b949e"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#30363d",
                thickness=14,
                len=0.6,
            ),
        ),
        branchvalues="total",
        insidetextorientation="radial",
        textfont=dict(family="Inter", size=11, color="#e6edf3"),
        # Parent rings
        maxdepth=3,
    ))

    # Add parent (Category) and mid (Sub_Category) rings explicitly
    # via additional traces piggybacked as a second sunburst layer
    cat_agg = agg.groupby(cat_col)[["Amount", "GST_Amount"]].sum().reset_index()
    cat_agg["gst_ratio"] = (cat_agg["GST_Amount"] / cat_agg["Amount"].replace(0, float("nan"))).fillna(0)

    sub_agg = agg.groupby([cat_col, "Sub_Category"])[["Amount", "GST_Amount"]].sum().reset_index()
    sub_agg["gst_ratio"] = (sub_agg["GST_Amount"] / sub_agg["Amount"].replace(0, float("nan"))).fillna(0)

    # Build a single unified sunburst with all three levels
    ids, labels, parents, vals, col_vals, hover_vals = [], [], [], [], [], []

    # Level 1 — Category
    for _, r in cat_agg.iterrows():
        cat = r[cat_col]
        ids.append(cat)
        labels.append(cat)
        parents.append("")
        vals.append(float(r["Amount"]))
        col_vals.append(float(r["gst_ratio"]))
        hover_vals.append(
            f"<b>{cat}</b><br>Total: {inr(r['Amount'])}<br>"
            f"GST Burden: {r['gst_ratio']*100:.1f}%"
        )

    # Level 2 — Sub_Category
    for _, r in sub_agg.iterrows():
        cat, sub = r[cat_col], r["Sub_Category"]
        ids.append(f"{cat}|{sub}")
        labels.append(sub)
        parents.append(cat)
        vals.append(float(r["Amount"]))
        col_vals.append(float(r["gst_ratio"]))
        hover_vals.append(
            f"<b>{sub}</b><br>Total: {inr(r['Amount'])}<br>"
            f"GST: {inr(r['GST_Amount'])} ({r['gst_ratio']*100:.1f}%)"
        )

    # Level 3 — Payment_Mode
    for _, r in agg.iterrows():
        cat, sub, pm = r[cat_col], r["Sub_Category"], r["Payment_Mode"]
        ids.append(f"{cat}|{sub}|{pm}")
        labels.append(pm)
        parents.append(f"{cat}|{sub}")
        vals.append(float(r["Amount"]))
        col_vals.append(float(r["gst_ratio"]))
        hover_vals.append(r["hover"])

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=vals,
        customdata=hover_vals,
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(
            colors=col_vals,
            colorscale=[
                [0.0,  "#1e3a5f"],
                [0.33, "#2563eb"],
                [0.66, "#f59e0b"],
                [1.0,  "#ef4444"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="GST Ratio", font=dict(color="#8b949e", size=11)),
                tickfont=dict(color="#8b949e"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#30363d",
                thickness=14,
                len=0.6,
            ),
        ),
        branchvalues="remainder",
        insidetextorientation="radial",
        textfont=dict(family="Inter", size=11),
        maxdepth=3,
    ))

    fig.update_layout(
        title="Hierarchy: Category → Sub-Category → Payment Mode  (colour = GST burden)",
        height=620,
        **PLOT_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Gauge (unchanged — kept at bottom)
# ─────────────────────────────────────────────────────────────────────────────
def chart_compliance_score_gauge(compliance_score: int, grade: str) -> None:
    """
    Gauge chart displaying the computed compliance score.

    Args:
        compliance_score: Integer 0–100.
        grade:            Letter grade (A/B/C/D/F).
    """
    color = (
        "#10b981" if compliance_score >= 75 else
        "#f59e0b" if compliance_score >= 50 else "#ef4444"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=compliance_score,
        title={"text": f"Compliance Score (Grade: {grade})", "font": {"size": 16, "color": "#e6edf3"}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8b949e"},
            "bar":   {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 75], "color": "rgba(245,158,11,0.15)"},
                {"range": [75, 100],"color": "rgba(16,185,129,0.15)"},
            ],
        },
        number={"suffix": "/100", "font": {"color": color, "size": 28}},
    ))
    fig.update_layout(**PLOT_LAYOUT, height=280)
    st.plotly_chart(fig, use_container_width=True)
