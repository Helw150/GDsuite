#!/usr/bin/env python3
"""Plot Delphi GDsuite results with OpenAthena-style Plotly figures.

Uses only the blog-style GDsuite metrics:
- hard_acc for the five logprob families, match_rate for persona QA
- prob_margin = P(expected) - P(parrot) for the five logprob families,
  match_rate for persona QA

Writes Plotly JSON, standalone HTML, and PNG files.
"""

from __future__ import annotations

from pathlib import Path
import re

from datasets import load_dataset
from huggingface_hub import HfApi
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True)

REPO = "WillHeld/gdsuite-delphi-result"
COLLECTION = "marin-community/delphi-69f93cbd09845c03b070bae9"

INK = "#1f1e1b"
ACCENT = "#9e6d43"
BODY_FONT = "Noto Sans Display, Noto Sans, Lato, -apple-system, BlinkMacSystemFont, sans-serif"
HEAD_FONT = "Noto Sans Display, Noto Sans, Lato, -apple-system, BlinkMacSystemFont, sans-serif"
OA_COLORWAY = [
    "#9e6d43",
    "#2d4a3e",
    "#7a3b2e",
    "#4a5d8a",
    "#6b5b3e",
    "#8b3a62",
    "#3d5a4f",
    "#a86a2c",
]

FAMILY_LABELS = {
    "flipped_answer": "Flipped Answer",
    "intuitive_answer": "Intuitive Answer",
    "multihop_persona_qa": "Multi-hop Persona QA",
    "repetitive_answer": "Repetitive Answer",
    "successive_answer": "Successive Answer",
    "truthy_answer": "Truthy Answer",
}
FAMILIES = list(FAMILY_LABELS)


def parse_compute(model: str) -> float:
    return float(model.split("-")[1].replace("e", "E"))


def parse_num(s: str) -> float:
    x = float(re.match(r"([0-9.]+)", s).group(1))
    if "B" in s:
        return x * 1e9
    if "M" in s:
        return x * 1e6
    return x


def parse_params(model: str) -> float:
    return parse_num(re.search(r"([0-9.]+[BM])params", model).group(1))


def parse_tokens(model: str) -> float:
    return parse_num(re.search(r"([0-9.]+B)tokens", model).group(1))


def flops_label(x: float) -> str:
    if x >= 1e23:
        return f"{x / 1e23:g}e23"
    if x >= 1e22:
        return f"{x / 1e22:g}e22"
    if x >= 1e21:
        return f"{x / 1e21:g}e21"
    if x >= 1e20:
        return f"{x / 1e20:g}e20"
    return f"{x / 1e18:g}e18"


def params_label(x: float) -> str:
    return f"{x / 1e9:g}B" if x >= 1e9 else f"{x / 1e6:g}M"


def metric_filter(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return rows for a blog-style metric.

    Persona QA is generative, so it always uses match_rate.
    """
    return df[
        ((df.family != "multihop_persona_qa") & (df.metric == metric))
        | ((df.family == "multihop_persona_qa") & (df.metric == "match_rate"))
    ].copy()


def apply_oa_layout(fig: go.Figure, height: int, top_margin: int = 72) -> None:
    axis_defaults = dict(
        gridcolor="rgba(31,30,27,0.12)",
        zerolinecolor="rgba(31,30,27,0.25)",
        linecolor="rgba(31,30,27,0.55)",
        tickcolor="rgba(31,30,27,0.55)",
        tickfont=dict(family=BODY_FONT, color=INK, size=16),
        title=dict(font=dict(family=BODY_FONT, color=INK, size=18)),
        ticks="outside",
        automargin=True,
    )
    fig.update_layout(
        colorway=OA_COLORWAY,
        font=dict(family=BODY_FONT, color=INK, size=16),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(31,30,27,0.2)",
            borderwidth=1,
            font=dict(family=BODY_FONT, color=INK, size=15),
        ),
        hoverlabel=dict(
            bgcolor=INK,
            bordercolor=ACCENT,
            font=dict(family=BODY_FONT, color="#f5efe6", size=15),
        ),
        height=height,
        margin=dict(t=top_margin, r=28, b=72, l=72),
        hovermode="closest",
    )
    fig.update_xaxes(**axis_defaults)
    fig.update_yaxes(**axis_defaults)


def write_fig(fig: go.Figure, stem: str) -> None:
    json_path = OUT / f"{stem}.json"
    html_path = OUT / f"{stem}.html"
    png_path = OUT / f"{stem}.png"
    fig.write_json(json_path)
    fig.write_html(html_path, include_plotlyjs="cdn", config={"responsive": True, "displaylogo": False})
    fig.write_image(png_path, width=1500, height=800, scale=2)
    print("wrote", json_path)
    print("wrote", html_path)
    print("wrote", png_path)


def write_fig_size(fig: go.Figure, stem: str, width: int, height: int) -> None:
    json_path = OUT / f"{stem}.json"
    html_path = OUT / f"{stem}.html"
    png_path = OUT / f"{stem}.png"
    fig.write_json(json_path)
    fig.write_html(html_path, include_plotlyjs="cdn", config={"responsive": True, "displaylogo": False})
    fig.write_image(png_path, width=width, height=height, scale=2)
    print("wrote", json_path)
    print("wrote", html_path)
    print("wrote", png_path)


def family_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = metric_filter(df, metric)
    return (
        rows.groupby(["model", "compute_flops", "params", "tokens", "family", "metric"], as_index=False)
        .agg(score_mean=("score", "mean"), task_count=("task", "nunique"), n_total=("n", "sum"))
    )


def family_y_title(family: str, metric: str) -> str:
    if family == "multihop_persona_qa":
        return "match_rate"
    return metric


def metric_y_range(metric: str) -> list[float]:
    if metric == "hard_acc":
        return [0.0, 1.0]
    if metric == "prob_margin":
        return [-0.5, 0.5]
    return [0.0, 0.5]


def plot_grid(fam: pd.DataFrame, stem: str, metric: str) -> None:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[FAMILY_LABELS[f] for f in FAMILIES],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )
    compute_values = sorted(fam.compute_flops.unique())
    color_by_compute = {c: OA_COLORWAY[i % len(OA_COLORWAY)] for i, c in enumerate(compute_values)}
    for idx, family in enumerate(FAMILIES):
        row = idx // 3 + 1
        col = idx % 3 + 1
        sub_family = fam[fam.family == family]
        for cval, sub in sub_family.groupby("compute_flops"):
            sub = sub.sort_values("params")
            fig.add_trace(
                go.Scatter(
                    x=sub.params,
                    y=sub.score_mean,
                    mode="lines+markers",
                    name=flops_label(cval),
                    legendgroup=flops_label(cval),
                    showlegend=(idx == 0),
                    marker=dict(size=8, color=color_by_compute[cval]),
                    line=dict(width=2.6, color=color_by_compute[cval]),
                    customdata=list(zip(sub.model, sub.tokens, sub.task_count, sub.metric)),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "params=%{x:.3s}<br>tokens=%{customdata[1]:.3s}<br>"
                        "score=%{y:.4f}<br>tasks=%{customdata[2]}<br>metric=%{customdata[3]}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(type="log", title_text="Parameters", row=row, col=col)
        fig.update_yaxes(
            title_text=family_y_title(family, metric),
            range=metric_y_range(metric),
            row=row,
            col=col,
        )
    apply_oa_layout(fig, height=760, top_margin=72)
    write_fig(fig, stem)


def plot_optima(fam: pd.DataFrame, opt_models: list[str], stem: str, metric: str) -> None:
    opt = fam[fam.model.isin(opt_models)].copy()
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[FAMILY_LABELS[f] for f in FAMILIES],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )
    for idx, family in enumerate(FAMILIES):
        row = idx // 3 + 1
        col = idx % 3 + 1
        sub = opt[opt.family == family].sort_values("compute_flops")
        fig.add_trace(
            go.Scatter(
                x=sub.compute_flops,
                y=sub.score_mean,
                mode="lines+markers+text",
                text=[params_label(p) for p in sub.params],
                textposition="top center",
                textfont=dict(size=14, color=INK),
                marker=dict(size=9, color=OA_COLORWAY[0]),
                line=dict(width=2.2, color=OA_COLORWAY[0]),
                showlegend=False,
                customdata=list(zip(sub.model, sub.params, sub.tokens, sub.task_count, sub.metric)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "compute=%{x:.3s}<br>params=%{customdata[1]:.3s}<br>tokens=%{customdata[2]:.3s}<br>"
                    "score=%{y:.4f}<br>tasks=%{customdata[3]}<br>metric=%{customdata[4]}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(type="log", title_text="Training compute FLOPs", row=row, col=col)
        fig.update_yaxes(
            title_text=family_y_title(family, metric),
            range=metric_y_range(metric),
            row=row,
            col=col,
        )
    apply_oa_layout(fig, height=760, top_margin=72)
    write_fig(fig, stem)


def plot_twitter_optima(fam: pd.DataFrame, opt_models: list[str], stem: str, metric: str) -> None:
    opt = fam[fam.model.isin(opt_models)].copy()
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[FAMILY_LABELS[f] for f in FAMILIES],
        horizontal_spacing=0.075,
        vertical_spacing=0.16,
    )
    for idx, family in enumerate(FAMILIES):
        row = idx // 3 + 1
        col = idx % 3 + 1
        sub = opt[opt.family == family].sort_values("compute_flops")
        fig.add_trace(
            go.Scatter(
                x=sub.compute_flops,
                y=sub.score_mean,
                mode="lines+markers",
                marker=dict(size=11, color=OA_COLORWAY[0]),
                line=dict(width=3.2, color=OA_COLORWAY[0]),
                showlegend=False,
                customdata=list(zip(sub.model, sub.params, sub.tokens, sub.metric)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "compute=%{x:.3s}<br>params=%{customdata[1]:.3s}<br>tokens=%{customdata[2]:.3s}<br>"
                    "score=%{y:.4f}<br>metric=%{customdata[3]}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        # Label only the endpoints and the largest visible transition to keep
        # the image readable in compressed social previews.
        label_rows = pd.concat([sub.head(1), sub.tail(3)]).drop_duplicates("model")
        for _, point in label_rows.iterrows():
            fig.add_annotation(
                x=point.compute_flops,
                y=point.score_mean,
                text=params_label(point.params),
                showarrow=False,
                xshift=0,
                yshift=14,
                font=dict(family=BODY_FONT, size=17, color=INK),
                row=row,
                col=col,
            )
        fig.update_xaxes(
            type="log",
            title_text="Compute",
            tickvals=[3e18, 3e20, 1e22, 1e23],
            ticktext=["3e18", "3e20", "1e22", "1e23"],
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text=family_y_title(family, metric),
            range=metric_y_range(metric),
            row=row,
            col=col,
        )
    apply_oa_layout(fig, height=760, top_margin=58)
    fig.update_layout(
        width=1600,
        height=900,
        margin=dict(t=58, r=26, b=70, l=70),
        font=dict(family=BODY_FONT, color=INK, size=22),
    )
    fig.update_annotations(font=dict(family=HEAD_FONT, size=26, color=INK))
    fig.update_xaxes(tickfont=dict(size=21), title_font=dict(size=23))
    fig.update_yaxes(tickfont=dict(size=21), title_font=dict(size=23))
    write_fig_size(fig, stem, width=1600, height=900)


def main() -> None:
    api = HfApi()
    collection = api.get_collection(COLLECTION)
    collection_models = [
        getattr(item, "item_id").split("/")[-1]
        for item in collection.items
        if getattr(item, "item_type", None) == "model"
    ]
    nonseed_models = [m for m in collection_models if "-seed" not in m]
    opt_models = []
    for item_idx, item in enumerate(collection.items):
        if getattr(item, "item_type", None) != "model":
            continue
        model = getattr(item, "item_id").split("/")[-1]
        if "-seed" in model:
            continue
        if item_idx in (0, 1, 4) or 8 <= item_idx <= 14:
            opt_models.append(model)

    df = load_dataset(REPO, split="train", download_mode="force_redownload").to_pandas()
    df = df[df.model.isin(nonseed_models)].copy()
    df["compute_flops"] = df.model.map(parse_compute)
    df["params"] = df.model.map(parse_params)
    df["tokens"] = df.model.map(parse_tokens)

    specs = [
        (
            "hard_acc",
            "accuracy_like",
            "hard_acc",
        ),
        (
            "prob_margin",
            "prob_margin",
            "P(expected) - P(parrot)",
        ),
    ]
    for metric, stem_metric, y_title in specs:
        fam = family_means(df, metric)
        fam.to_csv(OUT / f"delphi_blog_{stem_metric}_family_means.csv", index=False)
        plot_grid(
            fam,
            f"delphi_plotly_blog_{stem_metric}_grid_all_nonseed",
            metric,
        )
        plot_optima(
            fam,
            opt_models,
            f"delphi_plotly_blog_{stem_metric}_collection_optima_to_1e23",
            metric,
        )
        plot_twitter_optima(
            fam,
            opt_models,
            f"delphi_twitter_blog_{stem_metric}_collection_optima_to_1e23",
            metric,
        )

    print("non-seed models", df.model.nunique())
    print("collection optima", len(opt_models))


if __name__ == "__main__":
    main()
