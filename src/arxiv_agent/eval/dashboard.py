"""A self-contained HTML page over the evaluation runs in ``eval/runs/``.

Retrieval runs (``retrieval.json``) and answer runs (``summary.json``) are read
as written. The one thing computed here is how far each headline can be
trusted: a 95% bootstrap interval from the per-question values in the same
files. The page has no scripts: bars are inline SVG with native hover titles,
and every number is also in a table.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arxiv_agent.eval.ragas_eval import METRICS, is_scored
from arxiv_agent.eval.stats import bootstrap_ci

_RETRIEVAL_METRICS = (("mrr", "MRR@5"), ("hit_rate", "Hit@5"), ("recall", "Recall@5"))
_MODES = (("dense", "var(--series-1)"), ("two-stage", "var(--series-2)"))

_STYLE = """
:root { color-scheme: light;
  --surface: #fcfcfb; --page: #f4f3ef; --text: #0b0b0b; --text-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
  color-scheme: dark;
  --surface: #1a1a19; --page: #111110; --text: #f0efec; --text-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; } }
:root[data-theme="dark"] { color-scheme: dark;
  --surface: #1a1a19; --page: #111110; --text: #f0efec; --text-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 0 0 12px; }
.sub { color: var(--text-2); margin: 0 0 24px; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin-bottom: 16px; overflow-x: auto; }
.note { color: var(--text-2); font-size: 13px; margin: 8px 0 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid);
  vertical-align: top; }
th { color: var(--text-2); font-weight: 600; } td.num, th.num { text-align: right;
  font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 16px; color: var(--text-2); font-size: 13px; margin-bottom: 8px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 6px; vertical-align: -1px; }
svg text { fill: var(--text-2); font-size: 12px; }
svg .value { fill: var(--text); font-variant-numeric: tabular-nums; }
.stats { display: flex; flex-wrap: wrap; gap: 24px; margin-top: 12px; }
.stat b { display: block; font-size: 18px; font-variant-numeric: tabular-nums; }
.stat span { color: var(--text-2); font-size: 12px; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if is_scored(value) else "-"


def _load(run_dirs: Sequence[Path]) -> tuple[list[tuple[str, dict[str, Any]]], ...]:
    retrieval, answers = [], []
    for run in sorted(run_dirs, key=lambda p: p.name):
        if (run / "retrieval.json").exists():
            retrieval.append((run.name, json.loads((run / "retrieval.json").read_text())))
        if (run / "summary.json").exists():
            answers.append((run.name, json.loads((run / "summary.json").read_text())))
    return retrieval, answers


def _bars(rows: Sequence[tuple[str, Sequence[tuple[str, float | None, str]]]]) -> str:
    """Horizontal bars on a shared 0-1 scale. rows: (label, [(series, value, color)])."""
    label_w, plot_w, bar_h, gap, group_gap = 150, 520, 14, 2, 14
    per_group = max((len(series) for _, series in rows), default=1)
    group_h = per_group * bar_h + (per_group - 1) * gap
    height = len(rows) * (group_h + group_gap) + 20
    parts = [f'<svg viewBox="0 0 {label_w + plot_w + 60} {height}" width="100%" role="img">']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = label_w + tick * plot_w
        parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{height - 18}" stroke="var(--grid)"/>'
            f'<text x="{x}" y="{height - 4}" text-anchor="middle">{tick:g}</text>'
        )
    y = 0
    for label, series in rows:
        parts.append(
            f'<text x="{label_w - 10}" y="{y + group_h / 2 + 4}" text-anchor="end">'
            f"{_esc(label)}</text>"
        )
        for name, value, color in series:
            width = max(0.0, min(1.0, value or 0.0)) * plot_w
            title = f"{label} / {name}: {_fmt(value)}"
            parts.append(
                f'<rect x="{label_w}" y="{y}" width="{width:.1f}" height="{bar_h}" rx="3" '
                f'fill="{color}"><title>{_esc(title)}</title></rect>'
                f'<text class="value" x="{label_w + width + 6:.1f}" y="{y + bar_h - 3}">'
                f"{_fmt(value)}</text>"
            )
            y += bar_h + gap
        y += group_gap - gap
    parts.append("</svg>")
    return "".join(parts)


def _gain_note(modes: Mapping[str, Any]) -> str:
    """Two-stage minus dense MRR@5, with an interval paired by question."""
    if "dense" not in modes or "two-stage" not in modes:
        return ""
    rr = [
        {item["id"]: 1 / item["rank"] if item["rank"] else 0.0 for item in modes[m]["items"]}
        for m in ("dense", "two-stage")
    ]
    ids = sorted(rr[0].keys() & rr[1].keys())
    if not ids:
        return ""
    # The gain and its interval come from the same differences, so they cannot disagree.
    diffs = [rr[1][i] - rr[0][i] for i in ids]
    low, high = bootstrap_ci(diffs)
    return (
        f'<p class="note">Two-stage minus dense, MRR@5: <b>{sum(diffs) / len(diffs):+.3f}</b> '
        f"(95% interval {low:+.3f} to {high:+.3f}, paired bootstrap over "
        f"{len(ids)} question{'s' if len(ids) != 1 else ''}).</p>"
    )


def _retrieval_section(runs: Sequence[tuple[str, dict[str, Any]]]) -> str:
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, data in runs:  # sorted by name, which starts with a timestamp
        latest[str(data["meta"].get("dataset", "?"))] = (name, data)
    blocks = []
    for dataset, (name, data) in sorted(latest.items(), key=lambda kv: kv[0] != "test.jsonl"):
        modes: Mapping[str, Any] = data["modes"]
        rows = [
            (
                label,
                [(m, modes[m][key], color) for m, color in _MODES if m in modes],
            )
            for key, label in _RETRIEVAL_METRICS
        ]
        meta = data["meta"]
        blocks.append(
            f"<h2>Retrieval, {_esc(dataset)}</h2>"
            '<div class="legend">'
            + "".join(
                f'<span><span class="swatch" style="background:{c}"></span>{_esc(m)}</span>'
                for m, c in _MODES
            )
            + "</div>"
            + _bars(rows)
            + _gain_note(modes)
            + f'<p class="note">Run {_esc(name)}, chunks {_esc(meta.get("chunk_size"))}/'
            f"{_esc(meta.get('chunk_overlap'))}, commit {_esc(str(meta.get('git_sha'))[:7])}. "
            "Relevance is judged by evidence quotes, without a model.</p>"
        )
    return '<section id="retrieval">' + "".join(blocks) + "</section>" if blocks else ""


def _answers_section(runs: Sequence[tuple[str, dict[str, Any]]]) -> str:
    if not runs:
        return ""
    # The headline is the newest test-set run; a dev run only when there is none.
    test_runs = [r for r in runs if r[1]["meta"].get("dataset") == "test.jsonl"]
    name, data = (test_runs or list(runs))[-1]
    ragas, ops, meta = data["ragas"], data["operations"], data["meta"]
    rows = [(m, [("latest", ragas["mean"].get(m), "var(--series-1)")]) for m in METRICS]
    unscored = ", ".join(f"{m} {n}" for m, n in ragas["unscored"].items() if n) or "none"
    stats = [
        (f"{ops['fallback_rate']:.0%}", "fell back to direct retrieval"),
        (f"{ops['mean_iterations']:.2f}", "agent calls per question (excl. fallback)"),
        (f"{ops['latency_p50_s']:.0f} s", "median latency"),
        (f"{ops['cited_rate']:.0%}", "answers with a verified citation"),
    ]
    intervals = ""
    for m in METRICS:
        scored = [s for item in data["items"] if is_scored(s := item["scores"].get(m))]
        span = "{:.3f} to {:.3f}".format(*bootstrap_ci(scored)) if scored else "-"
        intervals += (
            f'<tr><td>{_esc(m)}</td><td class="num">{_fmt(ragas["mean"].get(m))}</td>'
            f'<td class="num">{span}</td>'
            f'<td class="num">{len(scored)} of {len(data["items"])}</td></tr>'
        )
    head = "".join(f'<th class="num">{_esc(m)}</th>' for m in METRICS)
    body = "".join(
        "<tr>"
        f"<td>{_esc(item['id'])}</td><td>{_esc(item['question'])}</td>"
        f"<td>{_esc(item['reason'] or 'answered')}</td>"
        f'<td class="num">{item["iterations"]}</td>'
        f'<td class="num">{item["latency_s"]:.0f}</td>'
        f'<td class="num">{item["citations"]}</td>'
        + "".join(f'<td class="num">{_fmt(item["scores"].get(m))}</td>' for m in METRICS)
        + "</tr>"
        for item in data["items"]
    )
    return (
        '<section id="answers">'
        f"<h2>Answers, {_esc(meta.get('dataset'))} ({ragas['n']} questions)</h2>"
        + _bars(rows)
        + "<table><thead><tr><th>metric</th>"
        '<th class="num">mean</th><th class="num">95% interval</th>'
        f'<th class="num">scored</th></tr></thead><tbody>{intervals}</tbody></table>'
        + f'<p class="note">RAGAS means over scored questions; unscored by the judge: '
        f"{_esc(unscored)}. Agent {_esc(meta.get('agent_model'))}, judge "
        f"{_esc(meta.get('judge_model'))}, run {_esc(name)}, commit "
        f"{_esc(str(meta.get('git_sha'))[:7])}.</p>"
        '<div class="stats">'
        + "".join(f'<div class="stat"><b>{v}</b><span>{_esc(t)}</span></div>' for v, t in stats)
        + "</div></section>"
        '<section id="questions"><h2>Per question</h2><table><thead><tr>'
        "<th>id</th><th>question</th><th>outcome</th>"
        '<th class="num">calls</th><th class="num">s</th><th class="num">cites</th>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></section>"
    )


def _history_section(
    retrieval: Sequence[tuple[str, dict[str, Any]]], answers: Sequence[tuple[str, dict[str, Any]]]
) -> str:
    rows = []
    for name, data in retrieval:
        modes = data["modes"]
        headline = ", ".join(f"{m} MRR {_fmt(modes[m]['mrr'])}" for m, _ in _MODES if m in modes)
        rows.append((name, "retrieval", data["meta"], headline))
    for name, data in answers:
        mean = data["ragas"]["mean"]
        headline = ", ".join(f"{m} {_fmt(mean.get(m))}" for m in METRICS)
        rows.append((name, "answers", data["meta"], headline))
    body = "".join(
        f"<tr><td>{_esc(name)}</td><td>{kind}</td><td>{_esc(meta.get('dataset'))}</td>"
        f"<td>{_esc(str(meta.get('git_sha'))[:7])}</td><td>{_esc(headline)}</td></tr>"
        for name, kind, meta, headline in sorted(rows, reverse=True)
    )
    return (
        '<section id="history"><h2>All runs</h2><table><thead><tr><th>run</th><th>kind</th>'
        "<th>set</th><th>commit</th><th>headline</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def render_dashboard(run_dirs: Sequence[Path]) -> str:
    retrieval, answers = _load(run_dirs)
    if not retrieval and not answers:
        content = '<section><p class="note">No evaluation runs yet.</p></section>'
    else:
        content = (
            _answers_section(answers)
            + _retrieval_section(retrieval)
            + _history_section(retrieval, answers)
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Evaluation dashboard</title><style>{_STYLE}</style></head><body><main>"
        "<h1>arxiv-research-agent evaluation</h1>"
        '<p class="sub">Generated from eval/runs/ by arxiv-agent eval dashboard.</p>'
        f"{content}</main></body></html>\n"
    )
