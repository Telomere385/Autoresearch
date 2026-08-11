import argparse
import html
import json
from pathlib import Path
import time
import webbrowser


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
REPORTS_DIR = BASE_DIR / "reports"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latest_run_dir():
    candidates = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "state.json").exists()]
    if not candidates:
        raise FileNotFoundError("No completed Agent run found under runs/.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_summary(path):
    if path and Path(path).exists():
        return read_json(Path(path))
    return {}


def collect_experiments(run_dir, decisions):
    rows = []
    baseline_summary = load_summary(run_dir / "baseline" / "summary.json")
    if baseline_summary:
        rows.append({
            "label": "baseline",
            "iteration": 0,
            "decision": "baseline",
            "reason": "initial configuration",
            "changes": {},
            "summary": baseline_summary,
        })
    for item in decisions:
        if int(item.get("iteration", 0)) <= 0 or "summary" not in item:
            continue
        rows.append({
            "label": f"iteration_{int(item.get('iteration')):03d}",
            "iteration": int(item.get("iteration")),
            "decision": item.get("decision", "unknown"),
            "reason": item.get("reason", ""),
            "changes": item.get("candidate", {}).get("changes", {}),
            "summary": item.get("summary", {}),
            "reflection": item.get("reflection", {}),
        })
    return rows


def metric(summary, name):
    value = summary.get(name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def esc(value):
    return html.escape(str(value), quote=True)


def bar_chart(rows, metric_name):
    if not rows:
        return "<p>No experiment rows available.</p>"
    values = [metric(row["summary"], metric_name) for row in rows]
    max_value = max(values + [1.0])
    bars = []
    for row, value in zip(rows, values):
        height = max(3, int(170 * value / max_value)) if max_value > 0 else 3
        color = "#16a34a" if row["decision"] == "accept" else "#dc2626" if row["decision"] == "reject" else "#2563eb"
        bars.append(
            f"""
            <div class="bar-item">
              <div class="bar-value">{fmt(value, 2)}</div>
              <div class="bar" style="height:{height}px;background:{color}"></div>
              <div class="bar-label">{esc(row["label"])}</div>
            </div>
            """
        )
    return f'<div class="bar-chart">{"".join(bars)}</div>'


def progress_svg(rows, metric_name, title):
    series = []
    for row in rows:
        progress = row["summary"].get("train_result", {}).get("progress", [])
        points = []
        for item in progress:
            episode = float(item.get("episode", 0))
            key = "average_score" if metric_name == "score" else "average_reward"
            points.append((episode, float(item.get(key, 0.0))))
        if points:
            series.append((row["label"], row["decision"], points))
    if not series:
        return '<div class="empty-chart">No progress points were recorded.</div>'

    all_x = [x for _, _, pts in series for x, _ in pts]
    all_y = [y for _, _, pts in series for _, y in pts]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0

    def sx(x):
        return 50 + (x - min_x) * 680 / (max_x - min_x)

    def sy(y):
        return 230 - (y - min_y) * 180 / (max_y - min_y)

    colors = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]
    paths = []
    legend = []
    for i, (label, decision, pts) in enumerate(series):
        color = colors[i % len(colors)]
        d = " ".join(("M" if j == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(pts))
        paths.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5" />')
        legend.append(f'<span><i style="background:{color}"></i>{esc(label)} ({esc(decision)})</span>')

    return f"""
    <div class="svg-wrap">
      <div class="chart-title">{esc(title)}</div>
      <svg viewBox="0 0 780 270" role="img">
        <line x1="50" y1="230" x2="740" y2="230" stroke="#cbd5e1" />
        <line x1="50" y1="35" x2="50" y2="230" stroke="#cbd5e1" />
        <text x="50" y="255">episode {fmt(min_x, 0)}</text>
        <text x="660" y="255">episode {fmt(max_x, 0)}</text>
        <text x="8" y="42">{fmt(max_y, 1)}</text>
        <text x="8" y="230">{fmt(min_y, 1)}</text>
        {''.join(paths)}
      </svg>
      <div class="legend">{''.join(legend)}</div>
    </div>
    """


def decision_table(rows):
    body = []
    for row in rows:
        summary = row["summary"]
        body.append(
            f"""
            <tr>
              <td>{esc(row["label"])}</td>
              <td><span class="pill {esc(row["decision"])}">{esc(row["decision"])}</span></td>
              <td>{fmt(summary.get("mean_score"), 3)}</td>
              <td>{fmt(summary.get("max_score"), 1)}</td>
              <td>{fmt(summary.get("mean_reward"), 3)}</td>
              <td>{fmt(summary.get("total_training_time"), 2)}s</td>
              <td>{esc(row.get("reason", ""))}</td>
            </tr>
            """
        )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Experiment</th><th>Decision</th><th>Mean Score</th><th>Max Score</th>
          <th>Mean Reward</th><th>Train Time</th><th>Reason</th>
        </tr>
      </thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    """


def tool_table(tool_calls):
    rows = []
    for i, call in enumerate(tool_calls, 1):
        rows.append(
            f"""
            <tr>
              <td>{i}</td>
              <td>{call.get("returncode")}</td>
              <td>{fmt(call.get("duration_seconds"), 2)}s</td>
              <td><code>{esc(" ".join(call.get("command", [])))}</code></td>
            </tr>
            """
        )
    return f"""
    <table>
      <thead><tr><th>#</th><th>Return</th><th>Duration</th><th>Command</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def make_html(run_id, state, decisions, tool_calls, rows):
    best = state.get("best") or {}
    best_summary = best.get("summary") or {}
    best_label = best.get("label", "none")
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    accepted = sum(1 for row in rows if row["decision"] == "accept")
    rejected = sum(1 for row in rows if row["decision"] == "reject")
    rollbacks = sum(1 for item in decisions if item.get("decision") == "rollback")

    cards = f"""
    <section class="cards">
      <div class="card"><span>Best</span><strong>{esc(best_label)}</strong></div>
      <div class="card"><span>Mean Score</span><strong>{fmt(best_summary.get("mean_score"), 3)}</strong></div>
      <div class="card"><span>Max Score</span><strong>{fmt(best_summary.get("max_score"), 1)}</strong></div>
      <div class="card"><span>Mean Reward</span><strong>{fmt(best_summary.get("mean_reward"), 2)}</strong></div>
      <div class="card"><span>Accepted</span><strong>{accepted}</strong></div>
      <div class="card"><span>Rejected</span><strong>{rejected}</strong></div>
      <div class="card"><span>Rollbacks</span><strong>{rollbacks}</strong></div>
      <div class="card"><span>Status</span><strong>{esc(state.get("status", "unknown"))}</strong></div>
    </section>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini AutoResearch Dashboard - {esc(run_id)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e2ec;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --blue: #2563eb;
      --green: #16a34a;
      --red: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 34px 20px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{ margin: 0 0 6px; font-size: 26px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; letter-spacing: 0; }}
    .sub {{ color: var(--muted); font-size: 14px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    section.panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 84px;
    }}
    .card span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .card strong {{ font-size: 24px; line-height: 1.1; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 9px 8px; vertical-align: top; }}
    th {{ color: #475467; background: #f8fafc; font-weight: 650; }}
    code {{ white-space: normal; overflow-wrap: anywhere; }}
    .pill {{ display: inline-block; min-width: 74px; text-align: center; padding: 3px 8px; border-radius: 999px; font-size: 12px; }}
    .pill.accept {{ color: #166534; background: #dcfce7; }}
    .pill.reject {{ color: #991b1b; background: #fee2e2; }}
    .pill.rollback {{ color: #92400e; background: #fef3c7; }}
    .pill.baseline {{ color: #1d4ed8; background: #dbeafe; }}
    .bar-chart {{ height: 240px; display: flex; align-items: flex-end; gap: 18px; padding: 18px 8px 4px; border-left: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
    .bar-item {{ width: 110px; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }}
    .bar-value {{ color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    .bar {{ width: 42px; border-radius: 5px 5px 0 0; }}
    .bar-label {{ margin-top: 7px; color: var(--muted); font-size: 12px; text-align: center; overflow-wrap: anywhere; }}
    .svg-wrap svg {{ width: 100%; height: 280px; }}
    .chart-title {{ color: #475467; font-weight: 650; margin-bottom: 8px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 13px; }}
    .legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
    .empty-chart {{ color: var(--muted); padding: 18px; border: 1px dashed var(--line); border-radius: 8px; }}
    @media (max-width: 760px) {{
      main {{ padding: 14px; }}
      header {{ padding: 22px 18px; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .card strong {{ font-size: 20px; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Mini AutoResearch Dashboard</h1>
    <div class="sub">Run <code>{esc(run_id)}</code> · generated {esc(generated_at)}</div>
  </header>
  <main>
    {cards}
    <section class="panel">
      <h2>Experiment Score</h2>
      {bar_chart(rows, "mean_score")}
    </section>
    <section class="panel">
      <h2>Training Progress</h2>
      {progress_svg(rows, "score", "Average score over training checkpoints")}
      {progress_svg(rows, "reward", "Average reward over training checkpoints")}
    </section>
    <section class="panel">
      <h2>Decisions</h2>
      {decision_table(rows)}
    </section>
    <section class="panel">
      <h2>Tool Calls</h2>
      {tool_table(tool_calls)}
    </section>
  </main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Render a local HTML dashboard for an AutoResearch run.")
    parser.add_argument("--run-id", default=None, help="Run id under runs/. Defaults to the latest Agent run.")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults to reports/<run-id>/dashboard.html.")
    parser.add_argument("--open", action="store_true", help="Open the generated dashboard in the default browser.")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id if args.run_id else latest_run_dir()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    run_id = run_dir.name
    state = read_json(run_dir / "state.json")
    decisions = read_jsonl(run_dir / "decisions.jsonl")
    tool_calls = read_jsonl(run_dir / "tool_calls.jsonl")
    rows = collect_experiments(run_dir, decisions)

    output = Path(args.output) if args.output else REPORTS_DIR / run_id / "dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_html(run_id, state, decisions, tool_calls, rows), encoding="utf-8")

    print(json.dumps({
        "dashboard": str(output),
        "run_id": run_id,
        "experiments": len(rows),
        "best": (state.get("best") or {}).get("label"),
    }, indent=2))
    if args.open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
