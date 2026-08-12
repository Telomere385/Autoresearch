"""Evidence-based validation for final reports authored by the research agent."""

from pathlib import Path


REQUIRED_SECTIONS = (
    "Research Goal",
    "Plan",
    "Baseline",
    "Experiment Process",
    "Failure and Recovery",
    "Best Result",
    "Limitations",
)


def validate_report(path, state):
    """Independently check that an LLM-authored report matches persisted evidence."""
    path = Path(path)
    if not path.is_file():
        return [f"report file is missing: {path}"]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"report is unreadable: {exc}"]
    errors = []
    for section in REQUIRED_SECTIONS:
        if section.lower() not in text.lower():
            errors.append(f"required section is missing: {section}")
    for iteration in range(1, int(state.get("iteration", 0)) + 1):
        if f"iteration {iteration}" not in text.lower():
            errors.append(f"Iteration {iteration} is not documented")
    metric = state["objective"]["metric"]
    best = state.get("best_metrics") or {}
    if metric not in best:
        errors.append(f"best result is missing metric {metric}")
    elif str(best[metric]) not in text:
        errors.append(f"best metric value {best[metric]} is not present")
    if state.get("recovery_events") and "rollback" not in text.lower() and "recovery" not in text.lower():
        errors.append("recovery evidence is not discussed")
    if len(text.strip()) < 200:
        errors.append("report is too short to be complete")
    return errors
