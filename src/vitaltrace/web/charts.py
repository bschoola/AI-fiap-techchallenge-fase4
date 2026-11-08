"""Gráficos SVG gerados no servidor (sem JavaScript), acessíveis e leves."""

from datetime import datetime
from html import escape

import pandas as pd
from markupsafe import Markup

TEAL, CRIT, AMBER, INK, MUTED, GRID = "#0F766E", "#B42318", "#C98A0B", "#0E2A2B", "#3F5A5B", "#E1EAEA"
EVENT_COLORS = {"critical": CRIT, "high": "#C2570C", "medium": AMBER, "low": "#5B7C7D", "stable": "#5B7C7D"}


def line_chart(
    df: pd.DataFrame,
    column: str,
    *,
    title: str,
    unit: str,
    y_min: float,
    y_max: float,
    thresholds: list[tuple[float, str]] | None = None,
    markers: list[tuple[datetime, str]] | None = None,
    events: list[dict] | None = None,
    width: int = 640,
    height: int = 230,
) -> Markup:
    if df.empty:
        return Markup('<p class="muted">Sem dados de sinais vitais.</p>')
    left, right, top, bottom = 44, 12, 14, 30
    pw, ph = width - left - right, height - top - bottom
    t0, t1 = df["ts"].min(), df["ts"].max()
    last_reading = t1
    # análises feitas depois da última aferição ampliam o eixo, para continuarem visíveis
    inside = [ev for ev in events or [] if pd.Timestamp(ev["at"]) >= t0]
    if inside:
        t1 = max(t1, max(pd.Timestamp(ev["at"]) for ev in inside) + pd.Timedelta(hours=3))
    span = max((t1 - t0).total_seconds(), 1.0)

    def x(ts) -> float:
        return left + pw * (pd.Timestamp(ts) - t0).total_seconds() / span

    def y(value: float) -> float:
        value = min(max(value, y_min), y_max)
        return top + ph * (1 - (value - y_min) / (y_max - y_min))

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}: de {df[column].iloc[0]:.0f} a {df[column].iloc[-1]:.0f} {escape(unit)} '
        f'em {(t1 - t0).days} dias">'
    ]
    ticks = 4
    for i in range(ticks + 1):
        value = y_min + (y_max - y_min) * i / ticks
        yy = y(value)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{left - 6}" y="{yy + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{value:.0f}</text>')
    # eixo x: um rótulo a cada ~3 dias
    days = max(1, (t1 - t0).days)
    for d in range(0, days + 1, 3):
        ts = t0 + pd.Timedelta(days=d)
        parts.append(f'<text x="{x(ts):.1f}" y="{height - 8}" font-size="11" fill="{MUTED}" text-anchor="middle">{ts.strftime("%d/%m")}</text>')
    for value, label in thresholds or []:
        yy = y(value)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{CRIT}" stroke-width="1.5" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{width - right - 4}" y="{yy - 4:.1f}" font-size="11" fill="{CRIT}" text-anchor="end" font-weight="700">{escape(label)}</text>')
    for ts, label in markers or []:
        if not (t0 <= pd.Timestamp(ts) <= t1):
            continue
        xx = x(ts)
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{top + ph}" stroke="{AMBER}" stroke-width="1.5" stroke-dasharray="3 3"/>')
        parts.append(f'<text x="{xx - 4:.1f}" y="{top + 12}" font-size="11" fill="{AMBER}" text-anchor="end" font-weight="700">{escape(label)}</text>')
    points = " ".join(f"{x(ts):.1f},{y(v):.1f}" for ts, v in zip(df["ts"], df[column]))
    parts.append(f'<polyline points="{points}" fill="none" stroke="{TEAL}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
    lx, ly = x(last_reading), y(float(df[column].iloc[-1]))
    parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4.5" fill="{TEAL}" stroke="#fff" stroke-width="2"/>')
    for ev in sorted(inside, key=lambda e: e["at"]):
        parts.append(_event_marker(ev, x(ev["at"]), top + ph - 11, top, top + ph))
    parts.append("</svg>")
    return Markup("".join(parts))


def _event_marker(ev: dict, xx: float, yy: float, y_top: float, y_bottom: float) -> str:
    """Círculo = áudio, losango = vídeo; a cor é a gravidade. Clicar leva à entrada da linha do tempo."""
    color = EVENT_COLORS.get(ev["level"], EVENT_COLORS["stable"])
    label = escape(ev["label"])
    if ev["kind"] == "video":
        shape = f'<path d="M{xx:.1f} {yy - 7:.1f} L{xx + 7:.1f} {yy:.1f} L{xx:.1f} {yy + 7:.1f} L{xx - 7:.1f} {yy:.1f} Z"'
    else:
        shape = f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6.5"'
    return (
        f'<a href="{escape(ev["href"])}" aria-label="{label}"><title>{label}</title>'
        f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y_top}" y2="{y_bottom}" stroke="{color}" stroke-width="1" stroke-dasharray="2 3" opacity=".55"/>'
        f'{shape} fill="{color}" stroke="#fff" stroke-width="2"/></a>'
    )


def angle_curve(curve: dict | None, width: int = 640, height: int = 210) -> Markup:
    """Curva do ângulo da articulação mais ativa ao longo do vídeo, com as repetições marcadas."""
    pts = (curve or {}).get("points") or []
    if len(pts) < 3:
        return Markup("")
    left, right, top, bottom = 44, 12, 14, 30
    pw, ph = width - left - right, height - top - bottom
    t0, t1 = pts[0][0], max(pts[-1][0], pts[0][0] + 1)
    lo = max(0, min(p[1] for p in pts) - 10)
    hi = min(190, max(p[1] for p in pts) + 10)

    def x(t: float) -> float:
        return left + pw * (t - t0) / (t1 - t0)

    def y(a: float) -> float:
        return top + ph * (1 - (a - lo) / max(hi - lo, 1))

    joint = escape(str(curve.get("joint", "articulação")))
    reps = curve.get("reps") or []
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" aria-label="Ângulo do {joint} ao longo do vídeo, '
        f'{len(reps)} repetição(ões) marcadas">',
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="{GRID}"/>',
    ]
    for a in (lo, (lo + hi) / 2, hi):
        parts.append(f'<text x="{left - 6}" y="{y(a) + 4:.0f}" text-anchor="end" font-size="11" fill="{MUTED}">{a:.0f}°</text>')
    for t in (t0, (t0 + t1) / 2, t1):
        parts.append(f'<text x="{x(t):.0f}" y="{height - 10}" text-anchor="middle" font-size="11" fill="{MUTED}">{t:.0f} s</text>')
    line = " ".join(f"{x(t):.1f},{y(a):.1f}" for t, a in pts)
    parts.append(f'<polyline points="{line}" fill="none" stroke="{TEAL}" stroke-width="2" stroke-linejoin="round"/>')
    values = {round(p[0], 1): p[1] for p in pts}
    for t in reps:
        a = values.get(round(t, 1))
        if a is not None:
            parts.append(f'<circle cx="{x(t):.1f}" cy="{y(a):.1f}" r="4" fill="{AMBER}" stroke="#fff" stroke-width="1.5"/>')
    parts.append("</svg>")
    return Markup("".join(parts))
