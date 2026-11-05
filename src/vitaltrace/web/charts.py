"""Gráficos SVG gerados no servidor (sem JavaScript), acessíveis e leves."""

from datetime import datetime
from html import escape

import pandas as pd
from markupsafe import Markup

TEAL, CRIT, AMBER, INK, MUTED, GRID = "#0F766E", "#B42318", "#C98A0B", "#0E2A2B", "#3F5A5B", "#E1EAEA"


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
    width: int = 640,
    height: int = 230,
) -> Markup:
    if df.empty:
        return Markup('<p class="muted">Sem dados de sinais vitais.</p>')
    left, right, top, bottom = 44, 12, 14, 30
    pw, ph = width - left - right, height - top - bottom
    t0, t1 = df["ts"].min(), df["ts"].max()
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
    lx, ly = x(df["ts"].iloc[-1]), y(float(df[column].iloc[-1]))
    parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4.5" fill="{TEAL}" stroke="#fff" stroke-width="2"/>')
    parts.append("</svg>")
    return Markup("".join(parts))
