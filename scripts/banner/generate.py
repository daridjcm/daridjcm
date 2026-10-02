#!/usr/bin/env python3
"""Genera los banners SVG (claro/oscuro) con un retrato pixelado (dithering).

Desde la raíz del repositorio:
    pip install -r scripts/banner/requirements.txt
    python scripts/banner/generate.py

Necesita tu foto en assets/source/portrait.png
"""
from __future__ import annotations

import html
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/portrait.png"
ASSETS = ROOT / "assets"
W, H = 1180, 610
USER = "daridjcm"
BACKGROUND = (168, 132, 198)  # morado liso del avatar de GitHub

YAML_ROWS = [
    (0, "profile", ""),
    (1, "subject", "Dariana Correa"),
    (1, "role", "Desarrolladora de Software"),
    (1, "origin", "Santa Marta, Colombia"),
    (1, "studying", "Ingeniería de Sistemas · Unimagdalena"),
    (1, "focus", "Frontend · UX/UI · Web Apps"),
    (0, "stack", ""),
    (1, "frontend", "HTML · CSS · Tailwind · JavaScript"),
    (1, "frameworks", "React · Astro · TypeScript"),
    (1, "database", "MySQL · MariaDB"),
    (1, "tooling", "Git · Bun · Vite · Linux"),
    (1, "design", "Figma · Wireframes · Prototipos"),
    (0, "projects", ""),
    (1, "stylishoes", "tienda de zapatos con estilo"),
    (1, "lifesim", "simulador de vida (en desarrollo)"),
    (0, "contact", ""),
    (1, "portfolio", "daridjcm.vercel.app"),
    (1, "linkedin", "/in/daridjcm"),
    (1, "github", USER),
]

THEMES = {
    "dark": dict(bg="#0F0F1E", panel="#14142A", panel2="#191933", line="#2D2D55",
                 muted="#8A8AB0", text="#EDEBFF", portrait="#A9AAFF",
                 chrome="#C7A4F5", shadow="#05050F"),
    "light": dict(bg="#F1F0FF", panel="#FFFFFF", panel2="#F6F5FF", line="#CFCFF5",
                  muted="#7B7AA0", text="#1E1B3A", portrait="#5F5FD9",
                  chrome="#7B5EA7", shadow="#A9A9E0"),
}
MONO = "ui-monospace,SFMono-Regular,Consolas,monospace"


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Difusión de error Floyd-Steinberg serpenteante en 1 bit (True = claro)."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    h, w = work.shape
    for y in range(h):
        ltr = y % 2 == 0
        xs = range(w) if ltr else range(w - 1, -1, -1)
        d = 1 if ltr else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + d
            if 0 <= nx < w:
                work[y, nx] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - d < w:
                    work[y + 1, x - d] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < w:
                    work[y + 1, nx] += err * 1 / 16
    return out


def _flood(allowed: np.ndarray, seeds) -> np.ndarray:
    seen = np.zeros(allowed.shape, dtype=bool)
    q = deque(seeds)
    h, w = allowed.shape
    while q:
        y, x = q.popleft()
        if seen[y, x] or not allowed[y, x]:
            continue
        seen[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                q.append((ny, nx))
    return seen


def figure_mask(rgb: np.ndarray) -> np.ndarray:
    """Separa a la persona del fondo del avatar (morado liso + borde blanco + corazón).

    Si cambias el avatar por otro con fondo distinto, ajusta BACKGROUND.
    """
    a = rgb.astype(np.float32)
    h, w = a.shape[:2]
    bg_like = (np.linalg.norm(a - np.array(BACKGROUND, dtype=np.float32), axis=2) < 38) | (a.min(axis=2) > 235)
    seeds = [(0, x) for x in range(w)] + [(h - 1, x) for x in range(w)]
    seeds += [(y, 0) for y in range(h)] + [(y, w - 1) for y in range(h)]
    fg = ~_flood(bg_like, seeds)
    # Nos quedamos con el componente más grande (descarta el corazón y motas sueltas).
    best = np.zeros_like(fg)
    remaining = fg.copy()
    for sy, sx in zip(*np.where(fg)):
        if not remaining[sy, sx]:
            continue
        comp = _flood(remaining, [(sy, sx)])
        remaining &= ~comp
        if comp.sum() > best.sum():
            best = comp
    return best


def portrait_points(theme: str) -> np.ndarray:
    src = Image.open(SOURCE).convert("RGB")
    mask = Image.fromarray((figure_mask(np.asarray(src)) * 255).astype("uint8"))
    # Encuadre: avatar cuadrado -> 320x320 centrado en el marco de 300x340.
    size = 320
    fit = src.resize((size, size), Image.Resampling.LANCZOS)
    fit_mask = np.asarray(mask.resize((size, size), Image.Resampling.LANCZOS)) > 127
    canvas = Image.new("RGB", (300, 340), "white")
    canvas.paste(fit.crop((10, 0, 310, size)), (0, 20))
    cm = np.zeros((340, 300), dtype=bool)
    cm[20:20 + size, :] = fit_mask[:, 10:310]

    g = ImageOps.grayscale(canvas)
    g = ImageOps.autocontrast(g, cutoff=1)
    if theme == "light":
        g = ImageEnhance.Contrast(g).enhance(1.15)
        g = ImageEnhance.Brightness(g).enhance(1.15)
    else:
        # Fondo oscuro: se encienden las zonas claras; se levanta la sombra del pelo.
        g = g.point(lambda v: int(255 * (v / 255) ** 0.6))
        g = ImageEnhance.Contrast(g).enhance(1.2)
    g = g.filter(ImageFilter.UnsharpMask(radius=2.0, percent=150, threshold=1))
    bits = floyd_steinberg(np.asarray(g))
    active = (~bits if theme == "light" else bits) & cm
    ys, xs = np.where(active)
    return np.column_stack((74 + xs, 154 + ys)).astype(int)


def point_path(points: np.ndarray) -> str:
    uniq = sorted({(int(x), int(y)) for x, y in points}, key=lambda p: (p[1], p[0]))
    out, i = [], 0
    while i < len(uniq):
        x0, y = uniq[i]
        x1 = x0
        i += 1
        while i < len(uniq) and uniq[i][1] == y and uniq[i][0] <= x1 + 1:
            x1 = uniq[i][0]
            i += 1
        out.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(out)


def render(theme: str, pts: np.ndarray, rng: np.random.Generator) -> str:
    t = THEMES[theme]
    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Perfil de Dariana Correa</title>',
        '<desc id="desc">Perfil animado en formato terminal con un retrato pixelado y los datos de Dariana Correa, desarrolladora de software.</desc>',
        "<defs>",
        f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".28"/></filter>',
        '<clipPath id="reveal"><rect x="49" y="124" width="390" height="414" rx="3">'
        '<animate attributeName="height" from="0" to="414" dur="2.4s" fill="freeze"/></rect></clipPath>',
        '<clipPath id="frame"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/><circle cx="59" cy="38" r="6" fill="#FEBC2E"/><circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" font-family="{MONO}" font-size="13" letter-spacing=".4">vim profile.yml</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="{MONO}" font-size="13" font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" font-size="11">300×340 / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        # El retrato se revela de arriba hacia abajo. El atributo height final
        # deja el cuadro completo en visores que no soportan animación.
        '<g clip-path="url(#reveal)" shape-rendering="crispEdges">',
    ]
    # Bandas con parpadeo suave para dar vida al retrato
    bands = rng.integers(0, 24, size=len(pts))
    for b in range(24):
        sel = pts[bands == b]
        if not len(sel):
            continue
        dur = 3.0 + (b % 6) * 0.7
        begin = 2.4 + (b % 8) * 0.35
        p.append(
            f'<path d="{point_path(sel)}" fill="none" stroke="{t["portrait"]}" stroke-width="1" opacity=".95">'
            f'<animate attributeName="opacity" begin="{begin:.2f}s" dur="{dur:.1f}s" repeatCount="indefinite" '
            'values=".95;.95;.55;.95;.95" keyTimes="0;.55;.62;.7;1"/></path>'
        )
    p.append("</g>")
    # Línea de escaneo
    p.append(
        '<g clip-path="url(#frame)">'
        f'<rect x="49" y="124" width="390" height="3" fill="{t["chrome"]}" opacity="0">'
        '<animateTransform attributeName="transform" type="translate" values="0 0;0 411" dur="4.5s" begin="2.4s" repeatCount="indefinite"/>'
        '<animate attributeName="opacity" values="0;.5;.5;0" keyTimes="0;.1;.9;1" dur="4.5s" begin="2.4s" repeatCount="indefinite"/>'
        "</rect></g>"
    )
    p.append(
        f'<text x="58" y="551" fill="{t["muted"]}" font-family="{MONO}" font-size="10">PTS {len(pts):05d} · FS/SERPENTINE</text>'
    )
    # Panel derecho
    p += [
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        f'<text x="490" y="111" fill="{t["chrome"]}" font-family="{MONO}" font-size="13" font-weight="700" letter-spacing=".5">profile.yml</text>',
        f'<text x="580" y="111" fill="{t["muted"]}" font-family="{MONO}" font-size="11">[YAML]</text>',
        f'<rect x="996" y="94" width="132" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" stroke="{t["chrome"]}"/>',
        f'<text x="1062" y="111" text-anchor="middle" fill="{t["chrome"]}" font-family="{MONO}" font-size="13" font-weight="700">@{USER}</text>',
    ]
    y = 148.0
    for i, (indent, key, val) in enumerate(YAML_ROWS, 1):
        if indent == 0:
            body = f'<tspan fill="{t["chrome"]}" font-weight="700">{html.escape(key)}:</tspan>'
            x = 525
        else:
            body = (f'<tspan fill="{t["portrait"]}">{html.escape(key)}: </tspan>'
                    f'<tspan fill="{t["text"]}">{html.escape(val)}</tspan>')
            x = 542
        p.append(f'<text x="506" y="{y:.1f}" text-anchor="end" fill="{t["muted"]}" opacity=".45" font-family="{MONO}" font-size="13">{i:2d}</text>')
        p.append(f'<text x="{x}" y="{y:.1f}" font-family="{MONO}" font-size="13">{body}</text>')
        y += 19.5
    p += [
        f'<path d="M474 526H1146" stroke="{t["line"]}"/>',
        f'<rect x="475" y="527" width="670" height="32" fill="{t["panel"]}"/>',
        f'<rect x="485" y="533" width="72" height="20" rx="3" fill="{t["portrait"]}"/>',
        f'<text x="521" y="547" text-anchor="middle" fill="{t["bg"]}" font-family="{MONO}" font-size="11" font-weight="700">NORMAL</text>',
        f'<text x="569" y="547" fill="{t["text"]}" font-family="{MONO}" font-size="12" font-weight="600">profile.yml</text>',
        f'<text x="740" y="547" fill="{t["muted"]}" font-family="{MONO}" font-size="11">[utf-8]</text>',
        f'<text x="1134" y="547" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" font-size="11">{len(YAML_ROWS)}L  100%  1:1</text>',
        "</svg>",
    ]
    return "".join(p)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Falta la foto: {SOURCE}")
    for i, theme in enumerate(THEMES):
        pts = portrait_points(theme)
        rng = np.random.default_rng(314159 + i)
        out = ASSETS / f"banner-{theme}.svg"
        out.write_text(render(theme, pts, rng), encoding="utf-8")
        print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1024:.1f} KiB, {len(pts):,} puntos")


if __name__ == "__main__":
    main()
