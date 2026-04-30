"""Generate complex synthetic target txt files for 1000-Gaussian fitting."""
from __future__ import annotations

import math
import random
from pathlib import Path


def _header(title: str, n: int, fmt: str) -> str:
    return f"# {title} ({n} Gaussians)\n# Format: {fmt}\n"


def _iso_line(x, y, sigma, alpha, r, g, b) -> str:
    return f"{x:.4f} {y:.4f} {sigma:.4f} {alpha:.4f} {r:.4f} {g:.4f} {b:.4f}"


def _aniso_line(x, y, sx, sy, theta, alpha, r, g, b) -> str:
    return f"{x:.4f} {y:.4f} {sx:.4f} {sy:.4f} {theta:.4f} {alpha:.4f} {r:.4f} {g:.4f} {b:.4f}"


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# ── T6: Dense cityscape with gradient sky ──────────────────────────────
def generate_t6(seed: int = 100) -> tuple[str, int]:
    """Multi-layer city: gradient sky + clouds + buildings + windows + street lights."""
    random.seed(seed)
    lines = []

    # Layer 1: Sky gradient (large, low-alpha Gaussians)
    for i in range(60):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 0.5)
        sigma = random.uniform(0.08, 0.18)
        alpha = random.uniform(0.03, 0.08)
        # Sky: deep blue at top, orange near horizon
        t = y / 0.5  # 0=top, 1=horizon
        r = clamp(0.1 + 0.7 * t + random.gauss(0, 0.03))
        g = clamp(0.05 + 0.3 * t + random.gauss(0, 0.03))
        b = clamp(0.6 - 0.3 * t + random.gauss(0, 0.03))
        lines.append(_iso_line(x, y, sigma, alpha, r, g, b))

    # Layer 2: Clouds (medium, semi-transparent)
    for _ in range(40):
        cx = random.uniform(0.05, 0.95)
        cy = random.uniform(0.08, 0.30)
        for _ in range(3):
            x = cx + random.gauss(0, 0.04)
            y = cy + random.gauss(0, 0.015)
            sigma = random.uniform(0.02, 0.06)
            alpha = random.uniform(0.05, 0.15)
            r = clamp(0.85 + random.gauss(0, 0.05))
            g = clamp(0.80 + random.gauss(0, 0.05))
            b = clamp(0.90 + random.gauss(0, 0.03))
            lines.append(_iso_line(clamp(x), clamp(y), sigma, alpha, r, g, b))

    # Layer 3: Buildings (tall rectangles approximated by stacked Gaussians)
    building_colors = [
        (0.25, 0.25, 0.30), (0.35, 0.30, 0.28), (0.20, 0.22, 0.28),
        (0.40, 0.35, 0.32), (0.18, 0.20, 0.25), (0.30, 0.28, 0.35),
        (0.22, 0.18, 0.22), (0.38, 0.32, 0.30),
    ]
    n_buildings = 14
    for i in range(n_buildings):
        bx = 0.05 + i * (0.9 / n_buildings) + random.uniform(-0.01, 0.01)
        bw = random.uniform(0.025, 0.045)
        bh = random.uniform(0.15, 0.45)
        by_top = 0.55 - bh
        col = random.choice(building_colors)
        # Stack Gaussians vertically
        n_stack = int(bh / 0.02) + 3
        for j in range(n_stack):
            y = by_top + j * (bh / n_stack)
            x = bx + random.gauss(0, 0.003)
            sx = bw
            sy = bh / n_stack * 0.8
            theta = random.gauss(0, 0.02)
            alpha = random.uniform(0.4, 0.7)
            r = clamp(col[0] + random.gauss(0, 0.03))
            g = clamp(col[1] + random.gauss(0, 0.03))
            b = clamp(col[2] + random.gauss(0, 0.03))
            lines.append(_aniso_line(clamp(x), clamp(y), sx, sy, theta, alpha, r, g, b))

    # Layer 4: Windows (tiny bright dots on buildings)
    for i in range(n_buildings):
        bx = 0.05 + i * (0.9 / n_buildings)
        bw = random.uniform(0.025, 0.045)
        bh = random.uniform(0.15, 0.40)
        by_top = 0.55 - bh
        n_win = random.randint(8, 25)
        for _ in range(n_win):
            wx = bx + random.uniform(-bw * 0.7, bw * 0.7)
            wy = random.uniform(by_top + 0.01, 0.54)
            sigma = random.uniform(0.003, 0.007)
            lit = random.random() > 0.3
            if lit:
                alpha = random.uniform(0.5, 0.9)
                r = clamp(0.95 + random.gauss(0, 0.03))
                g = clamp(0.85 + random.gauss(0, 0.05))
                b = clamp(0.4 + random.gauss(0, 0.1))
            else:
                alpha = random.uniform(0.1, 0.3)
                r, g, b = 0.15, 0.15, 0.2
            lines.append(_iso_line(clamp(wx), clamp(wy), sigma, alpha, r, g, b))

    # Layer 5: Ground / street
    for _ in range(40):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.55, 0.70)
        sigma = random.uniform(0.03, 0.08)
        alpha = random.uniform(0.1, 0.3)
        r = clamp(0.15 + random.gauss(0, 0.03))
        g = clamp(0.15 + random.gauss(0, 0.03))
        b = clamp(0.18 + random.gauss(0, 0.03))
        lines.append(_iso_line(x, y, sigma, alpha, r, g, b))

    # Layer 6: Street lights and reflections
    for _ in range(30):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.56, 0.62)
        sigma = random.uniform(0.005, 0.015)
        alpha = random.uniform(0.5, 0.9)
        r = clamp(1.0 + random.gauss(0, 0.02))
        g = clamp(0.9 + random.gauss(0, 0.05))
        b = clamp(0.5 + random.gauss(0, 0.1))
        lines.append(_iso_line(x, y, sigma, alpha, r, g, b))
        # Reflection below
        ry = y + random.uniform(0.02, 0.06)
        rsigma = sigma * random.uniform(1.5, 3.0)
        ralpha = alpha * 0.3
        lines.append(_iso_line(x, ry, rsigma, ralpha, r * 0.7, g * 0.7, b * 0.5))

    # Layer 7: Stars in sky
    for _ in range(50):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 0.25)
        sigma = random.uniform(0.002, 0.005)
        alpha = random.uniform(0.4, 1.0)
        bright = random.uniform(0.7, 1.0)
        lines.append(_iso_line(x, y, sigma, alpha, bright, bright, clamp(bright + 0.1)))

    # Layer 8: Moon
    mx, my = 0.82, 0.10
    for _ in range(15):
        x = mx + random.gauss(0, 0.015)
        y = my + random.gauss(0, 0.015)
        sigma = random.uniform(0.008, 0.025)
        alpha = random.uniform(0.2, 0.6)
        lines.append(_iso_line(x, y, sigma, alpha, 0.95, 0.93, 0.85))
    # Moon glow
    for _ in range(8):
        x = mx + random.gauss(0, 0.03)
        y = my + random.gauss(0, 0.03)
        sigma = random.uniform(0.03, 0.06)
        alpha = random.uniform(0.03, 0.08)
        lines.append(_iso_line(x, y, sigma, alpha, 0.7, 0.7, 0.8))

    n = len(lines)
    header = _header("T6: Night cityscape with gradient sky, clouds, buildings, windows, street lights, moon, stars", n, "x y sigma alpha r g b  OR  x y sx sy theta alpha r g b")
    return header + "\n".join(lines) + "\n", n


# ── T7: Mandala / radial pattern ───────────────────────────────────────
def generate_t7(seed: int = 200) -> tuple[str, int]:
    """Concentric rings + radial spokes + petal clusters at multiple scales."""
    random.seed(seed)
    lines = []
    cx, cy = 0.5, 0.5

    # Background glow
    for _ in range(30):
        x = cx + random.gauss(0, 0.15)
        y = cy + random.gauss(0, 0.15)
        sigma = random.uniform(0.08, 0.20)
        alpha = random.uniform(0.02, 0.06)
        r = clamp(0.15 + random.gauss(0, 0.05))
        g = clamp(0.05 + random.gauss(0, 0.03))
        b = clamp(0.25 + random.gauss(0, 0.05))
        lines.append(_iso_line(clamp(x), clamp(y), sigma, alpha, r, g, b))

    # Concentric rings (5 rings, each made of many small Gaussians along circumference)
    ring_colors = [
        (0.9, 0.2, 0.2), (0.9, 0.6, 0.1), (0.2, 0.9, 0.3),
        (0.2, 0.5, 0.95), (0.8, 0.2, 0.9),
    ]
    for ring_i, radius in enumerate([0.08, 0.15, 0.22, 0.30, 0.40]):
        col = ring_colors[ring_i]
        n_pts = int(radius * 250) + 20
        for j in range(n_pts):
            angle = 2 * math.pi * j / n_pts + random.gauss(0, 0.02)
            x = cx + radius * math.cos(angle) + random.gauss(0, 0.003)
            y = cy + radius * math.sin(angle) + random.gauss(0, 0.003)
            sx = random.uniform(0.006, 0.015)
            sy = random.uniform(0.002, 0.005)
            theta = angle + math.pi / 2
            alpha = random.uniform(0.3, 0.7)
            r = clamp(col[0] + random.gauss(0, 0.05))
            g = clamp(col[1] + random.gauss(0, 0.05))
            b = clamp(col[2] + random.gauss(0, 0.05))
            lines.append(_aniso_line(clamp(x), clamp(y), sx, sy, theta, alpha, r, g, b))

    # Radial spokes (12 spokes)
    for spoke_i in range(12):
        angle = 2 * math.pi * spoke_i / 12
        n_seg = 18
        for j in range(n_seg):
            t = 0.05 + j * (0.38 / n_seg)
            x = cx + t * math.cos(angle)
            y = cy + t * math.sin(angle)
            sx = random.uniform(0.012, 0.022)
            sy = random.uniform(0.003, 0.006)
            theta = angle
            alpha = random.uniform(0.2, 0.5)
            # Rainbow along spoke
            hue = (spoke_i / 12 + t) % 1.0
            r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
            lines.append(_aniso_line(clamp(x), clamp(y), sx, sy, theta, alpha, r, g, b))

    # Petal clusters at ring intersections with spokes
    for spoke_i in range(12):
        angle = 2 * math.pi * spoke_i / 12
        for radius in [0.15, 0.30]:
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)
            n_petals = 5
            for p in range(n_petals):
                pa = angle + 2 * math.pi * p / n_petals
                for k in range(3):
                    dist = 0.01 + k * 0.008
                    x = px + dist * math.cos(pa) + random.gauss(0, 0.002)
                    y = py + dist * math.sin(pa) + random.gauss(0, 0.002)
                    sigma = random.uniform(0.004, 0.008)
                    alpha = random.uniform(0.3, 0.7)
                    hue = (spoke_i / 12 + p / n_petals * 0.3) % 1.0
                    r, g, b = _hsv_to_rgb(hue, 0.7, 1.0)
                    lines.append(_iso_line(clamp(x), clamp(y), sigma, alpha, r, g, b))

    # Center ornament
    for _ in range(20):
        x = cx + random.gauss(0, 0.015)
        y = cy + random.gauss(0, 0.015)
        sigma = random.uniform(0.005, 0.015)
        alpha = random.uniform(0.4, 0.9)
        lines.append(_iso_line(x, y, sigma, alpha, 1.0, 0.95, 0.7))

    # Corner decorations (4 corners)
    for corner_x, corner_y in [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]:
        for _ in range(12):
            x = corner_x + random.gauss(0, 0.04)
            y = corner_y + random.gauss(0, 0.04)
            sx = random.uniform(0.01, 0.03)
            sy = random.uniform(0.003, 0.008)
            theta = math.atan2(cy - y, cx - x) + random.gauss(0, 0.2)
            alpha = random.uniform(0.2, 0.5)
            r = clamp(0.7 + random.gauss(0, 0.1))
            g = clamp(0.5 + random.gauss(0, 0.1))
            b = clamp(0.9 + random.gauss(0, 0.05))
            lines.append(_aniso_line(clamp(x), clamp(y), sx, sy, theta, alpha, r, g, b))

    n = len(lines)
    header = _header("T7: Mandala with concentric rings, radial spokes, petals, and corner ornaments", n, "x y sigma alpha r g b  OR  x y sx sy theta alpha r g b")
    return header + "\n".join(lines) + "\n", n


# ── T8: Underwater coral reef ──────────────────────────────────────────
def generate_t8(seed: int = 300) -> tuple[str, int]:
    """Multi-layer underwater scene: water gradient + caustics + corals + fish + bubbles."""
    random.seed(seed)
    lines = []

    # Layer 1: Water gradient (deep blue at bottom, lighter at top)
    for _ in range(50):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 1.0)
        sigma = random.uniform(0.10, 0.20)
        alpha = random.uniform(0.04, 0.10)
        depth = y  # 0=surface, 1=deep
        r = clamp(0.05 + 0.15 * (1 - depth) + random.gauss(0, 0.02))
        g = clamp(0.20 + 0.25 * (1 - depth) + random.gauss(0, 0.02))
        b = clamp(0.40 + 0.30 * (1 - depth) + random.gauss(0, 0.02))
        lines.append(_iso_line(x, y, sigma, alpha, r, g, b))

    # Layer 2: Light caustics (wavy bright patches near top)
    for _ in range(40):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 0.35)
        sx = random.uniform(0.02, 0.06)
        sy = random.uniform(0.005, 0.015)
        theta = random.uniform(-0.5, 0.5)
        alpha = random.uniform(0.05, 0.15)
        lines.append(_aniso_line(x, y, sx, sy, theta, alpha, 0.6, 0.8, 1.0))

    # Layer 3: Sandy bottom
    for _ in range(50):
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.75, 1.0)
        sigma = random.uniform(0.02, 0.06)
        alpha = random.uniform(0.1, 0.3)
        r = clamp(0.65 + random.gauss(0, 0.05))
        g = clamp(0.55 + random.gauss(0, 0.05))
        b = clamp(0.35 + random.gauss(0, 0.05))
        lines.append(_iso_line(x, y, sigma, alpha, r, g, b))

    # Layer 4: Coral structures (branching clusters)
    coral_configs = [
        (0.15, 0.80, (0.95, 0.3, 0.3)),  # red coral left
        (0.35, 0.75, (0.95, 0.5, 0.1)),  # orange coral
        (0.55, 0.82, (0.9, 0.2, 0.6)),   # pink coral
        (0.75, 0.78, (0.3, 0.8, 0.4)),   # green coral
        (0.90, 0.80, (0.8, 0.8, 0.2)),   # yellow coral
        (0.25, 0.85, (0.6, 0.2, 0.8)),   # purple coral
        (0.65, 0.76, (0.2, 0.7, 0.7)),   # teal coral
    ]
    for base_x, base_y, col in coral_configs:
        # Each coral is a branching structure
        n_branches = random.randint(4, 8)
        for br in range(n_branches):
            angle = -math.pi / 2 + random.uniform(-0.8, 0.8)  # mostly upward
            length = random.uniform(0.06, 0.16)
            n_seg = random.randint(8, 16)
            bx, by = base_x, base_y
            for seg in range(n_seg):
                t = seg / n_seg
                bx += (length / n_seg) * math.cos(angle) + random.gauss(0, 0.003)
                by += (length / n_seg) * math.sin(angle) + random.gauss(0, 0.003)
                angle += random.gauss(0, 0.15)  # wiggle
                sx = random.uniform(0.005, 0.012) * (1 - 0.4 * t)
                sy = random.uniform(0.003, 0.007) * (1 - 0.4 * t)
                theta = angle
                alpha = random.uniform(0.3, 0.7)
                r = clamp(col[0] + random.gauss(0, 0.06))
                g = clamp(col[1] + random.gauss(0, 0.06))
                b = clamp(col[2] + random.gauss(0, 0.06))
                lines.append(_aniso_line(clamp(bx), clamp(by), sx, sy, theta, alpha, r, g, b))
            # Tip bloom
            for _ in range(3):
                tx = bx + random.gauss(0, 0.008)
                ty = by + random.gauss(0, 0.008)
                sigma = random.uniform(0.004, 0.010)
                alpha = random.uniform(0.4, 0.8)
                r = clamp(col[0] + 0.15 + random.gauss(0, 0.05))
                g = clamp(col[1] + 0.15 + random.gauss(0, 0.05))
                b = clamp(col[2] + 0.15 + random.gauss(0, 0.05))
                lines.append(_iso_line(clamp(tx), clamp(ty), sigma, alpha, r, g, b))

    # Layer 5: Sea anemones (radial tentacles)
    for anem_x, anem_y in [(0.45, 0.88), (0.80, 0.90)]:
        n_tent = 16
        for t in range(n_tent):
            angle = 2 * math.pi * t / n_tent - math.pi / 2
            for seg in range(6):
                dist = 0.01 + seg * 0.008
                x = anem_x + dist * math.cos(angle) + random.gauss(0, 0.002)
                y = anem_y + dist * math.sin(angle) + random.gauss(0, 0.002)
                sx = random.uniform(0.006, 0.012)
                sy = random.uniform(0.002, 0.004)
                theta = angle
                alpha = random.uniform(0.3, 0.6)
                hue = (t / n_tent) % 1.0
                r, g, b = _hsv_to_rgb(hue, 0.6, 0.9)
                lines.append(_aniso_line(clamp(x), clamp(y), sx, sy, theta, alpha, r, g, b))
        # Center
        for _ in range(5):
            sigma = random.uniform(0.005, 0.012)
            alpha = random.uniform(0.5, 0.9)
            lines.append(_iso_line(anem_x + random.gauss(0, 0.005), anem_y + random.gauss(0, 0.005),
                                   sigma, alpha, 0.95, 0.85, 0.3))

    # Layer 6: Fish (small clusters of colored Gaussians)
    fish_configs = [
        (0.30, 0.35, 0.015, (1.0, 0.6, 0.1)),
        (0.60, 0.25, 0.012, (0.2, 0.5, 1.0)),
        (0.80, 0.40, 0.018, (1.0, 0.2, 0.2)),
        (0.15, 0.50, 0.010, (0.9, 0.9, 0.2)),
        (0.50, 0.45, 0.014, (0.3, 0.9, 0.5)),
        (0.70, 0.55, 0.011, (0.9, 0.4, 0.8)),
    ]
    for fx, fy, fsize, fcol in fish_configs:
        # Body (elongated)
        sx = fsize * 1.5
        sy = fsize * 0.5
        heading = random.uniform(-0.3, 0.3)
        alpha = random.uniform(0.6, 0.9)
        lines.append(_aniso_line(fx, fy, sx, sy, heading, alpha, *fcol))
        # Tail
        tx = fx - fsize * 1.8 * math.cos(heading)
        ty = fy - fsize * 1.8 * math.sin(heading)
        lines.append(_aniso_line(clamp(tx), clamp(ty), fsize * 0.8, fsize * 0.3,
                                 heading + 0.3, alpha * 0.7, fcol[0] * 0.8, fcol[1] * 0.8, fcol[2] * 0.8))
        # Eye
        ex = fx + fsize * 1.0 * math.cos(heading)
        ey = fy + fsize * 1.0 * math.sin(heading) - fsize * 0.2
        lines.append(_iso_line(clamp(ex), clamp(ey), 0.003, 0.9, 0.1, 0.1, 0.1))
        # Small school mates
        for _ in range(4):
            mx = fx + random.gauss(0, 0.04)
            my = fy + random.gauss(0, 0.03)
            ms = fsize * random.uniform(0.5, 0.8)
            lines.append(_aniso_line(clamp(mx), clamp(my), ms * 1.2, ms * 0.4,
                                     heading + random.gauss(0, 0.2),
                                     alpha * random.uniform(0.6, 0.9),
                                     clamp(fcol[0] + random.gauss(0, 0.05)),
                                     clamp(fcol[1] + random.gauss(0, 0.05)),
                                     clamp(fcol[2] + random.gauss(0, 0.05))))

    # Layer 7: Bubbles
    for _ in range(60):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.85)
        sigma = random.uniform(0.004, 0.012)
        alpha = random.uniform(0.15, 0.45)
        lines.append(_iso_line(x, y, sigma, alpha, 0.7, 0.85, 1.0))
        # Highlight on bubble
        lines.append(_iso_line(x - sigma * 0.3, y - sigma * 0.3, sigma * 0.3,
                               alpha * 0.8, 0.95, 0.98, 1.0))

    # Layer 8: Kelp / seaweed strands
    for strand_x in [0.10, 0.42, 0.70, 0.92]:
        n_seg = random.randint(15, 25)
        x, y = strand_x, 0.95
        sway = random.uniform(-0.3, 0.3)
        for seg in range(n_seg):
            y -= random.uniform(0.015, 0.03)
            x += math.sin(y * 10 + sway) * 0.005 + random.gauss(0, 0.003)
            sx = random.uniform(0.003, 0.008)
            sy = random.uniform(0.008, 0.018)
            theta = math.sin(y * 8 + sway) * 0.3
            alpha = random.uniform(0.3, 0.6)
            r = clamp(0.1 + random.gauss(0, 0.03))
            g = clamp(0.5 + random.gauss(0, 0.06))
            b = clamp(0.15 + random.gauss(0, 0.03))
            lines.append(_aniso_line(clamp(x), clamp(y), sx, sy, theta, alpha, r, g, b))

    n = len(lines)
    header = _header("T8: Underwater coral reef with fish, anemones, bubbles, and kelp", n, "x y sigma alpha r g b  OR  x y sx sy theta alpha r g b")
    return header + "\n".join(lines) + "\n", n


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Simple HSV to RGB."""
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    h6 = int(h * 6) % 6
    if h6 == 0:
        r, g, b = c, x, 0
    elif h6 == 1:
        r, g, b = x, c, 0
    elif h6 == 2:
        r, g, b = 0, c, x
    elif h6 == 3:
        r, g, b = 0, x, c
    elif h6 == 4:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return clamp(r + m), clamp(g + m), clamp(b + m)


def main():
    out_dir = Path(__file__).resolve().parent / "data" / "txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("t6_night_cityscape.txt", generate_t6),
        ("t7_mandala.txt", generate_t7),
        ("t8_coral_reef.txt", generate_t8),
    ]

    for fname, gen_fn in generators:
        content, n = gen_fn()
        path = out_dir / fname
        path.write_text(content, encoding="utf-8")
        print(f"Generated {fname}: {n} Gaussians")


if __name__ == "__main__":
    main()
