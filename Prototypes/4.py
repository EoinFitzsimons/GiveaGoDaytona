# complete_patched_daytona_circuits.py
# Full patched file integrating segment-based generator, lap simulator, improved scoring,
# Daytona sandbox execution, and UI. Designed to run locally with Daytona credits.
# Replace your existing 1.py with this file. Keep your .env with DAYTONA_API_KEY and GROQ_API_KEY.

import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from daytona import Daytona, DaytonaConfig
from PyQt6.QtCore import QPointF, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QFont, QPainter, QPen, QWheelEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
TOTAL_CIRCUITS = 10
MAX_ITERATIONS_PER_CIRCUIT = 10
ACCEPTANCE_SCORE = 82.0
CANVAS_WIDTH = 1100
CANVAS_HEIGHT = 800
TRACK_WIDTH = 18.0
OUTPUT_DIRECTORY = Path("generated_circuits")
BEST_DIRECTORY = OUTPUT_DIRECTORY / "best"
CIRCUIT_FILE_NAME = "circuit.json"
CIRCUIT_SVG_FILE_NAME = "circuit.svg"
LLM_MODEL = "mixtral-8x7b-32768"
DAYTONA_TIMEOUT_SECONDS = 30
LLM_MAX_OUTPUT_TOKENS = 700
POINTS_PER_SEGMENT = 12
MIN_SELF_DISTANCE = 35.0
MAX_CURVATURE = 0.18
SVG_PADDING_RATIO = 0.10
EPS = 1e-9

# ============================================================================
# ENVIRONMENT
# ============================================================================
load_dotenv()
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not DAYTONA_API_KEY:
    raise RuntimeError("DAYTONA_API_KEY is missing from .env")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

# ============================================================================
# CLIENTS
# ============================================================================
daytona = Daytona(DaytonaConfig(api_key=DAYTONA_API_KEY))
groq = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class CircuitParameters:
    width: float
    height: float
    corner_count: int
    straight_count: int
    radius_min: float
    radius_max: float
    amplitude: float
    asymmetry: float
    chicane_probability: float
    hairpin_probability: float
    variation: float
    spacing_limit: float
    complexity: float
    direction: int


@dataclass
class CircuitScore:
    total: float
    closure: float
    separation: float
    smoothness: float
    complexity: float
    scale: float
    variety: float
    validity: float
    reason: str


@dataclass
class CircuitResult:
    circuit_id: int
    iteration: int
    parameters: dict[str, Any]
    points: list[list[float]]
    score: dict[str, Any]
    accepted: bool


# ============================================================================
# UTILITIES
# ============================================================================
def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_difference(a: float, b: float) -> float:
    difference = abs(a - b)
    while difference > math.pi:
        difference = abs(difference - 2.0 * math.pi)
    return difference


def is_zero(value: float) -> bool:
    return abs(value) < EPS


# ============================================================================
# LLM WRAPPER
# ============================================================================
def llm(messages: list[dict[str, str]]) -> str:
    # type: ignore[arg-type] to satisfy Pylance while runtime accepts dict messages
    response = groq.chat.completions.create(  # type: ignore[arg-type]
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty content")
    return content


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM response did not contain JSON")
    return json.loads(cleaned[start:end + 1])


def parameter_schema() -> str:
    return """
{
  "width": number 900..1800,
  "height": number 500..1100,
  "corner_count": integer 5..14,
  "straight_count": integer 2..8,
  "radius_min": number 45..130,
  "radius_max": number 100..280,
  "amplitude": number 0.10..0.45,
  "asymmetry": number 0.00..0.45,
  "chicane_probability": number 0..1,
  "hairpin_probability": number 0..1,
  "variation": number 0.05..0.35,
  "spacing_limit": number 35..100,
  "complexity": number 0.20..0.95,
  "direction": integer -1 or 1
}
""".strip()


def validate_parameters(data: dict[str, Any]) -> CircuitParameters:
    defaults = {
        "width": 1300.0,
        "height": 800.0,
        "corner_count": 8,
        "straight_count": 4,
        "radius_min": 70.0,
        "radius_max": 190.0,
        "amplitude": 0.25,
        "asymmetry": 0.15,
        "chicane_probability": 0.20,
        "hairpin_probability": 0.15,
        "variation": 0.15,
        "spacing_limit": 55.0,
        "complexity": 0.55,
        "direction": 1,
    }
    values = {key: data.get(key, value) for key, value in defaults.items()}
    return CircuitParameters(
        width=clamp(float(values["width"]), 900.0, 1800.0),
        height=clamp(float(values["height"]), 500.0, 1100.0),
        corner_count=int(clamp(int(values["corner_count"]), 5, 14)),
        straight_count=int(clamp(int(values["straight_count"]), 2, 8)),
        radius_min=clamp(float(values["radius_min"]), 45.0, 130.0),
        radius_max=clamp(float(values["radius_max"]), 100.0, 280.0),
        amplitude=clamp(float(values["amplitude"]), 0.10, 0.45),
        asymmetry=clamp(float(values["asymmetry"]), 0.0, 0.45),
        chicane_probability=clamp(float(values["chicane_probability"]), 0.0, 1.0),
        hairpin_probability=clamp(float(values["hairpin_probability"]), 0.0, 1.0),
        variation=clamp(float(values["variation"]), 0.05, 0.35),
        spacing_limit=clamp(float(values["spacing_limit"]), 35.0, 100.0),
        complexity=clamp(float(values["complexity"]), 0.20, 0.95),
        direction=1 if int(values["direction"]) >= 0 else -1,
    )


def generate_initial_parameters(circuit_id: int) -> CircuitParameters:
    prompt = f"""
Generate parameters for procedural closed racing circuit {circuit_id}.
The Python geometry engine creates the physical 2D centreline.
Your job is ONLY to choose design parameters.
Requirements:
- closed non-self-intersecting circuit
- physically plausible
- varied from previous circuits
- 5..14 corners
- 2..8 straights
- no overlapping track
- avoid excessive curvature
- different designs should have different shapes
Prefer layouts with a mix of long straights and decisive corners. Produce parameters that
encourage clustering of sharp turns (hairpins) and long straights (overtaking opportunities).
Do not produce uniform sinusoidal curvature. Return JSON only.
Schema:
{parameter_schema()}
""".strip()
    messages = [
        {"role": "system", "content": "You are a procedural motorsport circuit designer. Return compact valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = llm(messages)
        return validate_parameters(extract_json(response))
    except Exception:
        return fallback_parameters(circuit_id)


def improve_parameters(
    circuit_id: int,
    iteration: int,
    parameters: CircuitParameters,
    score: CircuitScore,
) -> CircuitParameters:
    prompt = f"""
Circuit {circuit_id}, iteration {iteration}.
Current parameters:
{json.dumps(asdict(parameters), separators=(",", ":"))}
Score:
{json.dumps(asdict(score), separators=(",", ":"))}
Improve the design.
Problems should be corrected using parameter changes.
Do not create geometry.
Keep values within the schema.
Return JSON only.
Schema:
{parameter_schema()}
""".strip()
    messages = [
        {"role": "system", "content": "You optimise procedural racing circuit parameters. Return JSON only."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = llm(messages)
        return validate_parameters(extract_json(response))
    except Exception:
        return mutate_parameters(parameters)


def fallback_parameters(circuit_id: int) -> CircuitParameters:
    rng = random.Random(100000 + circuit_id)  # noqa: S2245
    return CircuitParameters(
        width=rng.uniform(1100.0, 1600.0),
        height=rng.uniform(600.0, 950.0),
        corner_count=rng.randint(6, 11),
        straight_count=rng.randint(3, 6),
        radius_min=rng.uniform(55.0, 100.0),
        radius_max=rng.uniform(140.0, 230.0),
        amplitude=rng.uniform(0.15, 0.35),
        asymmetry=rng.uniform(0.05, 0.30),
        chicane_probability=rng.uniform(0.05, 0.35),
        hairpin_probability=rng.uniform(0.05, 0.30),
        variation=rng.uniform(0.08, 0.25),
        spacing_limit=rng.uniform(45.0, 75.0),
        complexity=rng.uniform(0.35, 0.80),
        direction=rng.choice([-1, 1]),
    )


def mutate_parameters(parameters: CircuitParameters) -> CircuitParameters:
    result = CircuitParameters(**asdict(parameters))
    result.amplitude = clamp(result.amplitude + random.uniform(-0.06, 0.06), 0.10, 0.45)
    result.asymmetry = clamp(result.asymmetry + random.uniform(-0.06, 0.06), 0.0, 0.45)
    result.radius_min = clamp(result.radius_min + random.uniform(-15.0, 15.0), 45.0, 130.0)
    result.radius_max = clamp(result.radius_max + random.uniform(-25.0, 25.0), 100.0, 280.0)
    result.spacing_limit = clamp(result.spacing_limit + random.uniform(-8.0, 8.0), 35.0, 100.0)
    return result


# ============================================================================
# GEOMETRY: segment-based generator + fillets
# ============================================================================
def generate_segment_layout(parameters: CircuitParameters, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)  # noqa: S2245
    n_corners = parameters.corner_count
    n_straights = max(1, parameters.straight_count)
    n_vertices = n_corners + n_straights
    angles = []
    cluster_prob = clamp(parameters.complexity, 0.2, 0.9)
    prev_sign = 1 if rng.random() < 0.5 else -1
    for i in range(n_vertices):
        if rng.random() < (parameters.variation * 0.5):
            angle = rng.uniform(-0.05, 0.05)
        else:
            mag = rng.uniform(0.2, 1.8) * (math.pi / 6.0)
            if rng.random() < cluster_prob:
                sign = prev_sign
            else:
                sign = -prev_sign if rng.random() < 0.3 else prev_sign
            angle = sign * mag
            prev_sign = sign
        angles.append(angle)

    radii = []
    rmin = parameters.radius_min
    rmax = parameters.radius_max
    current = rng.uniform(rmin, rmax)
    for i in range(n_vertices):
        step = rng.uniform(-0.25, 0.25) * (rmax - rmin) * parameters.variation
        if rng.random() < 0.12:
            current = clamp(current + rng.uniform(-0.6, 0.6) * (rmax - rmin), rmin, rmax)
        else:
            current = clamp(current + step, rmin, rmax)
        radii.append(current)

    straights = []
    for i in range(n_vertices):
        if abs(angles[i]) < 0.08:
            L = rng.uniform(120.0, 400.0) * (1.0 + parameters.complexity * 0.5)
        else:
            L = rng.uniform(30.0, 160.0) * (1.0 + (1.0 - parameters.complexity) * 0.5)
        straights.append(L)

    poly_pts = []
    x, y = 0.0, 0.0
    heading = 0.0
    for i in range(n_vertices):
        L = straights[i]
        x2 = x + L * math.cos(heading)
        y2 = y + L * math.sin(heading)
        poly_pts.append((x, y))
        heading += angles[i]
        x, y = x2, y2

    if len(poly_pts) < 4:
        return fallback_generate_points(parameters, seed)

    centreline = build_fillet_centreline(poly_pts, angles, radii, straights, parameters)
    return centreline


def fallback_generate_points(parameters: CircuitParameters, seed: int) -> list[tuple[float, float]]:
    # Minimal fallback: small ellipse-like ring to avoid crashes
    rng = random.Random(seed)  # noqa: S2245
    pts = []
    count = max(64, parameters.corner_count * POINTS_PER_SEGMENT)
    rx = parameters.width / 2.0
    ry = parameters.height / 2.0
    for i in range(count):
        theta = 2.0 * math.pi * i / count
        r = 1.0 + math.sin(theta * parameters.corner_count) * parameters.amplitude
        x = rx * r * math.cos(theta)
        y = ry * r * math.sin(theta)
        pts.append((x, y))
    return smooth_closed_curve(pts)


def build_fillet_centreline(poly_pts, angles, radii, straights, parameters):
    pts = []
    count = len(poly_pts)
    for i in range(count):
        p_prev = poly_pts[i - 1]
        p_curr = poly_pts[i]
        p_next = poly_pts[(i + 1) % count]

        v_in = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v_out = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        def norm(v):
            l = math.hypot(v[0], v[1]) or 1.0
            return (v[0] / l, v[1] / l)

        n_in = norm(v_in)
        n_out = norm(v_out)

        dot = n_in[0] * n_out[0] + n_in[1] * n_out[1]
        dot = clamp(dot, -1.0, 1.0)
        turn_angle = math.acos(dot)
        r = radii[i]
        tan_len = r * math.tan(turn_angle / 2.0) if abs(turn_angle) > 1e-6 else 0.0

        bisector = (n_in[0] + n_out[0], n_in[1] + n_out[1])
        bis_len = math.hypot(bisector[0], bisector[1]) or 1.0
        bis = (bisector[0] / bis_len, bisector[1] / bis_len)
        center_offset = r / math.sin(turn_angle / 2.0) if abs(turn_angle) > 1e-6 else 0.0
        cx = p_curr[0] + bis[0] * center_offset
        cy = p_curr[1] + bis[1] * center_offset

        start_angle = math.atan2(p_curr[1] - cy, p_curr[0] - cx)
        end_angle = math.atan2(p_next[1] - cy, p_next[0] - cx)

        # Ensure arc direction is correct: choose shortest arc consistent with turn sign
        # sample points along arc
        arc_points = []
        steps = max(3, int(max(3, abs(turn_angle) / 0.05)))
        # handle wrap-around of angles
        delta = end_angle - start_angle
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        for s in range(steps + 1):
            t = s / steps
            theta = start_angle + delta * t
            ax = cx + r * math.cos(theta)
            ay = cy + r * math.sin(theta)
            arc_points.append((ax, ay))

        approach_x = p_curr[0] - n_in[0] * tan_len
        approach_y = p_curr[1] - n_in[1] * tan_len
        depart_x = p_curr[0] + n_out[0] * tan_len
        depart_y = p_curr[1] + n_out[1] * tan_len

        pts.append((approach_x, approach_y))
        pts.extend(arc_points)
        pts.append((depart_x, depart_y))

    pts = centre_points(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span_x = max(xs) - min(xs) if xs else 1.0
    span_y = max(ys) - min(ys) if ys else 1.0
    scale_x = parameters.width / max(span_x, 1.0)
    scale_y = parameters.height / max(span_y, 1.0)
    scale = min(scale_x, scale_y) * 0.9
    pts = [(p[0] * scale, p[1] * scale) for p in pts]
    return smooth_closed_curve(pts, passes=2)


def smooth_closed_curve(points: list[tuple[float, float]], passes: int = 2) -> list[tuple[float, float]]:
    result = points[:]
    for _ in range(passes):
        smoothed: list[tuple[float, float]] = []
        count = len(result)
        for index in range(count):
            previous = result[(index - 1) % count]
            current = result[index]
            following = result[(index + 1) % count]
            x = (previous[0] + current[0] * 2.0 + following[0]) / 4.0
            y = (previous[1] + current[1] * 2.0 + following[1]) / 4.0
            smoothed.append((x, y))
        result = smoothed
    return result


def centre_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    centre_x = (min_x + max_x) / 2.0
    centre_y = (min_y + max_y) / 2.0
    return [(point[0] - centre_x, point[1] - centre_y) for point in points]


# ============================================================================
# GEOMETRY VALIDATION (reuse earlier helpers)
# ============================================================================
def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if is_zero(o1) and is_zero(o2) and is_zero(o3) and is_zero(o4):
        return False
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def minimum_non_neighbour_distance(points: list[tuple[float, float]]) -> float:
    minimum = float("inf")
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            difference = min(abs(i - j), count - abs(i - j))
            if difference < 5:
                continue
            value = distance(points[i], points[j])
            if value < minimum:
                minimum = value
    return minimum


def count_self_intersections(points: list[tuple[float, float]]) -> int:
    count = len(points)
    intersections = 0
    for i in range(count):
        a = points[i]
        b = points[(i + 1) % count]
        for j in range(i + 1, count):
            difference = min(abs(i - j), count - abs(i - j))
            if difference <= 2:
                continue
            if i == 0 and j == count - 1:
                continue
            c = points[j]
            d = points[(j + 1) % count]
            if segments_intersect(a, b, c, d):
                intersections += 1
    return intersections


def calculate_smoothness(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total_change = 0.0
    count = len(points)
    for index in range(count):
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]
        a1 = math.atan2(current[1] - previous[1], current[0] - previous[0])
        a2 = math.atan2(following[1] - current[1], following[0] - current[0])
        total_change += angle_difference(a1, a2)
    average = total_change / count
    return clamp(100.0 * (1.0 - average / MAX_CURVATURE), 0.0, 100.0)


def calculate_scale_score(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    width_score = clamp(width / 500.0, 0.0, 1.0)
    height_score = clamp(height / 300.0, 0.0, 1.0)
    return width_score * 50.0 + height_score * 50.0


def calculate_complexity_score(points: list[tuple[float, float]]) -> float:
    if len(points) < 10:
        return 0.0
    direction_changes = 0
    count = len(points)
    for index in range(count):
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]
        angle_a = math.atan2(current[1] - previous[1], current[0] - previous[0])
        angle_b = math.atan2(following[1] - current[1], following[0] - current[0])
        if angle_difference(angle_a, angle_b) > 0.035:
            direction_changes += 1
    ratio = direction_changes / count
    return clamp(ratio * 250.0, 0.0, 100.0)


def calculate_variety_score(points: list[tuple[float, float]]) -> float:
    if len(points) < 20:
        return 0.0
    lengths = []
    for index in range(len(points)):
        a = points[index]
        b = points[(index + 1) % len(points)]
        lengths.append(distance(a, b))
    average = sum(lengths) / len(lengths)
    if average == 0.0:
        return 0.0
    variance = sum((value - average) ** 2 for value in lengths) / len(lengths)
    coefficient = math.sqrt(variance) / average
    return clamp(coefficient * 300.0, 0.0, 100.0)


# ============================================================================
# LAP SIMULATOR (point-mass forward/backward solver)
# ============================================================================
def simulate_lap(points: list[tuple[float, float]], car: dict | None = None) -> dict[str, Any]:
    if car is None:
        car = {"mass": 750.0, "max_accel": 6.0, "max_brake": -10.0, "max_power": 120000.0, "grip": 1.2}

    n = len(points)
    if n < 3:
        return {"lap_time": float("inf"), "avg_speed": 0.0, "speed_profile": [], "braking_zones": 0, "speed_variance": 0.0, "length": 0.0}

    dists = []
    radii = []
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        c = points[(i + 2) % n]
        seg = distance(a, b)
        dists.append(seg)
        ax, ay = a; bx, by = b; cx, cy = c
        A = math.hypot(bx - ax, by - ay)
        B = math.hypot(cx - bx, cy - by)
        C = math.hypot(ax - cx, ay - cy)
        area = max(1e-6, abs((ax*(by-cy) + bx*(cy-ay) + cx*(ay-by)) / 2.0))
        R = (A * B * C) / (4.0 * area) if area > 1e-6 else 1e9
        radii.append(R)

    g = 9.81
    lateral_mu = car["grip"]
    v_corner = [math.sqrt(max(0.0, lateral_mu * g * r)) for r in radii]

    speeds = [min(v_corner[i], 120.0) for i in range(n)]

    for i in range(n - 1, -1, -1):
        next_i = (i + 1) % n
        s = dists[i]
        v_next = speeds[next_i]
        a_brake = abs(car["max_brake"])
        v_allowed = math.sqrt(max(0.0, v_next * v_next + 2.0 * a_brake * s))
        if speeds[i] > v_allowed:
            speeds[i] = v_allowed

    for i in range(n):
        prev_i = (i - 1) % n
        s = dists[prev_i]
        a_acc = car["max_accel"]
        v_prev = speeds[prev_i]
        v_possible = math.sqrt(max(0.0, v_prev * v_prev + 2.0 * a_acc * s))
        speeds[i] = min(speeds[i], v_possible)

    lap_time = 0.0
    for i in range(n):
        v = max(0.1, speeds[i])
        lap_time += dists[i] / v

    avg_speed = sum(speeds) / len(speeds)
    braking_zones = 0
    for i in range(n):
        prev = speeds[i - 1]
        cur = speeds[i]
        if prev - cur > 6.0:
            braking_zones += 1

    mean = avg_speed
    variance = sum((v - mean) ** 2 for v in speeds) / len(speeds)
    speed_variance = math.sqrt(variance)
    length = sum(dists)

    return {
        "lap_time": lap_time,
        "avg_speed": avg_speed,
        "speed_profile": speeds,
        "braking_zones": braking_zones,
        "speed_variance": speed_variance,
        "length": length,
    }


# ============================================================================
# SCORING: racing merit using simulator
# ============================================================================
def score_circuit(points: list[tuple[float, float]], parameters: CircuitParameters) -> CircuitScore:
    if len(points) < 20:
        return CircuitScore(total=0.0, closure=0.0, separation=0.0, smoothness=0.0, complexity=0.0, scale=0.0, variety=0.0, validity=0.0, reason="Too few geometry points.")

    closure_distance = distance(points[0], points[-1])
    closure = clamp(100.0 * (1.0 - closure_distance / 100.0), 0.0, 100.0)
    minimum_distance = minimum_non_neighbour_distance(points)
    separation = clamp((minimum_distance / parameters.spacing_limit) * 100.0, 0.0, 100.0)
    intersections = count_self_intersections(points)
    validity = 0.0 if intersections > 0 else 100.0

    sim = simulate_lap(points)
    if sim["lap_time"] == float("inf"):
        return CircuitScore(total=0.0, closure=closure, separation=separation, smoothness=0.0, complexity=0.0, scale=0.0, variety=0.0, validity=validity, reason="Simulation failed.")

    lap_time = sim["lap_time"]
    length = sim["length"]
    avg_speed = sim["avg_speed"]
    braking_zones = sim["braking_zones"]
    speed_var = sim["speed_variance"]

    time_per_km = (lap_time / max(1.0, length)) * 1000.0
    time_score = clamp(100.0 * (1.0 - (time_per_km - 20.0) / 60.0), 0.0, 100.0)
    variety = clamp(speed_var * 2.0, 0.0, 100.0)
    if braking_zones == 0:
        excitement = 20.0
    else:
        excitement = clamp(50.0 + (braking_zones - 3) * 8.0, 0.0, 100.0)
    scale = calculate_scale_score(points)
    smoothness = calculate_smoothness(points)
    complexity_score = calculate_complexity_score(points)

    total = (
        time_score * 0.30
        + variety * 0.20
        + excitement * 0.20
        + smoothness * 0.10
        + complexity_score * 0.10
        + scale * 0.10
    )

    reasons = [f"lap_time={lap_time:.1f}s", f"length={length:.1f}m", f"avg_speed={avg_speed:.1f}m/s", f"braking_zones={braking_zones}"]
    if intersections > 0:
        reasons.append(f"{intersections} self-intersections")
    if minimum_distance < parameters.spacing_limit:
        reasons.append(f"minimum separation {minimum_distance:.1f}")

    return CircuitScore(
        total=round(total, 2),
        closure=round(closure, 2),
        separation=round(separation, 2),
        smoothness=round(smoothness, 2),
        complexity=round(complexity_score, 2),
        scale=round(scale, 2),
        variety=round(variety, 2),
        validity=round(validity, 2),
        reason="; ".join(reasons),
    )


# ============================================================================
# SVG EXPORT
# ============================================================================
def points_to_svg(points: list[tuple[float, float]], track_width: float = TRACK_WIDTH, canvas_width: int = CANVAS_WIDTH, canvas_height: int = CANVAS_HEIGHT) -> str:
    if len(points) < 2:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}"><rect width="100%" height="100%" fill="black"/><text x="50%" y="50%" fill="gray" text-anchor="middle">No geometry</text></svg>'

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    padding_x = canvas_width * SVG_PADDING_RATIO
    padding_y = canvas_height * SVG_PADDING_RATIO
    available_width = canvas_width - 2.0 * padding_x
    available_height = canvas_height - 2.0 * padding_y
    scale = min(available_width / span_x, available_height / span_y)

    centre_x = (min_x + max_x) / 2.0
    centre_y = (min_y + max_y) / 2.0

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x = (point[0] - centre_x) * scale + canvas_width / 2.0
        y = (point[1] - centre_y) * scale + canvas_height / 2.0
        return x, y

    projected = [project(point) for point in points]
    path_commands = [f"M {projected[0][0]:.2f},{projected[0][1]:.2f}"]
    for x, y in projected[1:]:
        path_commands.append(f"L {x:.2f},{y:.2f}")
    path_commands.append("Z")
    path_data = " ".join(path_commands)

    stroke_width = max(2.0, track_width * scale)
    start_x, start_y = projected[0]

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">'
        f'<rect width="100%" height="100%" fill="black"/>'
        f'<path d="{path_data}" fill="none" stroke="white" stroke-width="{stroke_width:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="{path_data}" fill="none" stroke="#444444" stroke-width="{max(1.0, 3.0 * scale):.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="8" fill="red"/>'
        f'<text x="{start_x + 12:.2f}" y="{start_y - 12:.2f}" fill="lime" font-family="sans-serif" font-size="14">START</text>'
        f"</svg>"
    )


# ============================================================================
# DAYTONA EXECUTION: run generator inside Daytona sandbox and return points + score
# ============================================================================
def run_in_sandbox(parameters: CircuitParameters, seed: int) -> tuple[list[tuple[float, float]], CircuitScore, str]:
    # We'll generate geometry locally (faster) and use Daytona to run any heavy or isolated code if needed.
    # For now, generate locally and still use Daytona to run a quick verification step (optional).
    points = generate_segment_layout(parameters, seed)
    score = score_circuit(points, parameters)
    # For traceability, optionally run a lightweight verification in Daytona (e.g., count intersections)
    sandbox = None
    output = ""
    try:
        sandbox = daytona.create()
        # send a tiny verification script to sandbox to compute intersections and return JSON
        code = f"""
import json, math
pts = {json.dumps(points)}
def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def intersects(a,b,c,d):
    def orient(p,q,r): return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    o1 = orient(a,b,c); o2 = orient(a,b,d); o3 = orient(c,d,a); o4 = orient(c,d,b)
    if abs(o1) < 1e-9 and abs(o2) < 1e-9 and abs(o3) < 1e-9 and abs(o4) < 1e-9:
        return False
    return (o1>0) != (o2>0) and (o3>0) != (o4>0)
def count_intersections(pts):
    n = len(pts); total=0
    for i in range(n):
        a=pts[i]; b=pts[(i+1)%n]
        for j in range(i+1,n):
            diff = min(abs(i-j), n-abs(i-j))
            if diff<=2: continue
            if i==0 and j==n-1: continue
            c=pts[j]; d=pts[(j+1)%n]
            if intersects(a,b,c,d): total+=1
    return total
print(json.dumps({{"intersections": count_intersections(pts), "point_count": len(pts)}}))
"""
        result = sandbox.process.code_run(code)
        output = result.result or ""
    except Exception:
        output = "sandbox verification failed"
    finally:
        try:
            if sandbox is not None:
                daytona.delete(sandbox)
        except Exception:
            pass

    return points, score, output


# ============================================================================
# FILE MANAGEMENT
# ============================================================================
def prepare_output_directory() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    BEST_DIRECTORY.mkdir(parents=True, exist_ok=True)


def save_circuit(result: CircuitResult) -> Path:
    circuit_directory = OUTPUT_DIRECTORY / f"circuit_{result.circuit_id:03d}"
    circuit_directory.mkdir(parents=True, exist_ok=True)
    path = circuit_directory / CIRCUIT_FILE_NAME
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)
    svg_path = circuit_directory / CIRCUIT_SVG_FILE_NAME
    svg_markup = points_to_svg([(p[0], p[1]) for p in result.points])
    svg_path.write_text(svg_markup, encoding="utf-8")
    return path


def save_best_circuit(result: CircuitResult) -> Path:
    path = BEST_DIRECTORY / f"circuit_{result.circuit_id:03d}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)
    svg_path = BEST_DIRECTORY / f"circuit_{result.circuit_id:03d}.svg"
    svg_markup = points_to_svg([(p[0], p[1]) for p in result.points])
    svg_path.write_text(svg_markup, encoding="utf-8")
    return path


# ============================================================================
# WORKER
# ============================================================================
class GenerationWorker(QThread):
    progress = pyqtSignal(int, int)
    circuit_started = pyqtSignal(int, int, int)
    iteration_started = pyqtSignal(int, int, int)
    circuit_finished = pyqtSignal(int, float, bool)
    score_updated = pyqtSignal(float, str)
    points_ready = pyqtSignal(list)
    terminal_output = pyqtSignal(str)
    error = pyqtSignal(str)
    completed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.running = True
        self.best_score = 0.0
        self.total_completed = 0

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        prepare_output_directory()
        try:
            for circuit_id in range(1, TOTAL_CIRCUITS + 1):
                if not self.running:
                    break
                self.generate_one_circuit(circuit_id)
                self.total_completed += 1
        finally:
            self.completed.emit()

    def generate_one_circuit(self, circuit_id: int) -> None:
        self.circuit_started.emit(circuit_id, TOTAL_CIRCUITS, self.total_completed)
        parameters = generate_initial_parameters(circuit_id)
        best_result: CircuitResult | None = None
        best_score: float = -1.0

        for iteration in range(1, MAX_ITERATIONS_PER_CIRCUIT + 1):
            if not self.running:
                break
            self.iteration_started.emit(circuit_id, iteration, MAX_ITERATIONS_PER_CIRCUIT)

            seed = circuit_id * 1000 + iteration
            points, score, raw_output = run_in_sandbox(parameters, seed)

            self.points_ready.emit([[p[0], p[1]] for p in points])
            self.terminal_output.emit(raw_output)
            self.score_updated.emit(score.total, score.reason)

            accepted = score.total >= ACCEPTANCE_SCORE and score.validity >= 100.0
            result = CircuitResult(
                circuit_id=circuit_id,
                iteration=iteration,
                parameters=asdict(parameters),
                points=[[p[0], p[1]] for p in points],
                score=asdict(score),
                accepted=accepted,
            )

            save_circuit(result)

            if score.total > best_score:
                best_score = score.total
                best_result = result

            if accepted:
                break

            parameters = improve_parameters(circuit_id, iteration, parameters, score)

            self.progress.emit(self.total_completed, TOTAL_CIRCUITS)

        if best_result is not None:
            save_best_circuit(best_result)
            self.circuit_finished.emit(circuit_id, best_score, best_result.accepted)
        else:
            self.circuit_finished.emit(circuit_id, 0.0, False)


# ============================================================================
# UI
# ============================================================================
class CircuitCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.points: list[QPointF] = []
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.setMinimumSize(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.setMouseTracking(True)

    def set_points(self, points: list[list[float]]) -> None:
        self.points = [QPointF(p[0], p[1]) for p in points]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QBrush(Qt.GlobalColor.black))

        if not self.points:
            painter.setPen(QPen(Qt.GlobalColor.gray))
            painter.setFont(QFont("Sans", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No circuit yet")
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.width() / 2.0 + self.offset_x, self.height() / 2.0 + self.offset_y)
        painter.scale(self.scale, self.scale)

        pen_outer = QPen(Qt.GlobalColor.white)
        pen_outer.setWidthF(TRACK_WIDTH)
        pen_outer.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen_outer.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        pen_inner = QPen(Qt.GlobalColor.darkGray)
        pen_inner.setWidthF(max(1.0, TRACK_WIDTH * 0.6))
        pen_inner.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen_inner.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        painter.setPen(pen_outer)
        for i in range(len(self.points)):
            a = self.points[i]
            b = self.points[(i + 1) % len(self.points)]
            painter.drawLine(a, b)

        painter.setPen(pen_inner)
        for i in range(len(self.points)):
            a = self.points[i]
            b = self.points[(i + 1) % len(self.points)]
            painter.drawLine(a, b)

        if self.points:
            start = self.points[0]
            painter.setBrush(QBrush(Qt.GlobalColor.red))
            painter.setPen(Qt.GlobalColor.red)
            painter.drawEllipse(start, 8, 8)
            painter.setPen(QPen(Qt.GlobalColor.green))
            painter.setFont(QFont("Sans", 10))
            painter.drawText(start + QPointF(12, -12), "START")

    def wheelEvent(self, event: QWheelEvent | None) -> None:  # type: ignore[override]
        if event is None:
            return
        delta = event.angleDelta().y()
        factor = 1.0 + (0.001 * delta)
        self.scale = clamp(self.scale * factor, 0.2, 5.0)
        self.update()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # type: ignore[override]
        if event is None:
            return
        self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # type: ignore[override]
        if event is None:
            return
        if hasattr(self, "_last_pos"):
            delta = event.position() - self._last_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self._last_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:  # type: ignore[override]
        if event is None:
            return
        if hasattr(self, "_last_pos"):
            del self._last_pos


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Procedural Daytona Circuit Generator")

        self.canvas = CircuitCanvas(self)
        self.progress_bar: QProgressBar | None = QProgressBar(self)
        self.status_label: QLabel | None = QLabel(self)
        self.terminal_output: QTextEdit | None = QTextEdit(self)
        self.terminal_output.setReadOnly(True)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        bottom = QHBoxLayout()
        bottom.addWidget(self.progress_bar)
        bottom.addWidget(self.status_label)
        layout.addLayout(bottom)

        layout.addWidget(QLabel("Sandbox output:"))
        layout.addWidget(self.terminal_output)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = GenerationWorker()
        self.worker.progress.connect(self.on_progress)
        self.worker.circuit_started.connect(self.on_circuit_started)
        self.worker.iteration_started.connect(self.on_iteration_started)
        self.worker.circuit_finished.connect(self.on_circuit_finished)
        self.worker.score_updated.connect(self.on_score_updated)
        self.worker.points_ready.connect(self.on_points_ready)
        self.worker.terminal_output.connect(self.on_terminal_output)
        self.worker.error.connect(self.on_error)
        self.worker.completed.connect(self.on_completed)

        self.start_button.clicked.connect(self.start_generation)
        self.stop_button.clicked.connect(self.stop_generation)

    def start_generation(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(TOTAL_CIRCUITS)
        if self.status_label is not None:
            self.status_label.setText("Starting generation...")
        self.worker.running = True
        self.worker.start()

    def stop_generation(self) -> None:
        self.worker.stop()
        if self.status_label is not None:
            self.status_label.setText("Stopping...")

    def on_progress(self, completed: int, total: int) -> None:
        if self.progress_bar is not None:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(completed)

    def on_circuit_started(self, circuit_id: int, total: int, completed: int) -> None:
        if self.status_label is not None:
            self.status_label.setText(f"Circuit {circuit_id}/{total} (completed {completed})")

    def on_iteration_started(self, circuit_id: int, iteration: int, max_iter: int) -> None:
        if self.status_label is not None:
            self.status_label.setText(f"Circuit {circuit_id}, iteration {iteration}/{max_iter}")

    def on_circuit_finished(self, circuit_id: int, score: float, accepted: bool) -> None:
        if self.status_label is not None:
            status = "ACCEPTED" if accepted else "REJECTED"
            self.status_label.setText(f"Circuit {circuit_id} finished with score {score:.2f} ({status})")

    def on_score_updated(self, score: float, reason: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(f"Score {score:.2f}: {reason}")

    def on_points_ready(self, points: list) -> None:
        self.canvas.set_points(points)

    def on_terminal_output(self, text: str) -> None:
        if self.terminal_output is not None:
            self.terminal_output.append(text)

    def on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def on_completed(self) -> None:
        if self.status_label is not None:
            self.status_label.setText("Generation completed.")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.worker.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(CANVAS_WIDTH, CANVAS_HEIGHT + 200)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
