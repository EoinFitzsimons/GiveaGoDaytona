import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from daytona import Daytona, DaytonaConfig

from PyQt6.QtCore import (
    QPointF,
    QRectF,
    QThread,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QFont,
    QPainter,
    QPen,
    QWheelEvent,
    QMouseEvent,
)
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

TOTAL_CIRCUITS = 300

MAX_ITERATIONS_PER_CIRCUIT = 4

ACCEPTANCE_SCORE = 82.0

CANVAS_WIDTH = 1100
CANVAS_HEIGHT = 800

TRACK_WIDTH = 18.0

OUTPUT_DIRECTORY = Path("generated_circuits")
BEST_DIRECTORY = OUTPUT_DIRECTORY / "best"

CIRCUIT_FILE_NAME = "circuit.json"

LLM_MODEL = "openai/gpt-oss-20b"

DAYTONA_TIMEOUT_SECONDS = 30

# Keep prompts deliberately compact.
LLM_MAX_OUTPUT_TOKENS = 700

# Geometry sampling resolution.
POINTS_PER_SEGMENT = 12

# Minimum distance between non-neighbouring track centreline points.
MIN_SELF_DISTANCE = 35.0

# Maximum acceptable curvature proxy.
MAX_CURVATURE = 0.18


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

daytona = Daytona(
    DaytonaConfig(
        api_key=DAYTONA_API_KEY,
    )
)

groq = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


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
# UTILITY FUNCTIONS
# ============================================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(
    a: tuple[float, float],
    b: tuple[float, float],
    amount: float,
) -> tuple[float, float]:
    return (
        a[0] + (b[0] - a[0]) * amount,
        a[1] + (b[1] - a[1]) * amount,
    )


def angle_difference(a: float, b: float) -> float:
    difference = abs(a - b)
    while difference > math.pi:
        difference = abs(difference - 2.0 * math.pi)
    return difference


def point_to_tuple(point: QPointF) -> tuple[float, float]:
    return point.x(), point.y()


def tuple_to_point(point: tuple[float, float]) -> QPointF:
    return QPointF(point[0], point[1])


# ============================================================================
# LLM
# ============================================================================

def llm(messages: list[dict[str, str]]) -> str:
    response = groq.chat.completions.create(
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
        chicane_probability=clamp(
            float(values["chicane_probability"]),
            0.0,
            1.0,
        ),
        hairpin_probability=clamp(
            float(values["hairpin_probability"]),
            0.0,
            1.0,
        ),
        variation=clamp(float(values["variation"]), 0.05, 0.35),
        spacing_limit=clamp(
            float(values["spacing_limit"]),
            35.0,
            100.0,
        ),
        complexity=clamp(float(values["complexity"]), 0.20, 0.95),
        direction=1 if int(values["direction"]) >= 0 else -1,
    )


def generate_initial_parameters(
    circuit_id: int,
) -> CircuitParameters:
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

Return JSON only.

Schema:
{parameter_schema()}
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a procedural motorsport circuit designer. "
                "Return compact valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
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
        {
            "role": "system",
            "content": (
                "You optimise procedural racing circuit parameters. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        response = llm(messages)
        return validate_parameters(extract_json(response))
    except Exception:
        return mutate_parameters(parameters)


def fallback_parameters(circuit_id: int) -> CircuitParameters:
    rng = random.Random(100000 + circuit_id)

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


def mutate_parameters(
    parameters: CircuitParameters,
) -> CircuitParameters:
    result = CircuitParameters(**asdict(parameters))

    result.amplitude = clamp(
        result.amplitude + random.uniform(-0.06, 0.06),
        0.10,
        0.45,
    )

    result.asymmetry = clamp(
        result.asymmetry + random.uniform(-0.06, 0.06),
        0.0,
        0.45,
    )

    result.radius_min = clamp(
        result.radius_min + random.uniform(-15.0, 15.0),
        45.0,
        130.0,
    )

    result.radius_max = clamp(
        result.radius_max + random.uniform(-25.0, 25.0),
        100.0,
        280.0,
    )

    result.spacing_limit = clamp(
        result.spacing_limit + random.uniform(-8.0, 8.0),
        35.0,
        100.0,
    )

    return result


# ============================================================================
# GEOMETRY GENERATION
# ============================================================================

def generate_circuit_points(
    parameters: CircuitParameters,
    seed: int,
) -> list[tuple[float, float]]:
    rng = random.Random(seed)

    point_count = max(
        160,
        parameters.corner_count * POINTS_PER_SEGMENT * 3,
    )

    points: list[tuple[float, float]] = []

    base_radius_x = parameters.width / 2.0
    base_radius_y = parameters.height / 2.0

    phase = rng.uniform(0.0, math.pi * 2.0)

    corner_frequencies = []

    for index in range(parameters.corner_count):
        angle = (
            2.0
            * math.pi
            * index
            / parameters.corner_count
        )

        jitter = rng.uniform(
            -parameters.variation,
            parameters.variation,
        )

        corner_frequencies.append(
            angle + jitter
        )

    corner_frequencies.sort()

    for index in range(point_count):
        theta = (
            2.0
            * math.pi
            * index
            / point_count
        )

        local_noise = (
            math.sin(theta * parameters.corner_count + phase)
            * parameters.amplitude
        )

        secondary_noise = (
            math.sin(
                theta
                * max(2, parameters.corner_count // 2)
                + phase * 0.7
            )
            * parameters.variation
        )

        asymmetry_term = (
            math.cos(theta * 2.0)
            * parameters.asymmetry
        )

        radial_scale = (
            1.0
            + local_noise
            + secondary_noise
            + asymmetry_term
        )

        x = base_radius_x * radial_scale * math.cos(theta)
        y = base_radius_y * radial_scale * math.sin(theta)

        x += (
            math.sin(theta * 3.0 + phase)
            * base_radius_x
            * parameters.complexity
            * 0.08
        )

        y += (
            math.cos(theta * 2.0 + phase)
            * base_radius_y
            * parameters.complexity
            * 0.08
        )

        points.append((x, y))

    points = smooth_closed_curve(points)

    points = apply_turn_features(
        points,
        parameters,
        rng,
    )

    points = smooth_closed_curve(points)

    return points


def smooth_closed_curve(
    points: list[tuple[float, float]],
    passes: int = 2,
) -> list[tuple[float, float]]:
    result = points[:]

    for _ in range(passes):
        smoothed: list[tuple[float, float]] = []

        count = len(result)

        for index in range(count):
            previous = result[(index - 1) % count]
            current = result[index]
            following = result[(index + 1) % count]

            x = (
                previous[0]
                + current[0] * 2.0
                + following[0]
            ) / 4.0

            y = (
                previous[1]
                + current[1] * 2.0
                + following[1]
            ) / 4.0

            smoothed.append((x, y))

        result = smoothed

    return result


def apply_turn_features(
    points: list[tuple[float, float]],
    parameters: CircuitParameters,
    rng: random.Random,
) -> list[tuple[float, float]]:
    result = points[:]

    count = len(result)

    if rng.random() < parameters.hairpin_probability:
        centre = rng.randrange(count)

        for offset in range(-10, 11):
            index = (centre + offset) % count

            amount = abs(offset) / 10.0

            x, y = result[index]

            result[index] = (
                x * (1.0 - 0.10 * (1.0 - amount)),
                y * (1.0 + 0.35 * (1.0 - amount)),
            )

    if rng.random() < parameters.chicane_probability:
        centre = rng.randrange(count)

        for offset in range(-12, 13):
            index = (centre + offset) % count

            x, y = result[index]

            phase = offset / 12.0 * math.pi

            displacement = (
                math.sin(phase)
                * parameters.width
                * 0.07
            )

            result[index] = (
                x,
                y + displacement,
            )

    return result


def centre_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not points:
        return []

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    centre_x = (min_x + max_x) / 2.0
    centre_y = (min_y + max_y) / 2.0

    return [
        (
            point[0] - centre_x,
            point[1] - centre_y,
        )
        for point in points
    ]


# ============================================================================
# GEOMETRY VALIDATION
# ============================================================================

def orientation(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if (
        o1 == 0.0
        and o2 == 0.0
        and o3 == 0.0
        and o4 == 0.0
    ):
        return False

    return (
        (o1 > 0.0) != (o2 > 0.0)
        and (o3 > 0.0) != (o4 > 0.0)
    )


def minimum_non_neighbour_distance(
    points: list[tuple[float, float]],
) -> float:
    minimum = float("inf")
    count = len(points)

    for i in range(count):
        for j in range(i + 1, count):
            difference = min(
                abs(i - j),
                count - abs(i - j),
            )

            if difference < 5:
                continue

            value = distance(points[i], points[j])

            if value < minimum:
                minimum = value

    return minimum


def count_self_intersections(
    points: list[tuple[float, float]],
) -> int:
    count = len(points)
    intersections = 0

    for i in range(count):
        a = points[i]
        b = points[(i + 1) % count]

        for j in range(i + 1, count):
            difference = min(
                abs(i - j),
                count - abs(i - j),
            )

            if difference <= 2:
                continue

            if (
                i == 0
                and j == count - 1
            ):
                continue

            c = points[j]
            d = points[(j + 1) % count]

            if segments_intersect(a, b, c, d):
                intersections += 1

    return intersections


def calculate_smoothness(
    points: list[tuple[float, float]],
) -> float:
    if len(points) < 3:
        return 0.0

    total_change = 0.0
    count = len(points)

    for index in range(count):
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]

        a1 = math.atan2(
            current[1] - previous[1],
            current[0] - previous[0],
        )

        a2 = math.atan2(
            following[1] - current[1],
            following[0] - current[0],
        )

        total_change += angle_difference(a1, a2)

    average = total_change / count

    return clamp(
        100.0 * (1.0 - average / MAX_CURVATURE),
        0.0,
        100.0,
    )


def calculate_scale_score(
    points: list[tuple[float, float]],
) -> float:
    if not points:
        return 0.0

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if width <= 0.0 or height <= 0.0:
        return 0.0

    width_score = clamp(
        width / 500.0,
        0.0,
        1.0,
    )

    height_score = clamp(
        height / 300.0,
        0.0,
        1.0,
    )

    return (
        width_score * 50.0
        + height_score * 50.0
    )


def calculate_complexity_score(
    points: list[tuple[float, float]],
) -> float:
    if len(points) < 10:
        return 0.0

    direction_changes = 0
    count = len(points)

    for index in range(count):
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]

        angle_a = math.atan2(
            current[1] - previous[1],
            current[0] - previous[0],
        )

        angle_b = math.atan2(
            following[1] - current[1],
            following[0] - current[0],
        )

        if angle_difference(angle_a, angle_b) > 0.035:
            direction_changes += 1

    ratio = direction_changes / count

    return clamp(
        ratio * 250.0,
        0.0,
        100.0,
    )


def calculate_variety_score(
    points: list[tuple[float, float]],
) -> float:
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

    variance = sum(
        (value - average) ** 2
        for value in lengths
    ) / len(lengths)

    coefficient = math.sqrt(variance) / average

    return clamp(
        coefficient * 300.0,
        0.0,
        100.0,
    )


def score_circuit(
    points: list[tuple[float, float]],
    parameters: CircuitParameters,
) -> CircuitScore:
    if len(points) < 20:
        return CircuitScore(
            total=0.0,
            closure=0.0,
            separation=0.0,
            smoothness=0.0,
            complexity=0.0,
            scale=0.0,
            variety=0.0,
            validity=0.0,
            reason="Too few geometry points.",
        )

    first = points[0]
    last = points[-1]

    closure_distance = distance(first, last)

    closure = clamp(
        100.0
        * (
            1.0
            - closure_distance / 100.0
        ),
        0.0,
        100.0,
    )

    minimum_distance = minimum_non_neighbour_distance(points)

    separation = clamp(
        (
            minimum_distance
            / parameters.spacing_limit
        )
        * 100.0,
        0.0,
        100.0,
    )

    intersections = count_self_intersections(points)

    if intersections > 0:
        validity = 0.0
    else:
        validity = 100.0

    smoothness = calculate_smoothness(points)
    complexity = calculate_complexity_score(points)
    scale = calculate_scale_score(points)
    variety = calculate_variety_score(points)

    total = (
        closure * 0.15
        + separation * 0.25
        + smoothness * 0.15
        + complexity * 0.15
        + scale * 0.10
        + variety * 0.10
        + validity * 0.10
    )

    reasons = []

    if intersections > 0:
        reasons.append(
            f"{intersections} self-intersections"
        )

    if minimum_distance < parameters.spacing_limit:
        reasons.append(
            f"minimum separation {minimum_distance:.1f}"
        )

    if closure_distance > 20.0:
        reasons.append(
            f"closure error {closure_distance:.1f}"
        )

    if not reasons:
        reasons.append("No major geometric violations.")

    return CircuitScore(
        total=round(total, 2),
        closure=round(closure, 2),
        separation=round(separation, 2),
        smoothness=round(smoothness, 2),
        complexity=round(complexity, 2),
        scale=round(scale, 2),
        variety=round(variety, 2),
        validity=round(validity, 2),
        reason="; ".join(reasons),
    )


# ============================================================================
# DAYTONA EXECUTION
# ============================================================================

def run_in_sandbox(
    parameters: CircuitParameters,
    seed: int,
) -> tuple[list[tuple[float, float]], CircuitScore, str]:
    sandbox = daytona.create()

    code = f"""
import json
import math
import random

parameters = {json.dumps(asdict(parameters))}
seed = {seed}

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def smooth_closed_curve(points, passes=2):
    result = points[:]

    for _ in range(passes):
        smoothed = []
        count = len(result)

        for index in range(count):
            previous = result[(index - 1) % count]
            current = result[index]
            following = result[(index + 1) % count]

            smoothed.append((
                (
                    previous[0]
                    + current[0] * 2.0
                    + following[0]
                ) / 4.0,
                (
                    previous[1]
                    + current[1] * 2.0
                    + following[1]
                ) / 4.0,
            ))

        result = smoothed

    return result

def generate():
    rng = random.Random(seed)

    point_count = max(
        160,
        parameters["corner_count"] * 12 * 3,
    )

    width = parameters["width"]
    height = parameters["height"]

    radius_x = width / 2.0
    radius_y = height / 2.0

    phase = rng.uniform(0.0, math.pi * 2.0)

    points = []

    for index in range(point_count):
        theta = (
            2.0
            * math.pi
            * index
            / point_count
        )

        local_noise = (
            math.sin(
                theta
                * parameters["corner_count"]
                + phase
            )
            * parameters["amplitude"]
        )

        secondary_noise = (
            math.sin(
                theta
                * max(
                    2,
                    parameters["corner_count"] // 2
                )
                + phase * 0.7
            )
            * parameters["variation"]
        )

        asymmetry = (
            math.cos(theta * 2.0)
            * parameters["asymmetry"]
        )

        radial = (
            1.0
            + local_noise
            + secondary_noise
            + asymmetry
        )

        x = (
            radius_x
            * radial
            * math.cos(theta)
        )

        y = (
            radius_y
            * radial
            * math.sin(theta)
        )

        x += (
            math.sin(theta * 3.0 + phase)
            * radius_x
            * parameters["complexity"]
            * 0.08
        )

        y += (
            math.cos(theta * 2.0 + phase)
            * radius_y
            * parameters["complexity"]
            * 0.08
        )

        points.append((x, y))

    return smooth_closed_curve(points)

def orientation(a, b, c):
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )

def intersects(a, b, c, d):
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    return (
        (o1 > 0) != (o2 > 0)
        and (o3 > 0) != (o4 > 0)
    )

def count_intersections(points):
    total = 0
    count = len(points)

    for i in range(count):
        a = points[i]
        b = points[(i + 1) % count]

        for j in range(i + 1, count):
            difference = min(
                abs(i - j),
                count - abs(i - j),
            )

            if difference <= 2:
                continue

            if i == 0 and j == count - 1:
                continue

            c = points[j]
            d = points[(j + 1) % count]

            if intersects(a, b, c, d):
                total += 1

    return total

points = generate()
intersections = count_intersections(points)

print(json.dumps({{
    "points": points,
    "intersections": intersections,
    "point_count": len(points)
}}))
"""

    try:
        result = sandbox.process.code_run(code)

        output = result.result or ""

        parsed = json.loads(output)

        points = [
            (float(point[0]), float(point[1]))
            for point in parsed["points"]
        ]

        points = centre_points(points)

        score = score_circuit(
            points,
            parameters,
        )

        return points, score, output

    finally:
        try:
            daytona.delete(sandbox)
        except Exception:
            pass


# ============================================================================
# FILE MANAGEMENT
# ============================================================================

def prepare_output_directory() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    BEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_circuit(
    result: CircuitResult,
) -> Path:
    circuit_directory = (
        OUTPUT_DIRECTORY
        / f"circuit_{result.circuit_id:03d}"
    )

    circuit_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = circuit_directory / CIRCUIT_FILE_NAME

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
        )

    return path


def save_best_circuit(
    result: CircuitResult,
) -> Path:
    path = (
        BEST_DIRECTORY
        / f"circuit_{result.circuit_id:03d}.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
        )

    return path


# ============================================================================
# WORKER
# ============================================================================

class GenerationWorker(QThread):
    progress = pyqtSignal(int, int)

    circuit_started = pyqtSignal(
        int,
        int,
        int,
    )

    iteration_started = pyqtSignal(
        int,
        int,
        int,
    )

    circuit_finished = pyqtSignal(
        int,
        float,
        bool,
    )

    score_updated = pyqtSignal(
        float,
        str,
    )

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
            for circuit_id in range(
                1,
                TOTAL_CIRCUITS + 1,
            ):
                if not self.running:
                    break

                self.generate_one_circuit(
                    circuit_id
                )

                self.total_completed += 1

                self.progress.emit(
                    self.total_completed,
                    TOTAL_CIRCUITS,
                )

            self.completed.emit()

        except Exception as exc:
            self.error.emit(
                f"{type(exc).__name__}: {exc}"
            )

    def generate_one_circuit(
        self,
        circuit_id: int,
    ) -> None:
        self.circuit_started.emit(
            circuit_id,
            TOTAL_CIRCUITS,
            0,
        )

        parameters = generate_initial_parameters(
            circuit_id
        )

        best_result: CircuitResult | None = None

        for iteration in range(
            1,
            MAX_ITERATIONS_PER_CIRCUIT + 1,
        ):
            if not self.running:
                return

            self.iteration_started.emit(
                circuit_id,
                iteration,
                MAX_ITERATIONS_PER_CIRCUIT,
            )

            self.terminal_output.emit(
                (
                    f"[Circuit {circuit_id:03d}] "
                    f"Iteration {iteration}/"
                    f"{MAX_ITERATIONS_PER_CIRCUIT} "
                    f"generating geometry..."
                )
            )

            seed = (
                circuit_id * 1000
                + iteration
            )

            points, score, output = run_in_sandbox(
                parameters,
                seed,
            )

            self.terminal_output.emit(
                (
                    f"[Circuit {circuit_id:03d}] "
                    f"Daytona output: "
                    f"{len(output)} bytes"
                )
            )

            self.score_updated.emit(
                score.total,
                score.reason,
            )

            self.terminal_output.emit(
                (
                    f"[Circuit {circuit_id:03d}] "
                    f"Score={score.total:.2f} "
                    f"validity={score.validity:.2f} "
                    f"separation={score.separation:.2f} "
                    f"smoothness={score.smoothness:.2f}"
                )
            )

            accepted = (
                score.total >= ACCEPTANCE_SCORE
                and score.validity >= 100.0
                and score.separation >= 90.0
                and score.closure >= 90.0
            )

            result = CircuitResult(
                circuit_id=circuit_id,
                iteration=iteration,
                parameters=asdict(parameters),
                points=[
                    [
                        round(point[0], 3),
                        round(point[1], 3),
                    ]
                    for point in points
                ],
                score=asdict(score),
                accepted=accepted,
            )

            if (
                best_result is None
                or score.total
                > float(
                    best_result.score["total"]
                )
            ):
                best_result = result

            save_circuit(result)

            if accepted:
                self.terminal_output.emit(
                    (
                        f"[Circuit {circuit_id:03d}] "
                        f"ACCEPTED at iteration "
                        f"{iteration}."
                    )
                )

                save_best_circuit(result)

                self.best_score = max(
                    self.best_score,
                    score.total,
                )

                self.circuit_finished.emit(
                    circuit_id,
                    score.total,
                    True,
                )

                return

            if iteration < MAX_ITERATIONS_PER_CIRCUIT:
                self.terminal_output.emit(
                    (
                        f"[Circuit {circuit_id:03d}] "
                        "Score below threshold; "
                        "asking LLM for revised parameters."
                    )
                )

                parameters = improve_parameters(
                    circuit_id,
                    iteration,
                    parameters,
                    score,
                )

        if best_result is not None:
            save_best_circuit(best_result)

            self.best_score = max(
                self.best_score,
                float(
                    best_result.score["total"]
                ),
            )

            self.circuit_finished.emit(
                circuit_id,
                float(
                    best_result.score["total"]
                ),
                False,
            )


# ============================================================================
# CIRCUIT CANVAS
# ============================================================================

class CircuitCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.points: list[QPointF] = []

        self.zoom = 1.0

        self.pan_x = 0.0
        self.pan_y = 0.0

        self.dragging = False

        self.last_mouse_position = QPointF()

        self.setMinimumSize(
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
        )

    def set_points(
        self,
        points: list[list[float]],
    ) -> None:
        self.points = [
            QPointF(
                float(point[0]),
                float(point[1]),
            )
            for point in points
        ]

        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.update()

    def clear(self) -> None:
        self.points = []
        self.update()

    def fit_to_canvas(self) -> None:
        if not self.points:
            return

        xs = [point.x() for point in self.points]
        ys = [point.y() for point in self.points]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        width = max_x - min_x
        height = max_y - min_y

        if width <= 0.0 or height <= 0.0:
            return

        available_width = self.width() * 0.80
        available_height = self.height() * 0.80

        scale_x = available_width / width
        scale_y = available_height / height

        self.zoom = min(
            scale_x,
            scale_y,
        )

        self.pan_x = (
            self.width() / 2.0
            - ((min_x + max_x) / 2.0)
            * self.zoom
        )

        self.pan_y = (
            self.height() / 2.0
            - ((min_y + max_y) / 2.0)
            * self.zoom
        )

        self.update()

    def transform_point(
        self,
        point: QPointF,
    ) -> QPointF:
        return QPointF(
            point.x() * self.zoom + self.pan_x,
            point.y() * self.zoom + self.pan_y,
        )

    def paintEvent(self, a0) -> None:
        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        painter.fillRect(
            self.rect(),
            QBrush(Qt.GlobalColor.black),
        )

        if len(self.points) < 2:
            painter.setPen(
                QPen(Qt.GlobalColor.gray)
            )

            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for circuit geometry...",
            )

            return

        transformed = [
            self.transform_point(point)
            for point in self.points
        ]

        track_pen = QPen(
            Qt.GlobalColor.white,
            max(
                2,
                int(TRACK_WIDTH * self.zoom),
            ),
        )

        track_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )

        track_pen.setJoinStyle(
            Qt.PenJoinStyle.RoundJoin
        )

        painter.setPen(track_pen)

        for index in range(
            len(transformed)
        ):
            first = transformed[index]

            second = transformed[
                (index + 1) % len(transformed)
            ]

            painter.drawLine(
                first,
                second,
            )

        centre_pen = QPen(
            Qt.GlobalColor.darkGray,
            max(
                1,
                int(3 * self.zoom),
            ),
        )

        painter.setPen(centre_pen)

        for index in range(
            len(transformed)
        ):
            first = transformed[index]

            second = transformed[
                (index + 1) % len(transformed)
            ]

            painter.drawLine(
                first,
                second,
            )

        start_point = transformed[0]

        painter.setBrush(
            QBrush(Qt.GlobalColor.red)
        )

        painter.setPen(
            QPen(
                Qt.GlobalColor.red,
                2,
            )
        )

        painter.drawEllipse(
            start_point,
            8.0,
            8.0,
        )

        painter.setPen(
            QPen(Qt.GlobalColor.green, 2)
        )

        painter.drawText(
            start_point + QPointF(12.0, -12.0),
            "START",
        )

    def wheelEvent(self, a0: QWheelEvent) -> None:
        delta = a0.angleDelta().y()

        if delta > 0:
            self.zoom *= 1.15
        else:
            self.zoom /= 1.15

        self.zoom = clamp(
            self.zoom,
            0.1,
            20.0,
        )

        self.update()

    def mousePressEvent(
        self,
        a0: QMouseEvent,
    ) -> None:
        if (
            a0.button()
            == Qt.MouseButton.LeftButton
        ):
            self.dragging = True
            self.last_mouse_position = (
                a0.position()
            )

    def mouseMoveEvent(
        self,
        a0: QMouseEvent,
    ) -> None:
        if not self.dragging:
            return

        current = a0.position()

        delta = current - self.last_mouse_position

        self.pan_x += delta.x()
        self.pan_y += delta.y()

        self.last_mouse_position = current

        self.update()

    def mouseReleaseEvent(
        self,
        a0: QMouseEvent,
    ) -> None:
        if (
            a0.button()
            == Qt.MouseButton.LeftButton
        ):
            self.dragging = False


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.worker: GenerationWorker | None = None

        self.current_circuit = 0

        self.setup_ui()

    def setup_ui(self) -> None:
        self.setWindowTitle(
            "Procedural Circuit AI Lab"
        )

        self.resize(
            1500,
            950,
        )

        central = QWidget()

        self.setCentralWidget(central)

        layout = QHBoxLayout(central)

        left_layout = QVBoxLayout()

        right_layout = QVBoxLayout()

        self.canvas = CircuitCanvas()

        layout.addWidget(
            self.canvas,
            stretch=4,
        )

        layout.addLayout(
            right_layout,
            stretch=1,
        )

        right_layout.addLayout(
            left_layout
        )

        self.status_title = QLabel(
            "Procedural Circuit AI"
        )

        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)

        self.status_title.setFont(
            title_font
        )

        left_layout.addWidget(
            self.status_title
        )

        self.circuit_label = QLabel(
            "Circuit: 0 / 300"
        )

        self.iteration_label = QLabel(
            "Iteration: - / -"
        )

        self.score_label = QLabel(
            "Score: -"
        )

        self.best_label = QLabel(
            "Best score: -"
        )

        self.reason_label = QLabel(
            "Reason: -"
        )

        left_layout.addWidget(
            self.circuit_label
        )

        left_layout.addWidget(
            self.iteration_label
        )

        left_layout.addWidget(
            self.score_label
        )

        left_layout.addWidget(
            self.best_label
        )

        left_layout.addWidget(
            self.reason_label
        )

        left_layout.addSpacing(15)

        self.overall_progress = QProgressBar()

        self.overall_progress.setRange(
            0,
            TOTAL_CIRCUITS,
        )

        self.overall_progress.setValue(0)

        self.overall_progress.setFormat(
            "Overall: %p%"
        )

        left_layout.addWidget(
            self.overall_progress
        )

        self.iteration_progress = QProgressBar()

        self.iteration_progress.setRange(
            0,
            MAX_ITERATIONS_PER_CIRCUIT,
        )

        self.iteration_progress.setValue(0)

        self.iteration_progress.setFormat(
            "Circuit iteration: %v / %m"
        )

        left_layout.addWidget(
            self.iteration_progress
        )

        left_layout.addSpacing(10)

        self.start_button = QPushButton(
            "Start generation"
        )

        self.stop_button = QPushButton(
            "Stop generation"
        )

        self.fit_button = QPushButton(
            "Fit circuit to canvas"
        )

        self.stop_button.setEnabled(False)

        left_layout.addWidget(
            self.start_button
        )

        left_layout.addWidget(
            self.stop_button
        )

        left_layout.addWidget(
            self.fit_button
        )

        left_layout.addSpacing(10)

        terminal_label = QLabel(
            "Daytona / Agent Output"
        )

        left_layout.addWidget(
            terminal_label
        )

        self.terminal = QTextEdit()

        self.terminal.setReadOnly(True)

        self.terminal.setFont(
            QFont(
                "Consolas",
                9,
            )
        )

        left_layout.addWidget(
            self.terminal,
            stretch=1,
        )

        self.start_button.clicked.connect(
            self.start_generation
        )

        self.stop_button.clicked.connect(
            self.stop_generation
        )

        self.fit_button.clicked.connect(
            self.canvas.fit_to_canvas
        )

    def start_generation(self) -> None:
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            return

        self.terminal.clear()

        self.canvas.clear()

        self.overall_progress.setValue(0)

        self.iteration_progress.setValue(0)

        self.status_title.setText(
            "Generating circuits..."
        )

        self.worker = GenerationWorker()

        self.worker.progress.connect(
            self.on_progress
        )

        self.worker.circuit_started.connect(
            self.on_circuit_started
        )

        self.worker.iteration_started.connect(
            self.on_iteration_started
        )

        self.worker.score_updated.connect(
            self.on_score_updated
        )

        self.worker.terminal_output.connect(
            self.on_terminal_output
        )

        self.worker.circuit_finished.connect(
            self.on_circuit_finished
        )

        self.worker.error.connect(
            self.on_error
        )

        self.worker.completed.connect(
            self.on_completed
        )

        self.start_button.setEnabled(False)

        self.stop_button.setEnabled(True)

        self.worker.start()

    def stop_generation(self) -> None:
        if self.worker is None:
            return

        self.worker.stop()

        self.terminal.append(
            "Stopping after the current sandbox operation..."
        )

        self.stop_button.setEnabled(False)

    def on_progress(
        self,
        completed: int,
        total: int,
    ) -> None:
        self.overall_progress.setMaximum(
            total
        )

        self.overall_progress.setValue(
            completed
        )

        self.circuit_label.setText(
            f"Circuit: {completed} / {total}"
        )

    def on_circuit_started(
        self,
        circuit_id: int,
        total: int,
        iteration: int,
    ) -> None:
        self.current_circuit = circuit_id

        self.circuit_label.setText(
            f"Circuit: {circuit_id} / {total}"
        )

        self.score_label.setText(
            "Score: generating..."
        )

        self.reason_label.setText(
            "Reason: generating geometry..."
        )

    def on_iteration_started(
        self,
        circuit_id: int,
        iteration: int,
        maximum: int,
    ) -> None:
        self.iteration_label.setText(
            (
                f"Iteration: {iteration} / "
                f"{maximum}"
            )
        )

        self.iteration_progress.setMaximum(
            maximum
        )

        self.iteration_progress.setValue(
            iteration
        )

    def on_score_updated(
        self,
        score: float,
        reason: str,
    ) -> None:
        self.score_label.setText(
            f"Score: {score:.2f} / 100"
        )

        self.reason_label.setText(
            f"Reason: {reason}"
        )

    def on_terminal_output(
        self,
        text: str,
    ) -> None:
        self.terminal.append(text)

        scrollbar = (
            self.terminal.verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )

    def on_circuit_finished(
        self,
        circuit_id: int,
        score: float,
        accepted: bool,
    ) -> None:
        status = (
            "ACCEPTED"
            if accepted
            else "BEST RESULT SAVED"
        )

        self.terminal.append(
            (
                f"[Circuit {circuit_id:03d}] "
                f"{status} | score={score:.2f}"
            )
        )

        if (
            self.worker is not None
            and self.worker.best_score > 0
        ):
            self.best_label.setText(
                (
                    "Best score: "
                    f"{self.worker.best_score:.2f}"
                )
            )

    def on_error(
        self,
        message: str,
    ) -> None:
        self.terminal.append(
            f"ERROR: {message}"
        )

        QMessageBox.critical(
            self,
            "Generation error",
            message,
        )

        self.start_button.setEnabled(True)

        self.stop_button.setEnabled(False)

    def on_completed(self) -> None:
        self.status_title.setText(
            "Generation complete"
        )

        self.start_button.setEnabled(True)

        self.stop_button.setEnabled(False)

        self.terminal.append(
            (
                f"Completed {TOTAL_CIRCUITS} "
                "circuits."
            )
        )

    def closeEvent(self, a0) -> None:
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            self.worker.stop()

            self.worker.wait(5000)

        a0.accept()


# ============================================================================
# APPLICATION
# ============================================================================

def main() -> None:
    print("=== PROCEDURAL CIRCUIT AI ===")
    print(f"Target circuits: {TOTAL_CIRCUITS}")
    print(
        "Max iterations per circuit:",
        MAX_ITERATIONS_PER_CIRCUIT,
    )
    print(
        "Acceptance score:",
        ACCEPTANCE_SCORE,
    )
    print("Daytona: configured")
    print("Groq: configured")

    prepare_output_directory()

    application = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(
        application.exec()
    )


if __name__ == "__main__":
    main()