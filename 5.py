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
from PyQt6.QtGui import (
    QBrush,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
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

TOTAL_CIRCUITS = 20
MAX_ITERATIONS_PER_CIRCUIT = 20

ACCEPTANCE_SCORE = 82.0

CANVAS_WIDTH = 1100
CANVAS_HEIGHT = 800

TRACK_WIDTH = 18.0

OUTPUT_DIRECTORY = Path("generated_circuits")
BEST_DIRECTORY = OUTPUT_DIRECTORY / "best"

CIRCUIT_FILE_NAME = "circuit.json"

LLM_MODEL = "openai/gpt-oss-20b"
LLM_MAX_OUTPUT_TOKENS = 900

DAYTONA_TIME_SECONDS = 30

POINTS_PER_CORNER = 8

EPS = 1e-9

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

    layout_style: str


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
class AgentDecision:
    circuit_id: int
    iteration: int

    decision_type: str

    summary: str
    parameter_reasoning: str
    direction_reasoning: str

    previous_parameters: dict[str, Any] | None
    new_parameters: dict[str, Any]


@dataclass
class CircuitResult:
    circuit_id: int
    iteration: int

    parameters: dict[str, Any]
    points: list[list[float]]

    score: dict[str, Any]

    accepted: bool

    agent_decision: dict[str, Any]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def lerp(
    a: tuple[float, float],
    b: tuple[float, float],
    amount: float,
) -> tuple[float, float]:
    return (
        a[0] + (b[0] - a[0]) * amount,
        a[1] + (b[1] - a[1]) * amount,
    )


def quadratic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t

    return (
        u * u * p0[0]
        + 2.0 * u * t * p1[0]
        + t * t * p2[0],
        u * u * p0[1]
        + 2.0 * u * t * p1[1]
        + t * t * p2[1],
    )


def angle_difference(
    a: float,
    b: float,
) -> float:
    difference = abs(a - b)

    while difference > math.pi:
        difference = abs(
            difference - 2.0 * math.pi
        )

    return difference


# ============================================================================
# LLM
# ============================================================================

def llm(
    messages: list[dict[str, str]],
) -> str:
    response = groq.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError(
            "LLM returned empty content"
        )

    return content


def extract_json(
    text: str,
) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if (
            lines
            and lines[0].startswith("```")
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(lines)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "LLM response did not contain JSON"
        )

    return json.loads(
        cleaned[start:end + 1]
    )


def parameter_schema() -> str:
    return """
{
  "width": number 1000..1800,
  "height": number 550..1100,
  "corner_count": integer 6..14,
  "straight_count": integer 2..8,
  "radius_min": number 35..120,
  "radius_max": number 90..240,
  "amplitude": number 0.08..0.30,
  "asymmetry": number 0.00..0.40,
  "chicane_probability": number 0..1,
  "hairpin_probability": number 0..1,
  "variation": number 0.05..0.30,
  "spacing_limit": number 35..100,
  "complexity": number 0.20..0.95,
  "direction": integer -1 or 1,
  "layout_style": one of:
    "fast",
    "balanced",
    "technical",
    "hairpin",
    "flowing",
    "asymmetric"
}
""".strip()


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

VALID_LAYOUT_STYLES = {
    "fast",
    "balanced",
    "technical",
    "hairpin",
    "flowing",
    "asymmetric",
}


def validate_parameters(
    data: dict[str, Any],
) -> CircuitParameters:
    defaults = {
        "width": 1300.0,
        "height": 800.0,
        "corner_count": 8,
        "straight_count": 4,
        "radius_min": 60.0,
        "radius_max": 180.0,
        "amplitude": 0.18,
        "asymmetry": 0.15,
        "chicane_probability": 0.20,
        "hairpin_probability": 0.15,
        "variation": 0.15,
        "spacing_limit": 55.0,
        "complexity": 0.55,
        "direction": 1,
        "layout_style": "balanced",
    }

    values = {
        key: data.get(
            key,
            default,
        )
        for key, default in defaults.items()
    }

    layout_style = str(
        values["layout_style"]
    ).strip().lower()

    if layout_style not in VALID_LAYOUT_STYLES:
        layout_style = "balanced"

    return CircuitParameters(
        width=clamp(
            float(values["width"]),
            1000.0,
            1800.0,
        ),
        height=clamp(
            float(values["height"]),
            550.0,
            1100.0,
        ),
        corner_count=int(
            clamp(
                int(values["corner_count"]),
                6,
                14,
            )
        ),
        straight_count=int(
            clamp(
                int(values["straight_count"]),
                2,
                8,
            )
        ),
        radius_min=clamp(
            float(values["radius_min"]),
            35.0,
            120.0,
        ),
        radius_max=clamp(
            float(values["radius_max"]),
            90.0,
            240.0,
        ),
        amplitude=clamp(
            float(values["amplitude"]),
            0.08,
            0.30,
        ),
        asymmetry=clamp(
            float(values["asymmetry"]),
            0.0,
            0.40,
        ),
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
        variation=clamp(
            float(values["variation"]),
            0.05,
            0.30,
        ),
        spacing_limit=clamp(
            float(values["spacing_limit"]),
            35.0,
            100.0,
        ),
        complexity=clamp(
            float(values["complexity"]),
            0.20,
            0.95,
        ),
        direction=(
            1
            if int(values["direction"]) >= 0
            else -1
        ),
        layout_style=layout_style,
    )


# ============================================================================
# INITIAL AGENT DECISION
# ============================================================================

def generate_initial_parameters(
    circuit_id: int,
) -> tuple[
    CircuitParameters,
    AgentDecision,
]:
    prompt = f"""
Design starting parameters for procedural racing circuit {circuit_id}.

The Python geometry engine constructs the actual physical track.

You are responsible for:
1. Choosing a coherent set of starting parameters.
2. Giving this circuit a distinct design identity.
3. Explaining why the starting parameters were selected.
4. Explaining why the chosen driving direction is appropriate.

Do not create geometry.

The geometry engine can produce:
- long straights
- medium-speed corners
- tight corners
- hairpins
- chicanes
- linked corner sequences
- fast flowing sections
- technical sections
- asymmetric sections
- doglegs

Do not make every circuit an oval with small deformation.

The direction is meaningful because it changes the order in which
the circuit's sections are encountered.

Return JSON only:

{{
  "parameters": {parameter_schema()},
  "summary": "short design summary",
  "parameter_reasoning": "specific explanation of the starting values",
  "direction_reasoning": "specific explanation of the selected direction"
}}

Circuit ID: {circuit_id}
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional motorsport "
                "circuit designer and procedural-layout "
                "planner. Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        response = llm(
            messages
        )

        data = extract_json(
            response
        )

        parameters = validate_parameters(
            data["parameters"]
        )

        decision = AgentDecision(
            circuit_id=circuit_id,
            iteration=1,
            decision_type="initial",
            summary=str(
                data.get(
                    "summary",
                    "Initial circuit design selected.",
                )
            ),
            parameter_reasoning=str(
                data.get(
                    "parameter_reasoning",
                    "Initial parameters selected by the design agent.",
                )
            ),
            direction_reasoning=str(
                data.get(
                    "direction_reasoning",
                    "Driving direction selected by the design agent.",
                )
            ),
            previous_parameters=None,
            new_parameters=asdict(
                parameters
            ),
        )

        return (
            parameters,
            decision,
        )

    except Exception as exc:
        parameters = fallback_parameters(
            circuit_id
        )

        decision = AgentDecision(
            circuit_id=circuit_id,
            iteration=1,
            decision_type="fallback_initial",
            summary=(
                "The initial LLM request failed; "
                f"fallback parameters were used "
                f"({type(exc).__name__})."
            ),
            parameter_reasoning=(
                "Fallback parameters were generated "
                "from the circuit ID."
            ),
            direction_reasoning=(
                "Fallback direction was selected "
                "from the two legal driving directions."
            ),
            previous_parameters=None,
            new_parameters=asdict(
                parameters
            ),
        )

        return (
            parameters,
            decision,
        )


# ============================================================================
# ITERATIVE AGENT IMPROVEMENT
# ============================================================================

def improve_parameters(
    circuit_id: int,
    iteration: int,
    parameters: CircuitParameters,
    score: CircuitScore,
) -> tuple[
    CircuitParameters,
    AgentDecision,
]:
    previous = asdict(
        parameters
    )

    prompt = f"""
Optimise procedural racing circuit {circuit_id}.

This is optimisation iteration {iteration} of
{MAX_ITERATIONS_PER_CIRCUIT}.

Previous parameters:
{json.dumps(previous, separators=(",", ":"))}

Previous evaluation:
{json.dumps(asdict(score), separators=(",", ":"))}

The circuit geometry generated from those parameters was evaluated
by deterministic geometry checks.

Your task is to produce a NEW parameter set that addresses the
weaknesses identified by the evaluation.

You must explicitly consider:

1. Which parameters should change.
2. Why each important parameter should change.
3. What geometric consequence those changes should have.
4. Whether the circuit should become faster, slower, more technical,
   more flowing, more asymmetric, or more varied.
5. Whether the driving direction should be retained or changed.
6. Why moving away from the previous parameter set is justified.
7. How the proposed changes should address the score.

Do not create geometry.

Do not simply make random small numerical changes.

Return JSON only:

{{
  "parameters": {parameter_schema()},
  "summary": "summary of the revised design",
  "parameter_reasoning": "specific explanation of parameter changes",
  "direction_reasoning": "specific explanation of direction decision"
}}
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You optimise procedural motorsport "
                "circuit parameters. You must explain "
                "your changes using the previous "
                "evaluation. Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        response = llm(
            messages
        )

        data = extract_json(
            response
        )

        revised = validate_parameters(
            data["parameters"]
        )

        decision = AgentDecision(
            circuit_id=circuit_id,
            iteration=iteration,
            decision_type="revision",
            summary=str(
                data.get(
                    "summary",
                    "Parameters revised after evaluation.",
                )
            ),
            parameter_reasoning=str(
                data.get(
                    "parameter_reasoning",
                    "Parameters revised using the previous evaluation.",
                )
            ),
            direction_reasoning=str(
                data.get(
                    "direction_reasoning",
                    "Driving direction evaluated during optimisation.",
                )
            ),
            previous_parameters=previous,
            new_parameters=asdict(
                revised
            ),
        )

        return (
            revised,
            decision,
        )

    except Exception as exc:
        revised = mutate_parameters(
            parameters
        )

        decision = AgentDecision(
            circuit_id=circuit_id,
            iteration=iteration,
            decision_type="fallback_revision",
            summary=(
                "The optimisation request failed; "
                f"controlled mutation was used "
                f"({type(exc).__name__})."
            ),
            parameter_reasoning=(
                "Continuous geometric parameters "
                "were changed within their allowed bounds."
            ),
            direction_reasoning=(
                "The existing driving direction was "
                "retained by the fallback mutation."
            ),
            previous_parameters=previous,
            new_parameters=asdict(
                revised
            ),
        )

        return (
            revised,
            decision,
        )


# ============================================================================
# FALLBACK PARAMETERS
# ============================================================================

def fallback_parameters(
    circuit_id: int,
) -> CircuitParameters:
    rng = random.Random(
        100000 + circuit_id
    )

    styles = [
        "fast",
        "balanced",
        "technical",
        "hairpin",
        "flowing",
        "asymmetric",
    ]

    return CircuitParameters(
        width=rng.uniform(
            1100.0,
            1600.0,
        ),
        height=rng.uniform(
            600.0,
            950.0,
        ),
        corner_count=rng.randint(
            6,
            11,
        ),
        straight_count=rng.randint(
            3,
            7,
        ),
        radius_min=rng.uniform(
            45.0,
            90.0,
        ),
        radius_max=rng.uniform(
            130.0,
            220.0,
        ),
        amplitude=rng.uniform(
            0.10,
            0.25,
        ),
        asymmetry=rng.uniform(
            0.05,
            0.30,
        ),
        chicane_probability=rng.uniform(
            0.10,
            0.35,
        ),
        hairpin_probability=rng.uniform(
            0.05,
            0.30,
        ),
        variation=rng.uniform(
            0.08,
            0.25,
        ),
        spacing_limit=rng.uniform(
            45.0,
            75.0,
        ),
        complexity=rng.uniform(
            0.35,
            0.80,
        ),
        direction=rng.choice(
            [-1, 1]
        ),
        layout_style=rng.choice(
            styles
        ),
    )


# ============================================================================
# FALLBACK PARAMETER MUTATION
# ============================================================================

def mutate_parameters(
    parameters: CircuitParameters,
) -> CircuitParameters:
    result = CircuitParameters(
        **asdict(parameters)
    )

    result.width = clamp(
        result.width
        + random.uniform(
            -100.0,
            100.0,
        ),
        1000.0,
        1800.0,
    )

    result.height = clamp(
        result.height
        + random.uniform(
            -70.0,
            70.0,
        ),
        550.0,
        1100.0,
    )

    result.corner_count = int(
        clamp(
            result.corner_count
            + random.choice(
                [-1, 0, 0, 1]
            ),
            6,
            14,
        )
    )

    result.straight_count = int(
        clamp(
            result.straight_count
            + random.choice(
                [-1, 0, 0, 1]
            ),
            2,
            8,
        )
    )

    result.radius_min = clamp(
        result.radius_min
        + random.uniform(
            -15.0,
            15.0,
        ),
        35.0,
        120.0,
    )

    result.radius_max = clamp(
        result.radius_max
        + random.uniform(
            -25.0,
            25.0,
        ),
        90.0,
        240.0,
    )

    if result.radius_max <= result.radius_min:
        result.radius_max = min(
            240.0,
            result.radius_min + 30.0,
        )

    result.amplitude = clamp(
        result.amplitude
        + random.uniform(
            -0.04,
            0.04,
        ),
        0.08,
        0.30,
    )

    result.asymmetry = clamp(
        result.asymmetry
        + random.uniform(
            -0.05,
            0.05,
        ),
        0.0,
        0.40,
    )

    result.chicane_probability = clamp(
        result.chicane_probability
        + random.uniform(
            -0.08,
            0.08,
        ),
        0.0,
        1.0,
    )

    result.hairpin_probability = clamp(
        result.hairpin_probability
        + random.uniform(
            -0.08,
            0.08,
        ),
        0.0,
        1.0,
    )

    result.variation = clamp(
        result.variation
        + random.uniform(
            -0.04,
            0.04,
        ),
        0.05,
        0.30,
    )

    result.complexity = clamp(
        result.complexity
        + random.uniform(
            -0.08,
            0.08,
        ),
        0.20,
        0.95,
    )

    return result


# ============================================================================
# GEOMETRY
# ============================================================================

def generate_circuit_points(
    parameters: CircuitParameters,
    seed: int,
) -> list[tuple[float, float]]:
    rng = random.Random(
        seed
    )

    vertex_count = (
        parameters.corner_count
        + parameters.straight_count
    )

    vertex_count = max(
        10,
        min(
            22,
            vertex_count,
        ),
    )

    radius_x = (
        parameters.width
        * 0.46
    )

    radius_y = (
        parameters.height
        * 0.46
    )

    style = parameters.layout_style

    if style == "fast":
        corner_strength = 0.65
        radial_variation = 0.08
        straight_bias = 1.40
        tight_corner_bias = 0.10

    elif style == "technical":
        corner_strength = 1.25
        radial_variation = 0.14
        straight_bias = 0.65
        tight_corner_bias = 0.55

    elif style == "hairpin":
        corner_strength = 1.40
        radial_variation = 0.17
        straight_bias = 0.85
        tight_corner_bias = 0.85

    elif style == "flowing":
        corner_strength = 0.75
        radial_variation = 0.11
        straight_bias = 1.10
        tight_corner_bias = 0.15

    elif style == "asymmetric":
        corner_strength = 1.00
        radial_variation = 0.21
        straight_bias = 1.00
        tight_corner_bias = 0.40

    else:
        corner_strength = 0.90
        radial_variation = 0.12
        straight_bias = 1.00
        tight_corner_bias = 0.30

    # ------------------------------------------------------------------------
    # Uneven angular spacing.
    # ------------------------------------------------------------------------

    gap_weights: list[float] = []

    for index in range(
        vertex_count
    ):
        weight = rng.uniform(
            0.55,
            1.45,
        )

        weight *= (
            1.0
            + rng.uniform(
                -0.10,
                0.10,
            )
            * parameters.complexity
        )

        if (
            index % 3 == 0
            and rng.random()
            < parameters.complexity
        ):
            weight *= straight_bias

        gap_weights.append(
            max(
                0.20,
                weight,
            )
        )

    for _ in range(
        min(
            parameters.straight_count,
            vertex_count // 2,
        )
    ):
        selected = rng.randrange(
            vertex_count
        )

        gap_weights[selected] *= (
            1.30
            + parameters.complexity
            * 0.70
        )

    total_weight = sum(
        gap_weights
    )

    angles: list[float] = []

    current_angle = rng.uniform(
        0.0,
        2.0 * math.pi,
    )

    for weight in gap_weights:
        angles.append(
            current_angle
        )

        current_angle += (
            2.0
            * math.pi
            * weight
            / total_weight
        )

    # Reverse traversal for opposite driving direction.
    if parameters.direction < 0:
        angles = list(
            reversed(
                angles
            )
        )

    # ------------------------------------------------------------------------
    # Low-frequency deformation.
    # ------------------------------------------------------------------------

    radial_values: list[float] = []

    base_phase = rng.uniform(
        0.0,
        2.0 * math.pi,
    )

    for theta in angles:

        low_frequency = math.sin(
            theta * 1.5
            + base_phase
        )

        medium_frequency = math.sin(
            theta * 2.5
            - base_phase * 0.6
        )

        value = (
            1.0
            + low_frequency
            * radial_variation
            * parameters.amplitude
            + medium_frequency
            * parameters.asymmetry
            * 0.15
        )

        radial_values.append(
            clamp(
                value,
                0.72,
                1.30,
            )
        )

    # ------------------------------------------------------------------------
    # Build polygon.
    # ------------------------------------------------------------------------

    polygon: list[
        tuple[float, float]
    ] = []

    for index in range(
        vertex_count
    ):
        theta = angles[index]
        radial = radial_values[index]

        if (
            style == "asymmetric"
            and index % 4 == 0
        ):
            radial *= rng.uniform(
                0.85,
                1.15,
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

        polygon.append(
            (
                x,
                y,
            )
        )

    # ------------------------------------------------------------------------
    # Features.
    # ------------------------------------------------------------------------

    if (
        rng.random()
        < parameters.hairpin_probability
    ):
        apply_hairpin_feature(
            polygon,
            rng,
            strength=(
                1.0
                if style != "fast"
                else 0.65
            ),
        )

    if (
        rng.random()
        < parameters.chicane_probability
    ):
        apply_chicane_feature(
            polygon,
            rng,
        )

    if parameters.complexity > 0.70:
        apply_dogleg_feature(
            polygon,
            rng,
        )

    # ------------------------------------------------------------------------
    # Corner radius selection.
    # ------------------------------------------------------------------------

    radii: list[float] = []

    for index in range(
        vertex_count
    ):
        previous = polygon[
            (index - 1)
            % vertex_count
        ]

        current = polygon[index]

        following = polygon[
            (index + 1)
            % vertex_count
        ]

        incoming_length = distance(
            previous,
            current,
        )

        outgoing_length = distance(
            current,
            following,
        )

        shortest_edge = min(
            incoming_length,
            outgoing_length,
        )

        turn = local_turn_angle(
            previous,
            current,
            following,
        )

        radius = rng.uniform(
            parameters.radius_min,
            parameters.radius_max,
        )

        if (
            turn
            > math.radians(80.0)
            and rng.random()
            < (
                tight_corner_bias
                + 0.15
            )
        ):
            radius *= 0.55

        if style in {
            "technical",
            "hairpin",
        }:
            radius *= 0.55
        else:
            radius *= 0.78

        radius = clamp(
            radius,
            25.0,
            max(
                25.0,
                shortest_edge * 0.34,
            ),
        )

        radii.append(
            radius
        )

    centreline = round_polygon(
        polygon,
        radii,
    )

    centreline = centre_points(
        centreline
    )

    if len(centreline) < 30:
        return fallback_geometry(
            parameters,
            seed,
        )

    # ------------------------------------------------------------------------
    # Scale to requested dimensions.
    # ------------------------------------------------------------------------

    xs = [
        point[0]
        for point in centreline
    ]

    ys = [
        point[1]
        for point in centreline
    ]

    span_x = (
        max(xs)
        - min(xs)
    )

    span_y = (
        max(ys)
        - min(ys)
    )

    if (
        span_x <= EPS
        or span_y <= EPS
    ):
        return fallback_geometry(
            parameters,
            seed,
        )

    scale_x = (
        parameters.width
        * 0.90
        / span_x
    )

    scale_y = (
        parameters.height
        * 0.90
        / span_y
    )

    scale = min(
        scale_x,
        scale_y,
    )

    return [
        (
            point[0] * scale,
            point[1] * scale,
        )
        for point in centreline
    ]


def local_turn_angle(
    previous: tuple[float, float],
    current: tuple[float, float],
    following: tuple[float, float],
) -> float:

    incoming = (
        current[0] - previous[0],
        current[1] - previous[1],
    )

    outgoing = (
        following[0] - current[0],
        following[1] - current[1],
    )

    incoming_length = math.hypot(
        incoming[0],
        incoming[1],
    )

    outgoing_length = math.hypot(
        outgoing[0],
        outgoing[1],
    )

    if (
        incoming_length < EPS
        or outgoing_length < EPS
    ):
        return 0.0

    dot = (
        incoming[0]
        * outgoing[0]
        + incoming[1]
        * outgoing[1]
    )

    dot /= (
        incoming_length
        * outgoing_length
    )

    dot = clamp(
        dot,
        -1.0,
        1.0,
    )

    return math.acos(
        dot
    )


def apply_hairpin_feature(
    polygon: list[tuple[float, float]],
    rng: random.Random,
    strength: float,
) -> None:

    count = len(polygon)

    if count < 6:
        return

    centre = rng.randrange(
        count
    )

    previous_index = (
        centre - 1
    ) % count

    next_index = (
        centre + 1
    ) % count

    previous = polygon[
        previous_index
    ]

    current = polygon[
        centre
    ]

    following = polygon[
        next_index
    ]

    tangent = (
        following[0]
        - previous[0],
        following[1]
        - previous[1],
    )

    tangent_length = math.hypot(
        tangent[0],
        tangent[1],
    )

    if tangent_length < EPS:
        return

    normal = (
        -tangent[1]
        / tangent_length,
        tangent[0]
        / tangent_length,
    )

    local_scale = (
        180.0
        + min(
            abs(current[0]),
            abs(current[1]),
        )
    )

    amount = (
        local_scale
        * 0.18
        * strength
    )

    polygon[centre] = (
        current[0]
        + normal[0] * amount,
        current[1]
        + normal[1] * amount,
    )


def apply_chicane_feature(
    polygon: list[tuple[float, float]],
    rng: random.Random,
) -> None:

    count = len(polygon)

    if count < 8:
        return

    centre = rng.randrange(
        count
    )

    for offset in (
        -1,
        0,
        1,
    ):

        index = (
            centre
            + offset
        ) % count

        previous = polygon[
            (index - 1)
            % count
        ]

        following = polygon[
            (index + 1)
            % count
        ]

        tangent = (
            following[0]
            - previous[0],
            following[1]
            - previous[1],
        )

        length = math.hypot(
            tangent[0],
            tangent[1],
        )

        if length < EPS:
            continue

        normal = (
            -tangent[1] / length,
            tangent[0] / length,
        )

        magnitude = (
            55.0
            if offset == 0
            else 30.0
        )

        if offset < 0:
            magnitude *= -1.0

        current = polygon[
            index
        ]

        polygon[index] = (
            current[0]
            + normal[0] * magnitude,
            current[1]
            + normal[1] * magnitude,
        )


def apply_dogleg_feature(
    polygon: list[tuple[float, float]],
    rng: random.Random,
) -> None:

    count = len(polygon)

    if count < 8:
        return

    first_index = rng.randrange(
        count
    )

    second_index = (
        first_index + 1
    ) % count

    first = polygon[
        first_index
    ]

    second = polygon[
        second_index
    ]

    dx = (
        second[0]
        - first[0]
    )

    dy = (
        second[1]
        - first[1]
    )

    length = math.hypot(
        dx,
        dy,
    )

    if length < EPS:
        return

    normal = (
        -dy / length,
        dx / length,
    )

    amount = rng.uniform(
        30.0,
        75.0,
    )

    polygon[first_index] = (
        first[0]
        + normal[0] * amount,
        first[1]
        + normal[1] * amount,
    )

    polygon[second_index] = (
        second[0]
        + normal[0] * amount * 0.60,
        second[1]
        + normal[1] * amount * 0.60,
    )


def round_polygon(
    polygon: list[tuple[float, float]],
    radii: list[float],
) -> list[tuple[float, float]]:

    count = len(
        polygon
    )

    if count < 3:
        return polygon[:]

    entry_points: list[
        tuple[float, float]
    ] = []

    exit_points: list[
        tuple[float, float]
    ] = []

    for index in range(
        count
    ):

        previous = polygon[
            (index - 1)
            % count
        ]

        current = polygon[
            index
        ]

        following = polygon[
            (index + 1)
            % count
        ]

        incoming = (
            current[0]
            - previous[0],
            current[1]
            - previous[1],
        )

        outgoing = (
            following[0]
            - current[0],
            following[1]
            - current[1],
        )

        incoming_length = math.hypot(
            incoming[0],
            incoming[1],
        )

        outgoing_length = math.hypot(
            outgoing[0],
            outgoing[1],
        )

        if (
            incoming_length < EPS
            or outgoing_length < EPS
        ):
            entry_points.append(
                current
            )

            exit_points.append(
                current
            )

            continue

        incoming_unit = (
            incoming[0]
            / incoming_length,
            incoming[1]
            / incoming_length,
        )

        outgoing_unit = (
            outgoing[0]
            / outgoing_length,
            outgoing[1]
            / outgoing_length,
        )

        turn_angle = local_turn_angle(
            previous,
            current,
            following,
        )

        if turn_angle < 0.02:

            entry_points.append(
                current
            )

            exit_points.append(
                current
            )

            continue

        cut = clamp(
            radii[index],
            8.0,
            min(
                incoming_length,
                outgoing_length,
            ) * 0.35,
        )

        entry_points.append(
            (
                current[0]
                - incoming_unit[0]
                * cut,
                current[1]
                - incoming_unit[1]
                * cut,
            )
        )

        exit_points.append(
            (
                current[0]
                + outgoing_unit[0]
                * cut,
                current[1]
                + outgoing_unit[1]
                * cut,
            )
        )

    result: list[
        tuple[float, float]
    ] = []

    for index in range(
        count
    ):

        entry = entry_points[
            index
        ]

        exit_point = exit_points[
            index
        ]

        vertex = polygon[
            index
        ]

        if not result:
            result.append(
                entry
            )

        curve_steps = max(
            4,
            POINTS_PER_CORNER,
        )

        for step in range(
            1,
            curve_steps + 1,
        ):

            t = (
                step
                / curve_steps
            )

            result.append(
                quadratic_bezier(
                    entry,
                    vertex,
                    exit_point,
                    t,
                )
            )

        next_entry = entry_points[
            (index + 1)
            % count
        ]

        line_length = distance(
            exit_point,
            next_entry,
        )

        line_steps = max(
            2,
            int(
                line_length
                / 35.0
            ),
        )

        for step in range(
            1,
            line_steps + 1,
        ):

            result.append(
                lerp(
                    exit_point,
                    next_entry,
                    step / line_steps,
                )
            )

    cleaned: list[
        tuple[float, float]
    ] = []

    for point in result:

        if (
            not cleaned
            or distance(
                cleaned[-1],
                point,
            ) > 0.5
        ):
            cleaned.append(
                point
            )

    return cleaned


def fallback_geometry(
    parameters: CircuitParameters,
    seed: int,
) -> list[tuple[float, float]]:

    rng = random.Random(
        seed
    )

    count = 96

    rx = parameters.width * 0.45
    ry = parameters.height * 0.45

    points: list[
        tuple[float, float]
    ] = []

    for index in range(
        count
    ):

        theta = (
            2.0
            * math.pi
            * index
            / count
        )

        radial = (
            1.0
            + rng.uniform(
                -0.04,
                0.04,
            )
        )

        points.append(
            (
                rx
                * radial
                * math.cos(theta),
                ry
                * radial
                * math.sin(theta),
            )
        )

    return points


def centre_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:

    if not points:
        return []

    min_x = min(
        point[0]
        for point in points
    )

    max_x = max(
        point[0]
        for point in points
    )

    min_y = min(
        point[1]
        for point in points
    )

    max_y = max(
        point[1]
        for point in points
    )

    centre_x = (
        min_x + max_x
    ) / 2.0

    centre_y = (
        min_y + max_y
    ) / 2.0

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
        (b[0] - a[0])
        * (c[1] - a[1])
        - (b[1] - a[1])
        * (c[0] - a[0])
    )


def segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:

    o1 = orientation(
        a,
        b,
        c,
    )

    o2 = orientation(
        a,
        b,
        d,
    )

    o3 = orientation(
        c,
        d,
        a,
    )

    o4 = orientation(
        c,
        d,
        b,
    )

    if (
        abs(o1) < EPS
        and abs(o2) < EPS
        and abs(o3) < EPS
        and abs(o4) < EPS
    ):
        return False

    return (
        (o1 > 0.0)
        != (o2 > 0.0)
        and
        (o3 > 0.0)
        != (o4 > 0.0)
    )


def minimum_non_neighbour_distance(
    points: list[tuple[float, float]],
) -> float:

    minimum = float(
        "inf"
    )

    count = len(
        points
    )

    for i in range(
        count
    ):

        for j in range(
            i + 1,
            count,
        ):

            difference = min(
                abs(i - j),
                count - abs(i - j),
            )

            if difference < 8:
                continue

            value = distance(
                points[i],
                points[j],
            )

            if value < minimum:
                minimum = value

    return minimum


def count_self_intersections(
    points: list[tuple[float, float]],
) -> int:

    count = len(
        points
    )

    intersections = 0

    for i in range(
        count
    ):

        a = points[
            i
        ]

        b = points[
            (i + 1)
            % count
        ]

        for j in range(
            i + 1,
            count,
        ):

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

            c = points[
                j
            ]

            d = points[
                (j + 1)
                % count
            ]

            if segments_intersect(
                a,
                b,
                c,
                d,
            ):
                intersections += 1

    return intersections


def calculate_smoothness(
    points: list[tuple[float, float]],
) -> float:

    if len(points) < 3:
        return 0.0

    total_change = 0.0

    count = len(
        points
    )

    for index in range(
        count
    ):

        previous = points[
            (index - 1)
            % count
        ]

        current = points[
            index
        ]

        following = points[
            (index + 1)
            % count
        ]

        a1 = math.atan2(
            current[1]
            - previous[1],
            current[0]
            - previous[0],
        )

        a2 = math.atan2(
            following[1]
            - current[1],
            following[0]
            - current[0],
        )

        total_change += angle_difference(
            a1,
            a2,
        )

    average = (
        total_change
        / count
    )

    return clamp(
        100.0
        * (
            1.0
            - average
            / MAX_CURVATURE
        ),
        0.0,
        100.0,
    )


def calculate_scale_score(
    points: list[tuple[float, float]],
) -> float:

    if not points:
        return 0.0

    xs = [
        point[0]
        for point in points
    ]

    ys = [
        point[1]
        for point in points
    ]

    width = (
        max(xs)
        - min(xs)
    )

    height = (
        max(ys)
        - min(ys)
    )

    if (
        width <= 0.0
        or height <= 0.0
    ):
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

    count = len(
        points
    )

    for index in range(
        count
    ):

        previous = points[
            (index - 1)
            % count
        ]

        current = points[
            index
        ]

        following = points[
            (index + 1)
            % count
        ]

        angle_a = math.atan2(
            current[1]
            - previous[1],
            current[0]
            - previous[0],
        )

        angle_b = math.atan2(
            following[1]
            - current[1],
            following[0]
            - current[0],
        )

        if angle_difference(
            angle_a,
            angle_b,
        ) > 0.035:
            direction_changes += 1

    ratio = (
        direction_changes
        / count
    )

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

    lengths: list[float] = []

    for index in range(
        len(points)
    ):

        lengths.append(
            distance(
                points[index],
                points[
                    (index + 1)
                    % len(points)
                ],
            )
        )

    average = (
        sum(lengths)
        / len(lengths)
    )

    if average <= EPS:
        return 0.0

    variance = (
        sum(
            (
                value
                - average
            ) ** 2
            for value in lengths
        )
        / len(lengths)
    )

    coefficient = (
        math.sqrt(variance)
        / average
    )

    return clamp(
        coefficient * 300.0,
        0.0,
        100.0,
    )


def calculate_closure_score(
    points: list[tuple[float, float]],
) -> float:

    if len(points) < 3:
        return 0.0

    final_segment = distance(
        points[-2],
        points[-1],
    )

    closing_distance = distance(
        points[-1],
        points[0],
    )

    reference = max(
        final_segment,
        1.0,
    )

    ratio = (
        closing_distance
        / reference
    )

    return clamp(
        100.0
        * (
            1.0
            - max(
                0.0,
                ratio - 0.5,
            )
        ),
        0.0,
        100.0,
    )


# ============================================================================
# SCORING
# ============================================================================

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

    closure = calculate_closure_score(
        points
    )

    minimum_distance = (
        minimum_non_neighbour_distance(
            points
        )
    )

    separation = clamp(
        (
            minimum_distance
            / parameters.spacing_limit
        )
        * 100.0,
        0.0,
        100.0,
    )

    intersections = (
        count_self_intersections(
            points
        )
    )

    validity = (
        0.0
        if intersections > 0
        else 100.0
    )

    smoothness = (
        calculate_smoothness(
            points
        )
    )

    complexity = (
        calculate_complexity_score(
            points
        )
    )

    scale = (
        calculate_scale_score(
            points
        )
    )

    raw_variety = (
        calculate_variety_score(
            points
        )
    )

    variety_quality = (
        100.0
        - abs(
            raw_variety - 55.0
        )
        * 1.5
    )

    variety_quality = clamp(
        variety_quality,
        0.0,
        100.0,
    )

    total = (
        closure * 0.10
        + separation * 0.20
        + smoothness * 0.18
        + complexity * 0.16
        + scale * 0.10
        + variety_quality * 0.16
        + validity * 0.10
    )

    reasons: list[str] = []

    if intersections > 0:
        reasons.append(
            f"{intersections} self-intersections"
        )

    if (
        minimum_distance
        < parameters.spacing_limit
    ):
        reasons.append(
            (
                "minimum separation "
                f"{minimum_distance:.1f}"
            )
        )

    if closure < 85.0:
        reasons.append(
            f"closure score {closure:.1f}"
        )

    if smoothness < 55.0:
        reasons.append(
            f"smoothness {smoothness:.1f}"
        )

    if not reasons:
        reasons.append(
            "No major geometric violations."
        )

    return CircuitScore(
        total=round(
            total,
            2,
        ),
        closure=round(
            closure,
            2,
        ),
        separation=round(
            separation,
            2,
        ),
        smoothness=round(
            smoothness,
            2,
        ),
        complexity=round(
            complexity,
            2,
        ),
        scale=round(
            scale,
            2,
        ),
        variety=round(
            variety_quality,
            2,
        ),
        validity=round(
            validity,
            2,
        ),
        reason="; ".join(
            reasons
        ),
    )


# ============================================================================
# DAYTONA VERIFICATION
# ============================================================================

def run_in_sandbox(
    parameters: CircuitParameters,
    seed: int,
) -> tuple[
    list[tuple[float, float]],
    CircuitScore,
    str,
]:
    points = generate_circuit_points(
        parameters,
        seed,
    )

    score = score_circuit(
        points,
        parameters,
    )

    sandbox = None

    try:
        sandbox = daytona.create()

        code = f"""
import json
import math

points = {json.dumps(points)}

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

    if (
        abs(o1) < 1e-9
        and abs(o2) < 1e-9
        and abs(o3) < 1e-9
        and abs(o4) < 1e-9
    ):
        return False

    return (
        (o1 > 0) != (o2 > 0)
        and
        (o3 > 0) != (o4 > 0)
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

intersections = count_intersections(points)

print(json.dumps({{
    "point_count": len(points),
    "intersections": intersections,
    "first_point": points[0],
    "last_point": points[-1]
}}))
"""

        result = sandbox.process.code_run(
            code
        )

        output = result.result or ""

        return (
            points,
            score,
            output,
        )

    finally:
        try:
            if sandbox is not None:
                daytona.delete(
                    sandbox
                )
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
        / f"iteration_{result.iteration:02d}"
    )

    circuit_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        circuit_directory
        / CIRCUIT_FILE_NAME
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

    progress = pyqtSignal(
        int,
        int,
    )

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

    points_ready = pyqtSignal(
        list,
    )

    score_updated = pyqtSignal(
        float,
        str,
    )

    terminal_output = pyqtSignal(
        str,
    )

    circuit_finished = pyqtSignal(
        int,
        float,
        bool,
        int,
    )

    decision_ready = pyqtSignal(
        int,
        int,
        str,
        str,
        str,
        str,
        dict,
        dict,
    )

    error = pyqtSignal(
        str,
    )

    completed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.running = True
        self.best_score = 0.0
        self.total_completed = 0
        self.current_best_score = -1.0

    def stop(
        self,
    ) -> None:
        self.running = False

    def run(
        self,
    ) -> None:

        try:
            prepare_output_directory()

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

        self.current_best_score = -1.0

        self.circuit_started.emit(
            circuit_id,
            TOTAL_CIRCUITS,
            0,
        )

        self.terminal_output.emit(
            (
                "\n"
                "==================================================\n"
                f"CIRCUIT {circuit_id:03d}\n"
                "=================================================="
            )
        )

        parameters, decision = (
            generate_initial_parameters(
                circuit_id
            )
        )

        self.decision_ready.emit(
            circuit_id,
            1,
            decision.decision_type,
            decision.summary,
            decision.parameter_reasoning,
            decision.direction_reasoning,
            decision.previous_parameters
            or {},
            decision.new_parameters,
        )

        self.terminal_output.emit(
            (
                f"[Circuit {circuit_id:03d}] "
                f"Initial style={parameters.layout_style} | "
                f"direction={parameters.direction:+d}"
            )
        )

        best_result: CircuitResult | None = None
        previous_score: CircuitScore | None = None

        # ====================================================================
        # EXACTLY 20 ITERATIONS
        # ====================================================================

        for iteration in range(
            1,
            MAX_ITERATIONS_PER_CIRCUIT + 1,
        ):

            if not self.running:
                return

            # ----------------------------------------------------------------
            # Iteration 1 uses initial parameters.
            #
            # Iterations 2..20 ask the agent to revise the immediately
            # preceding parameter set using the preceding score.
            # ----------------------------------------------------------------

            if (
                iteration > 1
                and previous_score is not None
            ):

                self.terminal_output.emit(
                    (
                        f"[Circuit {circuit_id:03d}] "
                        f"Iteration {iteration:02d}: "
                        "asking agent to revise parameters..."
                    )
                )

                parameters, decision = (
                    improve_parameters(
                        circuit_id,
                        iteration,
                        parameters,
                        previous_score,
                    )
                )

                self.decision_ready.emit(
                    circuit_id,
                    iteration,
                    decision.decision_type,
                    decision.summary,
                    decision.parameter_reasoning,
                    decision.direction_reasoning,
                    decision.previous_parameters
                    or {},
                    decision.new_parameters,
                )

            self.iteration_started.emit(
                circuit_id,
                iteration,
                MAX_ITERATIONS_PER_CIRCUIT,
            )

            self.terminal_output.emit(
                (
                    f"[Circuit {circuit_id:03d}] "
                    f"Iteration {iteration:02d}/"
                    f"{MAX_ITERATIONS_PER_CIRCUIT:02d} | "
                    f"style={parameters.layout_style} | "
                    f"direction={parameters.direction:+d}"
                )
            )

            # ----------------------------------------------------------------
            # Generate geometry.
            # ----------------------------------------------------------------

            seed = (
                circuit_id
                * 100000
                + iteration
            )

            points, score, output = (
                run_in_sandbox(
                    parameters,
                    seed,
                )
            )

            # ----------------------------------------------------------------
            # SEND ACTUAL GEOMETRY TO CANVAS.
            # ----------------------------------------------------------------

            self.points_ready.emit(
                [
                    [
                        float(point[0]),
                        float(point[1]),
                    ]
                    for point in points
                ]
            )

            self.terminal_output.emit(
                (
                    f"[Circuit {circuit_id:03d}] "
                    f"Iteration {iteration:02d}: "
                    f"canvas updated with "
                    f"{len(points)} points."
                )
            )

            self.terminal_output.emit(
                (
                    f"[Circuit {circuit_id:03d}] "
                    f"Daytona verification output: "
                    f"{len(output)} bytes"
                )
            )

            # ----------------------------------------------------------------
            # Score.
            # ----------------------------------------------------------------

            previous_score = score

            self.score_updated.emit(
                score.total,
                score.reason,
            )

            self.terminal_output.emit(
                (
                    f"[Circuit {circuit_id:03d}] "
                    f"Iteration {iteration:02d} | "
                    f"score={score.total:.2f} | "
                    f"validity={score.validity:.2f} | "
                    f"closure={score.closure:.2f} | "
                    f"separation={score.separation:.2f} | "
                    f"smoothness={score.smoothness:.2f} | "
                    f"complexity={score.complexity:.2f}"
                )
            )

            accepted = (
                score.total >= ACCEPTANCE_SCORE
                and score.validity >= 100.0
                and score.separation >= 85.0
                and score.closure >= 85.0
            )

            result = CircuitResult(
                circuit_id=circuit_id,
                iteration=iteration,
                parameters=asdict(
                    parameters
                ),
                points=[
                    [
                        round(
                            point[0],
                            3,
                        ),
                        round(
                            point[1],
                            3,
                        ),
                    ]
                    for point in points
                ],
                score=asdict(
                    score
                ),
                accepted=accepted,
                agent_decision=asdict(
                    decision
                ),
            )

            # ----------------------------------------------------------------
            # Every iteration is saved.
            # ----------------------------------------------------------------

            save_circuit(
                result
            )

            # ----------------------------------------------------------------
            # Track the best iteration.
            # ----------------------------------------------------------------

            if (
                best_result is None
                or score.total
                > float(
                    best_result.score[
                        "total"
                    ]
                )
            ):

                best_result = result
                self.current_best_score = (
                    score.total
                )

                self.terminal_output.emit(
                    (
                        f"[Circuit {circuit_id:03d}] "
                        f"NEW BEST = iteration "
                        f"{iteration:02d} | "
                        f"score={score.total:.2f}"
                    )
                )

            else:

                self.terminal_output.emit(
                    (
                        f"[Circuit {circuit_id:03d}] "
                        f"Best remains iteration "
                        f"{best_result.iteration:02d} | "
                        f"score="
                        f"{float(best_result.score['total']):.2f}"
                    )
                )

            # ----------------------------------------------------------------
            # CRITICAL:
            #
            # An accepted result does NOT stop the loop.
            #
            # All 20 iterations are always evaluated. The highest score wins.
            # ----------------------------------------------------------------

            if iteration < MAX_ITERATIONS_PER_CIRCUIT:

                self.terminal_output.emit(
                    (
                        f"[Circuit {circuit_id:03d}] "
                        f"Continuing to iteration "
                        f"{iteration + 1:02d}/"
                        f"{MAX_ITERATIONS_PER_CIRCUIT:02d}."
                    )
                )

        # ====================================================================
        # AFTER ALL 20 ITERATIONS
        # ====================================================================

        if best_result is None:

            self.circuit_finished.emit(
                circuit_id,
                0.0,
                False,
                0,
            )

            return

        best_path = save_best_circuit(
            best_result
        )

        best_total = float(
            best_result.score[
                "total"
            ]
        )

        self.best_score = max(
            self.best_score,
            best_total,
        )

        self.terminal_output.emit(
            (
                "\n"
                f"[Circuit {circuit_id:03d}] "
                "ALL 20 ITERATIONS COMPLETE.\n"
                f"[Circuit {circuit_id:03d}] "
                f"BEST ITERATION = "
                f"{best_result.iteration:02d}\n"
                f"[Circuit {circuit_id:03d}] "
                f"BEST SCORE = "
                f"{best_total:.2f}\n"
                f"[Circuit {circuit_id:03d}] "
                f"SAVED = {best_path}\n"
            )
        )

        self.circuit_finished.emit(
            circuit_id,
            best_total,
            best_result.accepted,
            best_result.iteration,
        )


# ============================================================================
# CIRCUIT CANVAS
# ============================================================================

class CircuitCanvas(QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.points: list[
            QPointF
        ] = []

        self.zoom = 1.0

        self.pan_x = 0.0
        self.pan_y = 0.0

        self.dragging = False

        self.last_mouse_position = (
            QPointF()
        )

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

        self.fit_to_canvas()

    def clear(
        self,
    ) -> None:

        self.points = []

        self.update()

    def fit_to_canvas(
        self,
    ) -> None:

        if not self.points:
            return

        xs = [
            point.x()
            for point in self.points
        ]

        ys = [
            point.y()
            for point in self.points
        ]

        min_x = min(xs)
        max_x = max(xs)

        min_y = min(ys)
        max_y = max(ys)

        width = (
            max_x
            - min_x
        )

        height = (
            max_y
            - min_y
        )

        if (
            width <= 0.0
            or height <= 0.0
        ):
            return

        available_width = (
            self.width()
            * 0.82
        )

        available_height = (
            self.height()
            * 0.82
        )

        scale_x = (
            available_width
            / width
        )

        scale_y = (
            available_height
            / height
        )

        self.zoom = min(
            scale_x,
            scale_y,
        )

        self.pan_x = (
            self.width()
            / 2.0
            - (
                (min_x + max_x)
                / 2.0
            )
            * self.zoom
        )

        self.pan_y = (
            self.height()
            / 2.0
            - (
                (min_y + max_y)
                / 2.0
            )
            * self.zoom
        )

        self.update()

    def transform_point(
        self,
        point: QPointF,
    ) -> QPointF:

        return QPointF(
            point.x()
            * self.zoom
            + self.pan_x,
            point.y()
            * self.zoom
            + self.pan_y,
        )

    def paintEvent(
        self,
        event,
    ) -> None:

        del event

        painter = QPainter(
            self
        )

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        painter.fillRect(
            self.rect(),
            QBrush(
                Qt.GlobalColor.black
            ),
        )

        if len(self.points) < 2:

            painter.setPen(
                QPen(
                    Qt.GlobalColor.gray
                )
            )

            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for circuit geometry...",
            )

            return

        transformed = [
            self.transform_point(
                point
            )
            for point in self.points
        ]

        track_pen = QPen(
            Qt.GlobalColor.white,
            max(
                2,
                int(
                    TRACK_WIDTH
                    * self.zoom
                ),
            ),
        )

        track_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )

        track_pen.setJoinStyle(
            Qt.PenJoinStyle.RoundJoin
        )

        painter.setPen(
            track_pen
        )

        for index in range(
            len(transformed)
        ):

            first = transformed[
                index
            ]

            second = transformed[
                (index + 1)
                % len(transformed)
            ]

            painter.drawLine(
                first,
                second,
            )

        centre_pen = QPen(
            Qt.GlobalColor.darkGray,
            max(
                1,
                int(
                    3
                    * self.zoom
                ),
            ),
        )

        centre_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )

        centre_pen.setJoinStyle(
            Qt.PenJoinStyle.RoundJoin
        )

        painter.setPen(
            centre_pen
        )

        for index in range(
            len(transformed)
        ):

            first = transformed[
                index
            ]

            second = transformed[
                (index + 1)
                % len(transformed)
            ]

            painter.drawLine(
                first,
                second,
            )

        start_point = (
            transformed[0]
        )

        painter.setBrush(
            QBrush(
                Qt.GlobalColor.red
            )
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
            QPen(
                Qt.GlobalColor.green,
                2,
            )
        )

        painter.drawText(
            start_point
            + QPointF(
                12.0,
                -12.0,
            ),
            "START",
        )

    def wheelEvent(
        self,
        event: QWheelEvent,
    ) -> None:

        delta = (
            event.angleDelta()
            .y()
        )

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
        event: QMouseEvent,
    ) -> None:

        if (
            event.button()
            == Qt.MouseButton.LeftButton
        ):

            self.dragging = True

            self.last_mouse_position = (
                event.position()
            )

    def mouseMoveEvent(
        self,
        event: QMouseEvent,
    ) -> None:

        if not self.dragging:
            return

        current = event.position()

        delta = (
            current
            - self.last_mouse_position
        )

        self.pan_x += delta.x()
        self.pan_y += delta.y()

        self.last_mouse_position = (
            current
        )

        self.update()

    def mouseReleaseEvent(
        self,
        event: QMouseEvent,
    ) -> None:

        if (
            event.button()
            == Qt.MouseButton.LeftButton
        ):

            self.dragging = False


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.worker: (
            GenerationWorker | None
        ) = None

        self.current_circuit = 0

        self.feedback_data: dict[
            tuple[int, int],
            dict[str, Any],
        ] = {}

        self.setup_ui()

    def setup_ui(
        self,
    ) -> None:

        self.setWindowTitle(
            "Procedural Circuit AI Lab"
        )

        self.resize(
            1600,
            1000,
        )

        central = QWidget()

        self.setCentralWidget(
            central
        )

        layout = QHBoxLayout(
            central
        )

        self.canvas = CircuitCanvas()

        layout.addWidget(
            self.canvas,
            stretch=4,
        )

        side = QVBoxLayout()

        layout.addLayout(
            side,
            stretch=2,
        )

        title = QLabel(
            "Procedural Circuit AI"
        )

        title_font = QFont()

        title_font.setPointSize(
            18
        )

        title_font.setBold(
            True
        )

        title.setFont(
            title_font
        )

        side.addWidget(
            title
        )

        self.circuit_label = QLabel(
            "Circuit: 0 / 20"
        )

        self.iteration_label = QLabel(
            "Iteration: - / 20"
        )

        self.score_label = QLabel(
            "Score: -"
        )

        self.best_label = QLabel(
            "Global best score: -"
        )

        self.style_label = QLabel(
            "Style: -"
        )

        side.addWidget(
            self.circuit_label
        )

        side.addWidget(
            self.iteration_label
        )

        side.addWidget(
            self.score_label
        )

        side.addWidget(
            self.best_label
        )

        side.addWidget(
            self.style_label
        )

        self.overall_progress = (
            QProgressBar()
        )

        self.overall_progress.setRange(
            0,
            TOTAL_CIRCUITS,
        )

        self.overall_progress.setValue(
            0
        )

        self.overall_progress.setFormat(
            "Overall circuits: %p%"
        )

        side.addWidget(
            self.overall_progress
        )

        self.iteration_progress = (
            QProgressBar()
        )

        self.iteration_progress.setRange(
            0,
            MAX_ITERATIONS_PER_CIRCUIT,
        )

        self.iteration_progress.setValue(
            0
        )

        self.iteration_progress.setFormat(
            "Current circuit: iteration %v / %m"
        )

        side.addWidget(
            self.iteration_progress
        )

        # --------------------------------------------------------------------
        # History controls.
        # --------------------------------------------------------------------

        history_box = QGroupBox(
            "Inspect circuit / iteration"
        )

        history_layout = QFormLayout(
            history_box
        )

        self.circuit_selector = (
            QComboBox()
        )

        self.iteration_selector = (
            QComboBox()
        )

        self.circuit_selector.addItem(
            "No circuits yet"
        )

        self.iteration_selector.addItem(
            "No iterations yet"
        )

        history_layout.addRow(
            "Circuit:",
            self.circuit_selector,
        )

        history_layout.addRow(
            "Iteration:",
            self.iteration_selector,
        )

        side.addWidget(
            history_box
        )

        # --------------------------------------------------------------------
        # Parameters.
        # --------------------------------------------------------------------

        parameter_box = QGroupBox(
            "Starting / selected parameters"
        )

        parameter_layout = QVBoxLayout(
            parameter_box
        )

        self.parameters_view = (
            QTextEdit()
        )

        self.parameters_view.setReadOnly(
            True
        )

        self.parameters_view.setFont(
            QFont(
                "Consolas",
                8,
            )
        )

        parameter_layout.addWidget(
            self.parameters_view
        )

        side.addWidget(
            parameter_box,
            stretch=1,
        )

        # --------------------------------------------------------------------
        # Agent feedback.
        # --------------------------------------------------------------------

        feedback_box = QGroupBox(
            "Agent feedback"
        )

        feedback_layout = QVBoxLayout(
            feedback_box
        )

        self.feedback_view = (
            QTextEdit()
        )

        self.feedback_view.setReadOnly(
            True
        )

        self.feedback_view.setFont(
            QFont(
                "Consolas",
                9,
            )
        )

        feedback_layout.addWidget(
            self.feedback_view
        )

        side.addWidget(
            feedback_box,
            stretch=2,
        )

        # --------------------------------------------------------------------
        # Buttons.
        # --------------------------------------------------------------------

        buttons = QHBoxLayout()

        self.start_button = (
            QPushButton(
                "Start generation"
            )
        )

        self.stop_button = (
            QPushButton(
                "Stop generation"
            )
        )

        self.stop_button.setEnabled(
            False
        )

        buttons.addWidget(
            self.start_button
        )

        buttons.addWidget(
            self.stop_button
        )

        side.addLayout(
            buttons
        )

        self.fit_button = (
            QPushButton(
                "Fit circuit to canvas"
            )
        )

        side.addWidget(
            self.fit_button
        )

        # --------------------------------------------------------------------
        # Terminal.
        # --------------------------------------------------------------------

        side.addWidget(
            QLabel(
                "Daytona / Agent Output"
            )
        )

        self.terminal = (
            QTextEdit()
        )

        self.terminal.setReadOnly(
            True
        )

        self.terminal.setFont(
            QFont(
                "Consolas",
                8,
            )
        )

        side.addWidget(
            self.terminal,
            stretch=2,
        )

        # --------------------------------------------------------------------
        # Connections.
        # --------------------------------------------------------------------

        self.start_button.clicked.connect(
            self.start_generation
        )

        self.stop_button.clicked.connect(
            self.stop_generation
        )

        self.fit_button.clicked.connect(
            self.canvas.fit_to_canvas
        )

        self.circuit_selector.currentIndexChanged.connect(
            self.refresh_iteration_selector
        )

        self.iteration_selector.currentIndexChanged.connect(
            self.display_selected_feedback
        )

    # ------------------------------------------------------------------------
    # Start.
    # ------------------------------------------------------------------------

    def start_generation(
        self,
    ) -> None:

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            return

        self.terminal.clear()

        self.canvas.clear()

        self.feedback_data.clear()

        self.circuit_selector.blockSignals(
            True
        )

        self.iteration_selector.blockSignals(
            True
        )

        self.circuit_selector.clear()

        self.iteration_selector.clear()

        self.circuit_selector.blockSignals(
            False
        )

        self.iteration_selector.blockSignals(
            False
        )

        self.overall_progress.setValue(
            0
        )

        self.iteration_progress.setValue(
            0
        )

        self.circuit_label.setText(
            f"Circuit: 0 / {TOTAL_CIRCUITS}"
        )

        self.iteration_label.setText(
            (
                "Iteration: - / "
                f"{MAX_ITERATIONS_PER_CIRCUIT}"
            )
        )

        self.score_label.setText(
            "Score: -"
        )

        self.best_label.setText(
            "Global best score: -"
        )

        self.style_label.setText(
            "Style: -"
        )

        self.parameters_view.clear()

        self.feedback_view.clear()

        self.start_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            True
        )

        self.worker = (
            GenerationWorker()
        )

        self.worker.progress.connect(
            self.on_progress
        )

        self.worker.circuit_started.connect(
            self.on_circuit_started
        )

        self.worker.iteration_started.connect(
            self.on_iteration_started
        )

        self.worker.points_ready.connect(
            self.on_points_ready
        )

        self.worker.score_updated.connect(
            self.on_score_updated
        )

        self.worker.terminal_output.connect(
            self.on_terminal_output
        )

        self.worker.decision_ready.connect(
            self.on_decision_ready
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

        self.worker.start()

    # ------------------------------------------------------------------------
    # Stop.
    # ------------------------------------------------------------------------

    def stop_generation(
        self,
    ) -> None:

        if self.worker is None:
            return

        self.worker.stop()

        self.stop_button.setEnabled(
            False
        )

        self.terminal.append(
            (
                "Stopping after the current "
                "agent / sandbox operation..."
            )
        )

    # ------------------------------------------------------------------------
    # Progress.
    # ------------------------------------------------------------------------

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
            (
                f"Circuit: "
                f"{self.current_circuit} / "
                f"{total}"
            )
        )

    def on_circuit_started(
        self,
        circuit_id: int,
        total: int,
        iteration: int,
    ) -> None:

        del iteration

        self.current_circuit = (
            circuit_id
        )

        self.circuit_label.setText(
            (
                f"Circuit: "
                f"{circuit_id} / "
                f"{total}"
            )
        )

        self.score_label.setText(
            "Score: generating..."
        )

        self.style_label.setText(
            "Style: generating..."
        )

        self.refresh_circuit_selector()

    def on_iteration_started(
        self,
        circuit_id: int,
        iteration: int,
        maximum: int,
    ) -> None:

        self.current_circuit = (
            circuit_id
        )

        self.circuit_label.setText(
            (
                f"Circuit: "
                f"{circuit_id} / "
                f"{TOTAL_CIRCUITS}"
            )
        )

        self.iteration_label.setText(
            (
                f"Iteration: "
                f"{iteration} / "
                f"{maximum}"
            )
        )

        self.iteration_progress.setMaximum(
            maximum
        )

        self.iteration_progress.setValue(
            iteration
        )

    def on_points_ready(
        self,
        points: list,
    ) -> None:

        self.canvas.set_points(
            points
        )

    def on_score_updated(
        self,
        score: float,
        reason: str,
    ) -> None:

        self.score_label.setText(
            (
                f"Score: "
                f"{score:.2f} / 100"
            )
        )

        self.terminal.append(
            (
                f"Score={score:.2f} | "
                f"{reason}"
            )
        )

    def on_terminal_output(
        self,
        text: str,
    ) -> None:

        self.terminal.append(
            text
        )

        scrollbar = (
            self.terminal.verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )

    # ------------------------------------------------------------------------
    # Agent feedback.
    # ------------------------------------------------------------------------

    def on_decision_ready(
        self,
        circuit_id: int,
        iteration: int,
        decision_type: str,
        summary: str,
        parameter_reasoning: str,
        direction_reasoning: str,
        previous_parameters: dict,
        new_parameters: dict,
    ) -> None:

        self.feedback_data[
            (
                circuit_id,
                iteration,
            )
        ] = {
            "decision_type": decision_type,
            "summary": summary,
            "parameter_reasoning": (
                parameter_reasoning
            ),
            "direction_reasoning": (
                direction_reasoning
            ),
            "previous_parameters": (
                previous_parameters
            ),
            "new_parameters": (
                new_parameters
            ),
        }

        self.refresh_circuit_selector()

        self.set_selector_value(
            self.circuit_selector,
            circuit_id,
        )

        self.refresh_iteration_selector()

        self.set_selector_value(
            self.iteration_selector,
            iteration,
        )

        self.display_selected_feedback()

        self.terminal.append(
            (
                f"[Agent] Circuit "
                f"{circuit_id:03d} | "
                f"Iteration {iteration:02d} | "
                f"{summary}"
            )
        )

    def display_selected_feedback(
        self,
    ) -> None:

        circuit_data = (
            self.circuit_selector.currentData()
        )

        iteration_data = (
            self.iteration_selector.currentData()
        )

        if (
            circuit_data is None
            or iteration_data is None
        ):
            return

        circuit_id = int(
            circuit_data
        )

        iteration = int(
            iteration_data
        )

        data = self.feedback_data.get(
            (
                circuit_id,
                iteration,
            )
        )

        if data is None:
            self.load_saved_result(
                circuit_id,
                iteration,
            )
            return

        previous_parameters = (
            data.get(
                "previous_parameters",
                {},
            )
        )

        new_parameters = (
            data.get(
                "new_parameters",
                {},
            )
        )

        feedback_text = (
            f"DECISION TYPE\n"
            f"{data['decision_type']}\n\n"
            f"SUMMARY\n"
            f"{data['summary']}\n\n"
            f"PARAMETER REASONING\n"
            f"{data['parameter_reasoning']}\n\n"
            f"DIRECTION REASONING\n"
            f"{data['direction_reasoning']}\n\n"
            f"PREVIOUS PARAMETERS\n"
            f"{json.dumps(previous_parameters, indent=2)}\n\n"
            f"NEW PARAMETERS\n"
            f"{json.dumps(new_parameters, indent=2)}"
        )

        self.feedback_view.setPlainText(
            feedback_text
        )

        self.parameters_view.setPlainText(
            json.dumps(
                new_parameters,
                indent=2,
            )
        )

        self.style_label.setText(
            (
                "Style: "
                f"{new_parameters.get('layout_style', '-')}"
                " | Direction: "
                f"{new_parameters.get('direction', '-')}"
            )
        )

        self.load_saved_result(
            circuit_id,
            iteration,
            update_feedback=False,
        )

    # ------------------------------------------------------------------------
    # Saved result loading.
    # ------------------------------------------------------------------------

    def load_saved_result(
        self,
        circuit_id: int,
        iteration: int,
        update_feedback: bool = True,
    ) -> None:

        path = (
            OUTPUT_DIRECTORY
            / f"circuit_{circuit_id:03d}"
            / f"iteration_{iteration:02d}"
            / CIRCUIT_FILE_NAME
        )

        if not path.exists():
            return

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(
                    file
                )

            parameters = data.get(
                "parameters",
                {},
            )

            score = data.get(
                "score",
                {},
            )

            points = data.get(
                "points",
                [],
            )

            self.parameters_view.setPlainText(
                json.dumps(
                    parameters,
                    indent=2,
                )
            )

            self.score_label.setText(
                (
                    "Score: "
                    f"{float(score.get('total', 0.0)):.2f}"
                    " / 100"
                )
            )

            direction = parameters.get(
                "direction",
                "-",
            )

            self.style_label.setText(
                (
                    "Style: "
                    f"{parameters.get('layout_style', '-')}"
                    " | Direction: "
                    f"{direction:+d}"
                    if isinstance(
                        direction,
                        int,
                    )
                    else (
                        "Style: "
                        f"{parameters.get('layout_style', '-')}"
                        " | Direction: "
                        f"{direction}"
                    )
                )
            )

            if points:
                self.canvas.set_points(
                    points
                )

            if update_feedback:

                decision = data.get(
                    "agent_decision",
                    {},
                )

                if decision:

                    self.feedback_view.setPlainText(
                        (
                            f"DECISION TYPE\n"
                            f"{decision.get('decision_type', '-')}\n\n"
                            f"SUMMARY\n"
                            f"{decision.get('summary', '-')}\n\n"
                            f"PARAMETER REASONING\n"
                            f"{decision.get('parameter_reasoning', '-')}\n\n"
                            f"DIRECTION REASONING\n"
                            f"{decision.get('direction_reasoning', '-')}"
                        )
                    )

        except Exception as exc:

            self.terminal.append(
                (
                    f"Could not load saved result "
                    f"for circuit {circuit_id:03d}, "
                    f"iteration {iteration:02d}: "
                    f"{type(exc).__name__}: {exc}"
                )
            )

    # ------------------------------------------------------------------------
    # Circuit history selectors.
    # ------------------------------------------------------------------------

    def refresh_circuit_selector(
        self,
    ) -> None:

        circuit_ids = sorted(
            {
                circuit_id
                for circuit_id, _
                in self.feedback_data.keys()
            }
        )

        if not circuit_ids:
            return

        current = (
            self.circuit_selector.currentData()
        )

        self.circuit_selector.blockSignals(
            True
        )

        self.circuit_selector.clear()

        for circuit_id in circuit_ids:

            self.circuit_selector.addItem(
                f"Circuit {circuit_id:03d}",
                circuit_id,
            )

        self.circuit_selector.blockSignals(
            False
        )

        if current in circuit_ids:

            self.set_selector_value(
                self.circuit_selector,
                int(current),
            )

        else:

            self.circuit_selector.setCurrentIndex(
                len(circuit_ids) - 1
            )

    def refresh_iteration_selector(
        self,
    ) -> None:

        circuit_data = (
            self.circuit_selector.currentData()
        )

        if circuit_data is None:
            return

        circuit_id = int(
            circuit_data
        )

        iterations = sorted(
            iteration
            for current_circuit, iteration
            in self.feedback_data.keys()
            if current_circuit == circuit_id
        )

        if not iterations:
            return

        current_iteration = (
            self.iteration_selector.currentData()
        )

        self.iteration_selector.blockSignals(
            True
        )

        self.iteration_selector.clear()

        for iteration in iterations:

            self.iteration_selector.addItem(
                f"Iteration {iteration:02d}",
                iteration,
            )

        self.iteration_selector.blockSignals(
            False
        )

        if (
            current_iteration
            in iterations
        ):

            self.set_selector_value(
                self.iteration_selector,
                int(current_iteration),
            )

        else:

            self.iteration_selector.setCurrentIndex(
                len(iterations) - 1
            )

        self.display_selected_feedback()

    @staticmethod
    def set_selector_value(
        selector: QComboBox,
        value: int,
    ) -> None:

        index = selector.findData(
            value
        )

        if index >= 0:

            selector.setCurrentIndex(
                index
            )

    # ------------------------------------------------------------------------
    # Circuit finished.
    # ------------------------------------------------------------------------

    def on_circuit_finished(
        self,
        circuit_id: int,
        score: float,
        accepted: bool,
        best_iteration: int,
    ) -> None:

        status = (
            "BEST ITERATION ACCEPTED"
            if accepted
            else "BEST RESULT SAVED"
        )

        self.terminal.append(
            (
                f"[Circuit {circuit_id:03d}] "
                f"{status} | "
                f"best iteration="
                f"{best_iteration:02d} | "
                f"score={score:.2f}"
            )
        )

        if (
            self.worker is not None
            and self.worker.best_score > 0.0
        ):

            self.best_label.setText(
                (
                    "Global best score: "
                    f"{self.worker.best_score:.2f}"
                )
            )

    # ------------------------------------------------------------------------
    # Errors.
    # ------------------------------------------------------------------------

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

        self.start_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

    # ------------------------------------------------------------------------
    # Completion.
    # ------------------------------------------------------------------------

    def on_completed(
        self,
    ) -> None:

        self.start_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

        self.circuit_label.setText(
            (
                f"Circuit: "
                f"{TOTAL_CIRCUITS} / "
                f"{TOTAL_CIRCUITS}"
            )
        )

        self.terminal.append(
            (
                "\n"
                "==================================================\n"
                f"COMPLETED {TOTAL_CIRCUITS} CIRCUITS\n"
                f"{MAX_ITERATIONS_PER_CIRCUIT} ITERATIONS PER CIRCUIT\n"
                f"{TOTAL_CIRCUITS * MAX_ITERATIONS_PER_CIRCUIT} "
                "TOTAL ITERATION EVALUATIONS\n"
                "=================================================="
            )
        )

    # ------------------------------------------------------------------------
    # Window close.
    # ------------------------------------------------------------------------

    def closeEvent(
        self,
        event,
    ) -> None:

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            self.worker.stop()

            self.worker.wait(
                5000
            )

        event.accept()


# ============================================================================
# APPLICATION
# ============================================================================

def main() -> None:

    print(
        "=== PROCEDURAL CIRCUIT AI ==="
    )

    print(
        f"Target circuits: {TOTAL_CIRCUITS}"
    )

    print(
        "Iterations per circuit:",
        MAX_ITERATIONS_PER_CIRCUIT,
    )

    print(
        "Total evaluations:",
        TOTAL_CIRCUITS
        * MAX_ITERATIONS_PER_CIRCUIT,
    )

    print(
        "Acceptance score:",
        ACCEPTANCE_SCORE,
    )

    print(
        "Geometry: section-based closed polygon "
        "with rounded corners"
    )

    print(
        "Daytona: configured"
    )

    print(
        "Groq: configured"
    )

    prepare_output_directory()

    application = QApplication(
        sys.argv
    )

    window = MainWindow()

    window.show()

    sys.exit(
        application.exec()
    )


if __name__ == "__main__":
    main()