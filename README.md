
# Procedural Circuit AI Lab

## Overview

Procedural Circuit AI Lab is an AI-assisted racing circuit generator built with Python, Groq, Daytona, and PyQt6.

The system uses an AI agent to choose high-level circuit design parameters. A procedural geometry engine turns those parameters into a complete racing circuit. The circuit is then generated, validated, and scored inside a Daytona sandbox.

The agent uses the result to change its parameters and tries again.

The current configuration is:

* 20 circuits
* 20 iterations per circuit
* 400 total evaluations
* 1 best result saved for each circuit

The purpose of the project is to demonstrate an AI system that can repeatedly design, evaluate, and improve a result instead of producing only one output.

## How It Works

The system follows the same loop for every circuit:

```text
AI selects parameters
        ↓
Daytona generates the circuit
        ↓
Daytona validates and scores it
        ↓
Result is shown in the GUI
        ↓
AI reviews the result
        ↓
AI changes the parameters
        ↓
Next iteration
```

Iteration 1 uses the initial design created by the AI.

Iterations 2–20 use the previous iteration's parameters and score. The AI explains what it wants to change and why.

All 20 iterations are evaluated, even when one reaches the acceptance score.

After iteration 20, the highest-scoring iteration becomes the best result for that circuit.

## Daytona

Daytona is a central part of the evaluation process.

For each circuit, the application creates a Daytona sandbox and loads a self-contained copy of the procedural geometry and scoring engine into it.

The same sandbox is reused for all 20 iterations of that circuit.

Inside Daytona, each iteration performs:

1. Circuit geometry generation
2. Geometry validation
3. Self-intersection detection
4. Track separation checks
5. Closure checks
6. Smoothness calculation
7. Complexity calculation
8. Scale calculation
9. Variety calculation
10. Final score calculation

Daytona returns the generated coordinates and evaluation results to the main application.

The GUI then displays the circuit produced by the Daytona execution.

This means Daytona is not being used only as a demonstration or a separate test. It is part of the actual generation and evaluation loop that determines the feedback used by the agent.

A new sandbox is created for each circuit and removed when that circuit has completed its 20 iterations.

## AI Agent

The AI is responsible for high-level design decisions rather than drawing individual coordinates.

It controls parameters including:

* Width
* Height
* Number of corners
* Number of straights
* Minimum corner radius
* Maximum corner radius
* Shape variation
* Asymmetry
* Hairpin probability
* Chicane probability
* Complexity
* Minimum spacing
* Driving direction
* Layout style

The available layout styles are:

* Fast
* Balanced
* Technical
* Hairpin
* Flowing
* Asymmetric

The AI also provides written reasoning for its decisions.

For the initial iteration, it explains why the starting parameters were chosen.

For later iterations, it explains:

* What changed
* Why it changed
* What effect the change is expected to have
* Whether the circuit should become faster, slower, more technical, more flowing, or more varied
* Whether the driving direction should change
* Why moving away from the previous design is justified

The application stores this information for every circuit and iteration.

## Circuit Generation

The geometry engine converts the AI's parameters into a closed racing circuit.

The generator creates a number of connected sections and then rounds the corners using curves.

It can produce combinations of:

* Long straights
* Short straights
* Fast corners
* Medium-speed corners
* Tight corners
* Hairpins
* Chicanes
* Linked corners
* Flowing sections
* Technical sections
* Doglegs
* Asymmetric layouts

The generator is deliberately designed to create different circuit structures rather than repeatedly producing the same oval shape with small changes.

The circuit is represented as a sequence of two-dimensional points forming the track centreline.

## Circuit Scoring

Each circuit receives a score from 0 to 100.

The score considers:

| Measure    | Purpose                                                    |
| ---------- | ---------------------------------------------------------- |
| Closure    | Checks whether the circuit forms a proper loop             |
| Separation | Checks the distance between different parts of the circuit |
| Smoothness | Measures how smoothly the track changes direction          |
| Complexity | Measures the amount of direction change                    |
| Scale      | Checks the overall size of the layout                      |
| Variety    | Measures variation in the generated sections               |
| Validity   | Checks for self-intersections                              |

The score is calculated inside Daytona.

The current acceptance requirements are:

```text
Score >= 82.0
Validity >= 100.0
Separation >= 85.0
Closure >= 85.0
```

Acceptance does not end the optimisation process. The program always completes all 20 iterations.

## Best Result

Every iteration is saved.

After all 20 iterations have been completed, the application compares their total scores.

The iteration with the highest score is saved as the best result for that circuit.

For example:

```text
Iteration 01    74.20
Iteration 02    78.51
Iteration 03    81.32
Iteration 04    84.10
...
Iteration 20    82.75
```

Iteration 4 would become the best result because it has the highest score.

The best result is stored separately from the complete iteration history.

## Output

A complete run creates:

```text
generated_circuits/
├── circuit_001/
│   ├── iteration_01/
│   │   └── circuit.json
│   ├── iteration_02/
│   │   └── circuit.json
│   ├── ...
│   └── iteration_20/
│       └── circuit.json
├── circuit_002/
│   └── ...
├── ...
├── circuit_020/
│   └── ...
└── best/
    ├── circuit_001.json
    ├── circuit_002.json
    ├── ...
    └── circuit_020.json
```

A new generation run clears the previous `generated_circuits` directory before starting.

Each saved iteration contains:

* Circuit number
* Iteration number
* Parameters
* Generated geometry
* Score
* Acceptance status
* AI decision
* AI reasoning

## Graphical Interface

The PyQt6 interface allows the generation process to be monitored while it is running.

The interface displays:

* Current circuit
* Current iteration
* Current score
* Global best score
* Layout style
* Driving direction
* Overall progress
* Progress through the current circuit
* Current circuit geometry
* Selected parameters
* AI feedback
* Daytona and agent output

The canvas updates after every iteration.

The interface also contains two dropdowns for inspecting previous results.

The Circuit dropdown selects a generated circuit.

The Iteration dropdown selects an iteration from that circuit.

Selecting an iteration displays its saved geometry, parameters, score, and AI feedback.

## Circuit Terminology

The project uses common racing terms.

### Straight

A section where the track travels mostly in one direction.

Long straights allow cars to accelerate and can create overtaking opportunities.

### Fast Corner

A gentle corner that can be taken at relatively high speed.

### Medium-Speed Corner

A corner that requires some braking and steering but is not extremely tight.

### Tight Corner

A sharper corner that requires a larger reduction in speed.

### Hairpin

A very tight corner that turns the circuit back towards the direction from which the car arrived.

### Chicane

A group of corners close together where the direction changes more than once.

### Esses

A series of connected corners that generally forms an S-shaped section.

### Technical Section

A section containing several corners or direction changes close together.

### Flowing Section

A section with longer, smoother changes of direction that allows more speed to be maintained.

### Dogleg

A small change in direction within a section that is otherwise mostly straight.

### Asymmetric Circuit

A circuit where different parts of the lap have noticeably different characteristics rather than following a repeating pattern.

### Topology

The overall structure of the circuit and the order in which its sections are arranged.

Changing a hairpin to a chicane, for example, changes the topology.

### Centreline

The path representing the middle of the track.

The generator stores the circuit as a series of centreline coordinates.

### Corner Radius

A measurement of how tightly a corner bends.

A larger radius creates a gentler corner. A smaller radius creates a tighter corner.

### Curvature

A measurement of how sharply the track is changing direction.

### Apex

The point on the inside of a corner that the driver normally aims towards.

### Racing Line

The path a driver would ideally take through the circuit to achieve a fast lap.

The current system generates the circuit centreline but does not calculate an optimal racing line.

### Self-Intersection

When one part of the circuit crosses another part of the same circuit.

The validation system treats this as invalid.

### Track Separation

The distance between different parts of the circuit that are close together.

Two sections can be close enough to cause a problem even when they do not cross.

### Closed Circuit

A continuous loop that returns to its starting point.

### Driving Direction

The direction in which the circuit is driven.

The system supports both clockwise and anticlockwise layouts.

## Requirements

The project requires Python and the following packages:

```text
python-dotenv
openai
daytona
PyQt6
```

Install them with:

```bash
pip install python-dotenv openai daytona PyQt6
```

## Running the Project

The main application is:

```text
agent.py
```

To run it:

```bash
python agent.py
```

Before running the program, create a `.env` file in the project directory.

Add your own API keys:

```env
DAYTONA_API_KEY=your_daytona_api_key
GROQ_API_KEY=your_groq_api_key
```

The program requires both keys.

Do not commit your `.env` file to GitHub.

A safe example file can be created as:

```text
.env.example
```

with:

```env
DAYTONA_API_KEY=
GROQ_API_KEY=
```

## Project Structure

The main application is contained in:

```text
agent.py
```

The program contains:

* AI parameter generation
* AI optimisation
* Parameter validation
* Procedural geometry generation
* Geometry validation
* Circuit scoring
* Daytona sandbox execution
* Result saving
* Iteration tracking
* PyQt6 interface

## Current Limitations

The project currently focuses on two-dimensional circuit layout.

It does not currently model:

* Elevation
* Banking
* Camber
* Kerbs
* Run-off areas
* Barriers
* Pit lanes
* Pit buildings
* Terrain
* Detailed vehicle physics
* Weather
* Optimal racing lines
* Full vehicle-based lap-time simulation

The score is a project-specific mathematical measure of layout quality. It is not a real-world circuit safety or engineering assessment.

## Future Development

Possible future additions include:

* Vehicle-based lap simulation
* Racing-line generation
* Overtaking analysis
* Sector generation
* Track edges
* Kerbs
* Run-off areas
* Pit lanes
* Elevation and banking
* More detailed geometry validation
* Multi-objective optimisation
* Better visual comparison between iterations
* Export to game engines and other formats
