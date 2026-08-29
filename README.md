
# Procedural Circuit AI Lab

## Overview

Procedural Circuit AI Lab is a Python application for generating racing circuits using procedural geometry and an AI-assisted optimisation process.

The application does not ask the AI model to draw the circuit directly. Instead, the AI selects a set of design parameters, and a Python geometry system uses those parameters to build the circuit.

Each circuit is tested, scored, shown in the graphical interface, and then improved over multiple iterations.

The current configuration generates:

* 20 circuits
* 20 iterations for each circuit
* 400 total circuit evaluations

The highest-scoring iteration from each circuit is saved as that circuit's best result.

The application also records the AI's reasoning for its design choices so that the user can inspect why the parameters were selected and why they changed during optimisation.

## Main Components

The project has three main parts.

### AI agent

The AI agent is connected through the Groq API.

It is responsible for choosing and improving high-level circuit parameters.

The AI does not generate the circuit coordinates itself.

### Geometry generator

The geometry generator is written in Python.

It takes the parameters selected by the AI and creates a closed racing circuit.

The geometry system controls the actual shape of the track and prevents the AI from directly producing invalid coordinate data.

### Daytona verification

Daytona is used to run an additional geometry verification process inside a sandbox.

The generated circuit is sent to the sandbox, where the geometry is checked independently.

### Graphical interface

The interface is written using PyQt6.

It displays the current circuit while it is being generated and provides access to the parameters, scores, iteration history, and AI feedback.

## How the Generation Process Works

Each circuit goes through 20 iterations.

The first iteration starts with parameters selected by the AI.

The generated circuit is then evaluated.

For iterations 2 through 20, the AI receives the previous parameters and the previous evaluation and is asked to produce a revised design.

The process is therefore:

1. Select parameters.
2. Generate the circuit.
3. Verify the geometry.
4. Calculate the score.
5. Display the circuit.
6. Save the iteration.
7. Ask the AI to revise the parameters.
8. Repeat until 20 iterations have been completed.
9. Select the highest-scoring iteration.
10. Save that iteration as the best result for the circuit.

The program does not stop early when an iteration reaches the acceptance score. All 20 iterations are evaluated so that the best result can be selected from the complete set.

## Circuit Generation

The geometry generator creates circuits from a collection of ordered points.

It starts with a general shape and then changes the positions of its sections based on the selected parameters.

The system deliberately avoids generating every circuit as a simple oval with small variations.

Different parameter combinations can produce different arrangements of:

* Long straights
* Short straights
* Fast corners
* Medium-speed corners
* Tight corners
* Hairpins
* Chicanes
* Flowing sections
* Technical sections
* Doglegs
* Asymmetric sections

The corners are rounded so that the final track does not consist of sharp polygon edges.

The result is a continuous centreline representing the middle of the racing surface.

## Circuit Design Terms

The application uses several motorsport terms. The following definitions explain them in basic language.

### Straight

A straight is a section where the track travels mostly in one direction.

Long straights allow the car to build speed. They can also create opportunities for overtaking.

Short straights are often used to connect one corner to another.

### Corner

A corner is any part of the circuit where the track changes direction.

Different corners can be faster or slower depending on how sharply the track turns.

### Fast Corner

A fast corner is a gentle bend that can normally be taken at relatively high speed.

The driver does not need to slow the car as much as they would for a tight corner.

### Medium-Speed Corner

A medium-speed corner sits between a fast corner and a very tight corner.

The driver has to reduce speed and turn the car, but the corner is not extremely sharp.

### Tight Corner

A tight corner requires the car to slow down more significantly.

The track changes direction over a shorter distance.

The exit of a tight corner is also important because it determines how quickly the car can accelerate onto the next section.

### Hairpin

A hairpin is a very tight corner that turns the circuit back towards the direction from which the car arrived.

It is similar to making a U-turn.

Hairpins create a large change in speed and can be useful before a long straight because a car can gain a large amount of speed after the corner.

### Chicane

A chicane is a group of corners close together where the direction changes more than once.

For example, the car may turn left and then immediately turn right.

A chicane forces the driver to change direction quickly.

### Esses

An esses section is a series of connected corners that creates a general S-shaped path.

The driver has to keep changing direction rather than completing one isolated corner.

### Linked Corners

Linked corners are a series of corners where one corner affects the next.

The driver cannot consider each corner completely separately because the exit of one corner becomes the entry to the next.

### Sweeping Corner

A sweeping corner is a long, gradual bend.

It changes direction over a longer distance than a tight corner and can usually be taken at a higher speed.

### Dogleg

A dogleg is a small change in direction along a section that is otherwise mostly straight.

It does not necessarily act like a major corner.

It can be used to make a straight less completely straight and to change the overall shape of the circuit.

### Technical Section

A technical section contains several corners or direction changes close together.

The car has less time to run at high speed and the driver has to repeatedly brake, turn, position the car, and accelerate.

### Flowing Section

A flowing section has longer and smoother changes of direction.

The driver can normally maintain more speed through the section than through a highly technical section.

### Asymmetric Circuit

An asymmetric circuit has different types of sections rather than repeating the same pattern.

One part of the circuit might contain a long straight and a hairpin while another part might contain several fast corners.

This is important for procedural generation because it creates more meaningful differences between circuits.

## Circuit Topology

Topology refers to the overall structure of the circuit.

It describes what kinds of sections exist and how they are arranged.

For example, one circuit might contain a long straight followed by a hairpin, then a short straight, then a group of fast corners.

Another circuit might contain several technical corners followed by a long straight and a chicane.

Those circuits have different structures even if both use similar overall dimensions.

The project aims to change the structure of the circuit rather than simply creating the same oval shape with slightly different distortion.

## Circuit Geometry

### Centreline

The centreline is the mathematical path running through the middle of the track.

The program stores the circuit as a list of two-dimensional coordinates.

These coordinates are used by the renderer, validation system, and scoring system.

### Track Width

Track width is the width of the racing surface around the centreline.

The current program mainly uses the centreline for its geometry and visualisation.

It does not yet generate detailed track edges, kerbs, barriers, or run-off areas.

### Corner Radius

Corner radius describes how sharply a corner bends.

A large radius produces a gradual corner.

A small radius produces a tighter corner.

The generator uses minimum and maximum radius values to control the range of corners that can be produced.

### Curvature

Curvature describes how quickly the track changes direction.

A nearly straight section has low curvature.

A tight bend has higher curvature.

Sudden changes in curvature can make a track look unnatural, so the scoring system includes a smoothness measurement.

### Closed Circuit

A closed circuit is a continuous loop.

The track eventually returns to its starting point.

The generator creates the circuit as a closed path rather than as a road with separate start and end points.

### Self-Intersection

A self-intersection occurs when one part of the circuit crosses another part of the same circuit.

A normal racing circuit should not contain these crossings.

The geometry validation system checks for self-intersections.

### Track Separation

Track separation is the distance between different parts of the circuit that are close to each other.

Two sections do not need to cross to cause a problem.

If they are too close together, the width of the real track could cause them to overlap.

The generator therefore measures the minimum distance between non-neighbouring sections.

## Apex and Racing Line

### Apex

The apex is the point on the inside of a corner that the car normally aims towards.

The driver usually approaches the corner from one side, moves towards the inside of the corner, and then moves back towards the outside on the exit.

### Racing Line

The racing line is the path that a driver would ideally take through the circuit to maintain speed and achieve a fast lap.

The racing line is not always in the centre of the track.

The current project generates the circuit centreline but does not calculate a separate optimal racing line.

## Driving Direction

The circuit can be driven in either direction.

The `direction` parameter is either:

```text
1
```

or:

```text
-1
```

The direction changes the order in which the driver encounters the sections of the circuit.

The AI therefore has to decide whether the circuit should run in one direction or the other.

The AI is also required to explain the reason for its choice.

During later iterations, the AI can keep the existing direction or change it.

The saved feedback records why that decision was made.

## Layout Styles

The generator supports several broad layout styles.

### Fast

Fast circuits favour longer straights and less severe corners.

### Balanced

Balanced circuits use a mixture of straights, different corner types, and moderate changes in speed.

### Technical

Technical circuits contain more closely connected corners and more frequent direction changes.

### Hairpin

Hairpin-focused circuits place more emphasis on tight corners and large changes in speed.

### Flowing

Flowing circuits use longer bends and connected sections that allow more speed to be maintained.

### Asymmetric

Asymmetric circuits intentionally create greater differences between different parts of the lap.

These styles do not directly draw the circuit. They change how the geometry generator interprets the selected parameters.

## AI Parameter Selection

The AI can control parameters including:

* Track width
* Track height
* Number of corners
* Number of straights
* Minimum corner radius
* Maximum corner radius
* Shape variation
* Asymmetry
* Chicane probability
* Hairpin probability
* Complexity
* Minimum spacing
* Driving direction
* Layout style

The geometry system then converts these values into an actual circuit.

The purpose of this separation is to keep the AI responsible for design decisions while keeping the actual geometry under program control.

## AI Optimisation

The optimisation process works from the previous iteration.

For example:

```text
Iteration 1
Initial parameters
        ↓
Circuit generated
        ↓
Score calculated
        ↓
AI reviews result
        ↓
Iteration 2 parameters
        ↓
Circuit generated
        ↓
Score calculated
        ↓
AI reviews result
        ↓
...
        ↓
Iteration 20
```

The AI is given the previous parameter values and the previous score.

It is asked to explain:

* What should change
* Why it should change
* What effect the change should have
* Why the new design should be better
* Whether the driving direction should remain the same
* Why the direction was retained or changed

This makes the optimisation process visible rather than hiding it inside the program.

## Circuit Scoring

Each generated circuit receives a score out of 100.

The current scoring system checks several characteristics.

### Closure

Measures whether the circuit forms a proper loop.

### Separation

Measures how far apart different parts of the circuit are.

### Smoothness

Measures how smoothly the track changes direction.

### Complexity

Measures the amount of direction change in the circuit.

### Scale

Checks whether the circuit occupies a reasonable area.

### Variety

Measures variation in the lengths of the generated geometry sections.

### Validity

Checks whether the circuit contains self-intersections.

The final score combines these values using weighted scoring.

## Acceptance

The current acceptance threshold is:

```text
82.0
```

A circuit is considered accepted only when it reaches the required score and also meets the geometry requirements.

The current requirements include:

```text
Score >= 82.0
Validity >= 100.0
Separation >= 85.0
Closure >= 85.0
```

An accepted iteration is still not automatically selected as the final circuit.

All 20 iterations are evaluated.

The iteration with the highest total score becomes the best result for that circuit.

## Best Result Selection

Each circuit has 20 evaluated iterations.

The program keeps track of the highest `score.total`.

For example:

```text
Iteration 1   74.20
Iteration 2   77.91
Iteration 3   81.42
Iteration 4   83.65
Iteration 5   82.14
...
Iteration 20  84.01
```

The best result would be Iteration 20 because `84.01` is the highest score.

The best result is saved as:

```text
generated_circuits/best/circuit_001.json
```

The best file therefore represents the highest-scoring iteration, not automatically the final iteration.

## Saving Iterations

Every iteration is saved.

The directory structure is:

```text
generated_circuits/
    circuit_001/
        iteration_01/
            circuit.json
        iteration_02/
            circuit.json
        ...
        iteration_20/
            circuit.json
    circuit_002/
        iteration_01/
            circuit.json
        ...
        iteration_20/
            circuit.json
    ...
    circuit_020/
        iteration_01/
            circuit.json
        ...
        iteration_20/
            circuit.json
    best/
        circuit_001.json
        circuit_002.json
        ...
        circuit_020.json
```

This allows every iteration to be inspected after generation.

## Starting a New Generation Run

When the program starts a new generation run, the existing `generated_circuits` directory is cleared.

This prevents results from older runs from remaining in the `best` directory or being confused with the current generation.

A completed run therefore contains only the results from that run.

For the current configuration, the final output contains:

* 20 circuit directories
* 20 iterations in each circuit directory
* 400 iteration results in total
* 20 files in the `best` directory

## Saved Circuit Data

Each `circuit.json` contains the important information for that iteration.

This includes:

* Circuit number
* Iteration number
* Parameter values
* Generated geometry points
* Score
* Acceptance status
* AI decision
* AI reasoning
* Previous parameters where applicable
* New parameters

This means the generation process can be examined later without rerunning the program.

## Agent Feedback

The GUI records feedback separately for each circuit and iteration.

The user can select a circuit and then select any iteration that has been generated.

The interface displays:

### Decision type

Shows whether the parameters were created initially or revised later.

### Summary

A short explanation of the design decision.

### Parameter reasoning

Explains why the parameters were selected or changed.

### Direction reasoning

Explains why the circuit was chosen to run in its selected direction.

### Previous parameters

Shows the parameter values used before the revision.

### New parameters

Shows the parameter values proposed for the new iteration.

This allows the user to see how the design developed from iteration 1 through iteration 20.

## Graphical Interface

The PyQt6 interface contains several main areas.

### Circuit display

The current generated circuit is shown on the main canvas.

The canvas updates after every iteration.

The displayed circuit therefore changes as the optimisation process progresses.

### Circuit information

The interface shows:

* Current circuit
* Current iteration
* Current score
* Global best score
* Layout style
* Driving direction

### Progress

Two progress bars are displayed.

The first tracks progress through the 20 circuits.

The second tracks progress through the 20 iterations of the current circuit.

### Circuit selector

The circuit dropdown allows the user to select any circuit that has already been generated.

### Iteration selector

The iteration dropdown allows the user to select any available iteration for the selected circuit.

Selecting an iteration loads its stored information.

### Parameter viewer

Shows the parameters associated with the selected iteration.

### Agent feedback viewer

Shows the reasoning associated with the selected iteration.

### Daytona and agent output

The terminal area displays status information from the generation and verification process.

## Canvas Controls

The circuit can be moved and enlarged using the mouse.

### Zoom

Use the mouse wheel to zoom in or out.

### Pan

Hold the left mouse button and move the mouse to move the circuit around the canvas.

### Fit to canvas

The Fit button automatically scales the circuit so that it fits inside the visible area.

## Requirements

The application requires Python and the following packages:

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

The exact Daytona package version should match the API used by the project.

## Environment Variables

Create a `.env` file in the project directory.

```env
DAYTONA_API_KEY=your_daytona_api_key
GROQ_API_KEY=your_groq_api_key
```

Both keys are required.

The application will stop at startup if either key is missing.

## Running the Program

Run the Python file:

```bash
python 1.py
```

The application will:

1. Load the environment variables.
2. Create the Groq client.
3. Create the Daytona client.
4. Clear previous generation output.
5. Open the PyQt6 interface.
6. Generate the circuits.
7. Evaluate 20 iterations for each circuit.
8. Display each generated iteration.
9. Save every iteration.
10. Save the highest-scoring iteration for each circuit.

## Project Architecture

The main parts of the program are:

```text
AI parameter generation
        ↓
Parameter validation
        ↓
Procedural geometry generation
        ↓
Geometry validation
        ↓
Scoring
        ↓
Daytona verification
        ↓
GUI update
        ↓
Save iteration
        ↓
AI parameter revision
        ↓
Next iteration
```

The `GenerationWorker` controls the long-running generation process so that the graphical interface remains responsive.

The `CircuitCanvas` displays the generated geometry.

The `MainWindow` controls the interface and displays the saved information and AI feedback.

## Design Approach

The main design principle is that the AI controls the high-level design decisions while Python controls the physical geometry.

The AI decides things such as:

* How many corners the circuit should have
* How complex it should be
* How much asymmetry it should have
* Whether it should focus on speed or technical sections
* Whether it should contain hairpins or chicanes
* Which direction it should be driven

The Python geometry engine then creates the actual track.

This prevents the AI from directly creating arbitrary coordinate sequences that could easily produce invalid or unrealistic geometry.

The optimisation loop can therefore be described as:

```text
Design parameters
        ↓
Procedural geometry
        ↓
Geometry evaluation
        ↓
Score
        ↓
AI revision
        ↓
New design parameters
```

## Current Limitations

The current system is focused on procedural circuit layout.

It does not currently model:

* Elevation
* Banking
* Track camber
* Kerbs
* Run-off areas
* Barriers
* Pit lanes
* Pit buildings
* Terrain
* Track surface changes
* Weather
* Detailed vehicle physics
* Real racing lines
* Full lap-time simulation
* Real-world safety requirements

The current score is a mathematical evaluation created for this project.

It should not be treated as an official measure of whether a real-world circuit would be safe or suitable for construction.

Daytona is currently used as a sandbox verification system rather than as the primary geometry generator.

## Future Development

Possible future additions include:

* Automatic racing-line generation
* Vehicle-based lap-time simulation
* Sector generation
* Overtaking opportunity analysis
* Track width generation
* Run-off areas
* Pit lanes
* Kerbs
* Barriers
* Elevation changes
* Banking
* More detailed geometry validation
* Multi-objective optimisation
* Better comparison between circuits
* Export to game engines
* Export to SVG or other geometry formats
* Visual comparison between iterations
* Graphs showing how parameters and scores change during optimisation

## Summary of the Current System

The current system generates 20 different circuits.

Each circuit goes through 20 optimisation iterations.

That produces 400 evaluated iterations.

Every iteration is saved.

The highest-scoring iteration from each circuit is saved separately in the `best` directory.

The graphical interface shows the circuit while it is being generated and allows the user to inspect the parameters and AI reasoning for individual circuits and iterations.

The AI is used to make and explain design decisions, while the Python geometry engine is responsible for creating the actual physical circuit.
