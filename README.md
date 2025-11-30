# Neuro‑Guided Hybrid GA+ACO for the University Course Timetabling Problem (UCTP)

This repository implements a two‑phase approach to build feasible, high‑quality teaching timetables:

- Phase 1: a Genetic Algorithm (GA) rapidly produces feasible timetables, prioritizing conflict repair and early convergence.
- Phase 2: an Ant Colony Optimization (ACO) procedure, guided by signals learned via a neural network (MLP), refines the solution by reducing soft costs (gaps, incoherent room floors, undesired time slots, etc.).

It includes notebooks with baseline comparisons and modular scripts to reproduce the full pipeline: data loading, feasibility, evaluation, optimization, and interactive visualization.

## Data
- Folder `data/` with four CSVs: `Asignaturas.csv`, `Aulas.csv`, `Clases.csv`, `Profesores.csv`.
- Module `myutils/datos_horarios.py` builds the required structures from these CSVs: `tasks_df`, teacher‑subject mapping, valid rooms per task, daily/weekly limits, and more.

## Methodology (2 phases)
- GA (Phase 1):
	- Mixed greedy + random initialization for controlled diversity.
	- Uniform crossover and mutation with repairs driven by detected violations.
	- Deterministic fitness with strong incentive for feasible solutions.
	- Main code: `arch/genetic_optimizer.py` with utilities in `myutils/_ga_functions.py`.

- ACO + Neural guidance (Phase 2):
	- Three ACO sub‑phases targeting teacher, room, and time slot, using valid assignment/swap moves.
	- Evaluation by a weighted soft‑cost function.
	- An MLP (scikit‑learn) learns to score local moves and guides exploration (exploitation vs exploration).
	- Main code: `arch/aco_horarios.py` and `arch/NeuroOptimizador.py`.

- Feasibility and evaluation:
	- Official feasibility validator in `src/factible.py` (`es_factible`) with hard rules: no teacher/group/room overlaps, capacity checks, daily/weekly limits, valid room types per subject, etc.
	- Soft evaluator in `arch/evaluacion_horarios.py` to measure timetable quality.

## Repository Structure
- `data/`: problem input CSVs.
- `arch/`: ACO/GA algorithms, evaluator, and neural optimizer class.
- `myutils/`: data loading/transformation, GA utilities.
- `src/`: feasibility and visualization utilities (`graficar_horarios.py`).
- Notebooks:
	- `Feasible-GA.ipynb`: GA generation and analysis of feasible solutions.
	- `Neural-ACO.ipynb`: refinement with neural‑guided ACO.
	- `Baselines_algorithms.ipynb`: comparisons against classic baselines.
- Final visualization: `horario_neuro_final.html` (Plotly interactive).

## Requirements and Setup (uv)
We use `uv` (Astral) to manage environments and dependencies from `pyproject.toml`.

1) Install `uv` (if not already):

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

2) Create and use the project environment, installing dependencies:

```bash
# From the repo root
uv venv
uv sync

# Activate the environment (bash)
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` and resolves/installs all dependencies.

## How to Reproduce (quick path)
1) Prepare data: place/validate the CSVs in `data/`.
2) Phase 1 (GA): run `Feasible-GA.ipynb` or your own script with `arch/genetic_optimizer.py` to obtain an initial feasible timetable.
3) Phase 2 (ACO+MLP): run `Neural-ACO.ipynb` to refine the solution using `arch/aco_horarios.py` and `arch/NeuroOptimizador.py`.
4) Save/visualize: use `src/graficar_horarios.py` to generate an interactive HTML by group.

Minimal Python example (if you already have a DataFrame `df_horario`):

```python
from src.graficar_horarios import plot_horario_interactivo
# df_horario columns: ["Grupo","Asignatura","Profesor","Día","Hora","Aula"]
plot_horario_interactivo(df_horario, filename="horario_neuro_final.html")
```

## View the Solution (HTML)
- Generated file: `horario_neuro_final.html` at the project root.
- GitHub README cannot embed interactive HTML for security reasons.
- Please download the file and open it locally with your browser.

Open locally (Linux):

```bash
xdg-open horario_neuro_final.html
```

Alternatively, serve it via a local HTTP server if your browser blocks local resources:

```bash
python -m http.server 8080
# then visit: http://localhost:8080/horario_neuro_final.html
```

Direct link to the artifact for download: [`horario_neuro_final.html`](./horario_neuro_final.html)

## Expected Results
- Phase 1 reliably finds feasible solutions in minutes depending on size.
- Phase 2 reduces soft costs (gaps, isolated afternoons, floor changes, etc.) and improves timetable readability.
- Interactive visualization by group/day/period with hover details (teacher and room).

## Quick References
- Feasibility validator: `src/factible.py`
- Quality evaluation: `arch/evaluacion_horarios.py`
- ACO: `arch/aco_horarios.py`
- MLP guidance: `arch/NeuroOptimizador.py`
- Visualization: `src/graficar_horarios.py`

---
If you want, I can add a small CLI script to run end‑to‑end (load → GA → ACO → HTML) with command‑line parameters.

## Resultados esperados
- Fase 1 consigue soluciones factibles de manera robusta en minutos según tamaño.
- Fase 2 reduce costes blandos (huecos, tardes aisladas, saltos de planta, etc.) y mejora la legibilidad del horario.
- Visualización interactiva por grupo/día/hora con hover de detalles (profesor y aula).

## Referencias rápidas
- Validador de factibilidad: `src/factible.py`
- Evaluación de calidad: `arch/evaluacion_horarios.py`
- ACO: `arch/aco_horarios.py`
- Red MLP guía: `arch/NeuroOptimizador.py`
- Visualización: `src/graficar_horarios.py`

---
Si quieres, puedo añadir un script CLI para ejecutar de extremo a extremo (carga → GA → ACO → HTML) con parámetros por línea de comandos.
