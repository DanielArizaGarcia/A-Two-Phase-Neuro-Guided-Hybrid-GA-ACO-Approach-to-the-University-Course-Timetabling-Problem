# Enfoque Híbrido Neuro‑Guiado GA+ACO para el Problema de Horarios Universitarios (UCTP)

Este repositorio implementa un enfoque en dos fases para construir horarios docentes factibles y de alta calidad:

- Fase 1: un Algoritmo Genético (GA) genera rápidamente horarios factibles, priorizando la reparación de conflictos y la convergencia temprana.
- Fase 2: una Colonia de Hormigas (ACO), guiada por señales aprendidas con una red neuronal (MLP), refina el horario reduciendo costes blandos (huecos, aulas y plantas incoherentes, franjas poco deseadas, etc.).

Incluye cuadernos con comparativas de líneas base y scripts modulares para reproducir todo el flujo: carga de datos, factibilidad, evaluación, optimización y visualización interactiva.

## Datos
- Carpeta `data/` con cuatro CSVs: `Asignaturas.csv`, `Aulas.csv`, `Clases.csv`, `Profesores.csv`.
- El módulo `myutils/datos_horarios.py` construye las estructuras necesarias a partir de esos CSVs: tareas `tasks_df`, profesores por asignatura, aulas válidas por tarea, límites diarios/semanales, etc.

## Metodología (2 fases)
- GA (Fase 1):
	- Inicialización mixta greedy + aleatoria para diversidad controlada.
	- Cruce uniforme y mutación con reparaciones dirigidas por violaciones detectadas.
	- Fitness determinista con fuerte incentivo a soluciones factibles.
	- Código principal: `arch/genetic_optimizer.py` y utilidades en `myutils/_ga_functions.py`.

- ACO + Red (Fase 2):
	- Tres sub‑fases ACO sobre profesor, aula y franja, con intercambio/assignación válidos por tarea.
	- Evaluación basada en un evaluador de costes blandos ponderados.
	- Una red MLP (scikit‑learn) aprende a puntuar movimientos locales y guía la exploración (exploitation vs exploration).
	- Código principal: `arch/aco_horarios.py` y `arch/NeuroOptimizador.py`.

- Factibilidad y evaluación:
	- Validador oficial en `src/factible.py` (función `es_factible`) con reglas duras: no solapes de profesor/grupo/aula, capacidades, límites diarios/semanales, aulas válidas por asignatura, etc.
	- Evaluador blando en `arch/evaluacion_horarios.py` para medir calidad del horario.

## Estructura del repositorio
- `data/`: CSVs de entrada del problema.
- `arch/`: algoritmos ACO/GA, evaluador y clase neuro‑optimizadora.
- `myutils/`: carga y transformación de datos, utilidades GA.
- `src/`: utilidades de factibilidad y visualización (`graficar_horarios.py`).
- Cuadernos:
	- `Feasible-GA.ipynb`: generación y análisis de soluciones factibles con GA.
	- `Neural-ACO.ipynb`: refinamiento con ACO guiado por la red.
	- `Baselines_algorithms.ipynb`: comparativas con líneas base clásicas.
- Visualización final: `horario_neuro_final.html` (interactiva con Plotly).

## Requisitos e instalación (uv)
Usamos `uv` (Astral) para gestionar entornos y dependencias a partir de `pyproject.toml`.

1) Instala `uv` (si no lo tienes):

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

2) Crea y usa el entorno del proyecto, instalando dependencias:

```bash
# Desde la raíz del repo
uv venv
uv sync

# Activar el entorno (bash)
source .venv/bin/activate
```

`uv sync` leerá `pyproject.toml` y resolverá/instalará todas las dependencias.

## Cómo reproducir (resumen)
1) Preparar datos: coloca/valida los CSVs en `data/`.
2) Fase 1 (GA): ejecutar `Feasible-GA.ipynb` o tus propios scripts con `arch/genetic_optimizer.py` para obtener una solución factible inicial.
3) Fase 2 (ACO+MLP): ejecutar `Neural-ACO.ipynb` para refinar la solución con `arch/aco_horarios.py` y `arch/NeuroOptimizador.py`.
4) Guardar/visualizar: usa `src/graficar_horarios.py` para generar HTML interactivo del horario por grupos.

Ejemplo mínimo de visualización en Python (si ya tienes un DataFrame `df_horario`):

```python
from src.graficar_horarios import plot_horario_interactivo
# df_horario con columnas: ["Grupo","Asignatura","Profesor","Día","Hora","Aula"]
plot_horario_interactivo(df_horario, filename="horario_neuro_final.html")
```

## Ver la solución (HTML)
- Archivo generado: `horario_neuro_final.html` en la raíz del proyecto.
- En local, puedes abrirlo con tu navegador.

Abrir directamente (Linux):

```bash
xdg-open horario_neuro_final.html
```

Servir por HTTP (útil si el navegador bloquea recursos locales):

```bash
python -m http.server 8080
# luego visita: http://localhost:8080/horario_neuro_final.html
```

Limitación README en GitHub: GitHub no embebe archivos HTML interactivos dentro del README por seguridad. Opciones para “verlo” desde GitHub:
- Enlace directo al archivo: [`horario_neuro_final.html`](./horario_neuro_final.html) (descárgalo y ábrelo localmente).
- Publicarlo vía GitHub Pages y enlazar la URL pública.
- Incluir una captura/animación del horario en el README como imagen de referencia.

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
