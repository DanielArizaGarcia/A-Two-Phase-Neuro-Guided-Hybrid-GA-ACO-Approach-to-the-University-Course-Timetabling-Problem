# aco_horarios.py
# ===============
# Colonia de hormigas para mejorar:
#  - profesores
#  - aulas
#  - franjas horarias
#
# Incluye:
#  - vecinos (intercambios_validos_*)
#  - run_phase (una fase ACO)
#  - evaluate_config (3 fases)
#  - random_search (búsqueda aleatoria de hiperparámetros)
#  - run_with_fixed_params (varios runs con params fijos)

import random
import time
from copy import deepcopy
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt

from myutils.datos_horarios import TimetablingData
from arch.evaluacion_horarios import TimetablingEvaluator

# --- IMPORTANTE: Importamos el validador oficial ---
from src.factible import es_factible  
from myutils.datos_horarios import TimetablingData
from arch.evaluacion_horarios import TimetablingEvaluator

def es_factible_debug(sol,
                      tasks_df,
                      prof_subject,
                      prof_max_hours,
                      valid_rooms_for,
                      rooms_df,
                      Hmax_subject_day,
                      Hmax_prof_day):
    """
    Comprueba la factibilidad y devuelve una lista de errores (vacía si es factible).
    Ahora Hmax_subject_day se aplica por clase (grupo) y día.
    """
    from collections import defaultdict

    errors = []
    all_tasks = set(tasks_df.index)
    sol_tasks = set(sol.keys())

    # R1: cada tarea aparece exactamente una vez
    missing = all_tasks - sol_tasks
    extra   = sol_tasks - all_tasks
    if missing:
        errors.append(f"Tareas no asignadas: {sorted(missing)}")
    if extra:
        errors.append(f"Tareas extra en la solución: {sorted(extra)}")

    prof_total   = defaultdict(int)
    prof_daily   = defaultdict(int)
    subj_daily   = defaultdict(int)   # clave: ((grupo, subj), day)
    occ_prof     = set()
    occ_group    = set()
    occ_room     = set()
    occ_subj_ts  = set()

    for t, (p, (day, period), r) in sol.items():
        grupo = tasks_df.loc[t, "group"]
        subj  = tasks_df.loc[t, "subject"]
        n_est = tasks_df.loc[t, "n_students"]

        # R11: profesor capacitado
        if subj not in prof_subject.get(p, []):
            errors.append(f"{t}: Profesor {p} no capacitado para {subj}")

        # R7: aula válida
        if r not in valid_rooms_for[t]:
            errors.append(f"{t}: Aula {r} no válida para {subj}")

        # R10: capacidad
        cap = rooms_df.loc[r, "capacity"]
        if cap < n_est:
            errors.append(f"{t}: Aula {r} con capacidad {cap} < {n_est} alumnos")

        # R2: horas semanales por profesor
        prof_total[p] += 1
        if prof_total[p] > prof_max_hours[p]:
            errors.append(f"Profesor {p} imparte {prof_total[p]}h > máximo {prof_max_hours[p]}h semanales")

        # R9: horas diarias prof
        prof_daily[(p, day)] += 1
        if prof_daily[(p, day)] > Hmax_prof_day[(p, day)]:
            errors.append(f"Profesor {p} imparte {prof_daily[(p, day)]}h en {day} > límite diario {Hmax_prof_day[(p, day)]}")

        # R8: horas diarias asignatura por clase
        key_sd = ((grupo, subj), day)
        subj_daily[key_sd] += 1
        if subj_daily[key_sd] > Hmax_subject_day[(subj, day)]:
            errors.append(
                f"{subj} en grupo {grupo} suma {subj_daily[key_sd]}h en {day} > máximo diario {Hmax_subject_day[(subj, day)]}"
            )

        # R3: conflicto profesor
        key_prof = (p, day, period)
        if key_prof in occ_prof:
            errors.append(f"Profesor {p} tiene dos clases a {day}-{period}")
        occ_prof.add(key_prof)

        # R4: conflicto grupo
        key_grp = (grupo, day, period)
        if key_grp in occ_group:
            errors.append(f"Grupo {grupo} tiene dos clases a {day}-{period}")
        occ_group.add(key_grp)

        # R6: conflicto aula
        key_room = (r, day, period)
        if key_room in occ_room:
            errors.append(f"Aula {r} ocupada dos veces a {day}-{period}")
        occ_room.add(key_room)

        # R5: misma asignatura para un grupo
        key_subj = ((grupo, subj), day, period)
        if key_subj in occ_subj_ts:
            errors.append(f"{subj} en grupo {grupo} repetida a {day}-{period}")
        occ_subj_ts.add(key_subj)

    return errors



class ACOTimetabler:
    def __init__(self, data: TimetablingData, evaluator: TimetablingEvaluator):
        self.data = data
        self.evaluator = evaluator
        
        # Espacios de búsqueda (sin cambios)
        self.search_space_prof = dict(n_ants=("int", 50, 300), alpha=("float", 0.01, 1.0), beta=("float", 2.0, 8.0), evaporation=("float", 0.3, 0.95), q=("float", 0.1, 1.0), max_time=("int", 1, 20), changes_per_ant=("int", 1, 8))
        self.search_space_room = dict(n_ants=("int", 50, 300), alpha=("float", 0.01, 1.0), beta=("float", 2.0, 8.0), evaporation=("float", 0.3, 0.95), q=("float", 0.1, 1.0), max_time=("int", 1, 20), changes_per_ant=("int", 1, 5))
        self.search_space_time = dict(n_ants=("int", 50, 300), alpha=("float", 0.01, 1.0), beta=("float", 2.0, 8.0), evaporation=("float", 0.3, 1.0), q=("float", 0.1, 1.0), max_time=("int", 1, 20), changes_per_ant=("int", 1, 5))
    
    @staticmethod
    def _select_option(options, pheromones, heuristics, alpha, beta):
        weights = np.array([(pheromones.get(opt, 1.0) ** alpha) * (heuristics.get(opt, 1.0) ** beta) for opt in options])
        if weights.sum() == 0: weights = np.ones(len(options)) / len(options)
        else: weights /= weights.sum()
        return random.choices(options, weights=weights, k=1)[0]

    def profesores_posibles(self, t, sol):
        subj = self.data.tasks_df.loc[t, "subject"]
        horas_prof = {p: 0 for p in self.data.prof_max_hours.keys()}
        for _, (p, slot, _) in sol.items():
            if p in horas_prof: horas_prof[p] += 1
        return [p for p in self.data.prof_subject if subj in self.data.prof_subject[p] and horas_prof[p] < self.data.prof_max_hours[p]]

    def aulas_posibles(self, t, sol):
        return self.data.valid_rooms_for[t][:]

    def intercambios_validos_prof(self, t, sol):
        _, slot, _ = sol[t]
        posibles = []
        ocupados = {sol[other][0] for other in sol if sol[other][1] == slot and other != t}
        libres = [p for p in self.profesores_posibles(t, sol) if p not in ocupados]
        for p in libres: posibles.append(("asigna", p))
        for t2 in sol:
            if t2 == t: continue
            prof2, slot2, _ = sol[t2]
            if slot2 == slot:
                prof1 = sol[t][0]
                if (prof2 in self.profesores_posibles(t, sol)) and (prof1 in self.profesores_posibles(t2, sol)):
                    posibles.append(("swap", t2))
        return posibles

    def intercambios_validos_room(self, t, sol):
        _, slot, _ = sol[t]
        ocupadas = {sol[other][2] for other in sol if sol[other][1] == slot and other != t}
        libres = [r for r in self.aulas_posibles(t, sol) if r not in ocupadas]
        posibles = [("asigna", r) for r in libres]
        for t2 in sol:
            if t2 == t: continue
            _, slot2, room2 = sol[t2]
            if slot2 == slot and room2 in self.aulas_posibles(t, sol) and sol[t][2] in self.aulas_posibles(t2, sol):
                posibles.append(("swap", t2))
        return posibles

    def intercambios_validos_slot(self, t, sol):
        prof, _, room = sol[t]
        ocupadas = {sol[other][1] for other in sol if (sol[other][0] == prof or sol[other][2] == room) and other != t}
        libres = [(d, p) for d in self.data.days for p in range(1, 11) if (d, p) not in ocupadas]
        posibles = [("asigna", f) for f in libres]
        for t2 in sol:
            if t2 == t: continue
            prof2, slot2, room2 = sol[t2]
            slot1 = sol[t][1]
            if prof == prof2 or room == room2: continue
            ocupadas2 = {sol[other][1] for other in sol if (sol[other][0] == prof2 or sol[other][2] == room2) and other != t2}
            if slot2 not in ocupadas and slot1 not in ocupadas2:
                posibles.append(("swap", t2))
        return posibles

  
    def run_phase(self, sol_inicial, intercambios_func, element_name, n_ants, alpha, beta, evaporation, q, max_time, changes_per_ant):
        pheromones = dict()
        best_sol = sol_inicial.copy()
        best_cost = self.evaluator.cost(best_sol)
        costs_evol = []
        times_evol = []
        start = time.time()
        all_tasks = list(sol_inicial.keys())

        while (time.time() - start) < max_time:
            ants = []
            for _ in range(n_ants):
                sol_ant = sol_inicial.copy()
                tareas_a_cambiar = random.sample(all_tasks, k=min(changes_per_ant, len(all_tasks)))
                cambios = []

                for t in tareas_a_cambiar:
                    opciones = intercambios_func(t, sol_ant)
                    if not opciones: continue

                    heuristics = {o: 1.0 for o in opciones}
                    # Inicializar feromona si no existe
                    for o in opciones:
                        if (t, o) not in pheromones: pheromones[(t, o)] = 1.0

                    opcion = self._select_option(opciones, {o: pheromones[(t, o)] for o in opciones}, heuristics, alpha, beta)
                    
                    # Aplicar cambio
                    prof, slot, room = sol_ant[t]
                    if opcion[0] == "asigna":
                        if element_name == "profesor": sol_ant[t] = (opcion[1], slot, room)
                        elif element_name == "aula": sol_ant[t] = (prof, slot, opcion[1])
                        elif element_name == "franja": sol_ant[t] = (prof, opcion[1], room)
                    elif opcion[0] == "swap":
                        t2 = opcion[1]
                        p2, s2, r2 = sol_ant[t2]
                        if element_name == "profesor": sol_ant[t], sol_ant[t2] = (p2, slot, room), (prof, s2, r2)
                        elif element_name == "aula": sol_ant[t], sol_ant[t2] = (prof, slot, r2), (p2, s2, room)
                        elif element_name == "franja": sol_ant[t], sol_ant[t2] = (prof, s2, room), (p2, slot, r2)
                    
                    cambios.append((t, opcion))

                # Verificar factibilidad
                es_valida = es_factible(
                    sol_ant,
                    self.data.tasks_df,
                    self.data.prof_subject,
                    self.data.prof_max_hours,
                    self.data.valid_rooms_for, # Asegúrate que en TimetablingData se llame así
                    self.data.rooms_df,
                    self.data.Hmax_subject_day,
                    self.data.Hmax_prof_day
                )

                if es_valida is True:
                    cost = self.evaluator.cost(sol_ant)
                    ants.append((sol_ant.copy(), cost, cambios))
            
            # Actualización de feromonas y mejor solución
            if ants:
                ants.sort(key=lambda x: x[1])
                # Actualizar base para la siguiente iteración de hormigas dentro de esta fase (Elitista)
                sol_inicial = ants[0][0].copy() 

                if len(ants) > 1:
                    _, cost2, cambios2 = ants[1]
                    for (t, o) in cambios2:
                        pheromones[(t, o)] += q / (1.0 + cost2)
                
                for k in list(pheromones.keys()):
                    pheromones[k] *= (1 - evaporation)

                if ants[0][1] < best_cost:
                    best_sol = ants[0][0].copy()
                    best_cost = ants[0][1]

            costs_evol.append(best_cost)
            times_evol.append(time.time() - start)

        return best_sol, costs_evol, times_evol

    # -----------------------------------
    # Evaluar una configuración (3 fases)
    # -----------------------------------
    def evaluate_config(
        self,
        base_sol,
        params_prof: Dict[str, Any],
        params_room: Dict[str, Any],
        params_time: Dict[str, Any],
    ):
        sol0 = deepcopy(base_sol)

        sol_fase1, costs_prof, times_prof = self.run_phase(
            sol0,
            self.intercambios_validos_prof,
            "profesor",
            **params_prof,
        )

        sol_fase2, costs_room, times_room = self.run_phase(
            sol_fase1,
            self.intercambios_validos_room,
            "aula",
            **params_room,
        )

        sol_fase3, costs_time, times_time = self.run_phase(
            sol_fase2,
            self.intercambios_validos_slot,
            "franja",
            **params_time,
        )

        coste_final = self.evaluator.cost(sol_fase3)

        info = dict(
            sol_fase3  = sol_fase3,
            costs_prof = costs_prof,
            costs_room = costs_room,
            costs_time = costs_time,
            times_prof = times_prof,
            times_room = times_room,
            times_time = times_time,
        )
        return coste_final, info

    # -----------------------------------
    # Random search de hiperparámetros
    # -----------------------------------
    @staticmethod
    def _sample_from_space(space):
        params = {}
        for name, (kind, lo, hi) in space.items():
            if kind == "int":
                params[name] = random.randint(int(lo), int(hi))
            else:
                params[name] = random.uniform(float(lo), float(hi))
        return params

    def random_search(
        self,
        base_sol,
        n_trials: int = 30,
        seed: int = 0,
        early_stop_cost: Optional[float] = None,
        verbose: bool = True,
    ):
        random.seed(seed)

        best_cost = float("inf")
        best_params_prof = None
        best_params_room = None
        best_params_time = None
        best_run_info = None

        history = []

        base_cost = self.evaluator.cost(base_sol)
        if verbose:
            print(f"Coste inicial (sol de partida): {base_cost:.3f}")

        for trial in range(1, n_trials + 1):
            params_prof = self._sample_from_space(self.search_space_prof)
            params_room = self._sample_from_space(self.search_space_room)
            params_time = self._sample_from_space(self.search_space_time)

            coste_final, info = self.evaluate_config(
                base_sol=base_sol,
                params_prof=params_prof,
                params_room=params_room,
                params_time=params_time,
            )

            history.append({
                "trial": trial,
                "cost": coste_final,
                "params_prof": params_prof,
                "params_room": params_room,
                "params_time": params_time,
            })

            if verbose:
                print(f"[{trial}/{n_trials}] Coste = {coste_final:.3f}")

            if coste_final < best_cost:
                best_cost = coste_final
                best_params_prof = params_prof
                best_params_room = params_room
                best_params_time = params_time
                best_run_info = info
                if verbose:
                    print("  -> Nuevo mejor encontrado.")

            if early_stop_cost is not None and best_cost <= early_stop_cost:
                if verbose:
                    print(f"Parando pronto: best_cost={best_cost:.3f} ≤ {early_stop_cost}")
                break

        if verbose:
            print("\n=== MEJORES HIPERPARÁMETROS ENCONTRADOS ===")
            print(f"Mejor coste final: {best_cost:.3f}\n")
            print("params_prof =", best_params_prof)
            print("params_room =", best_params_room)
            print("params_time =", best_params_time)

        return best_params_prof, best_params_room, best_params_time, best_run_info, history

    # -----------------------------------
    # Runs con parámetros fijos (tipo n_runs)
    # -----------------------------------
    def run_with_fixed_params(
        self,
        base_sol,
        params_prof: Dict[str, Any],
        params_room: Dict[str, Any],
        params_time: Dict[str, Any],
        n_runs: int = 1,
        save_prefix: Optional[str] = None,
        grafica: bool = False,
    ):
        """
        Ejecuta n_runs empezando en base_sol, y cada run empieza
        desde la solución final de la anterior.

        Devuelve:
        - best_sol, best_df, best_cost, best_costs_prof, best_costs_room, best_costs_time
        (y opcionalmente dibuja la evolución de coste del mejor run si grafica=True)
        """
        best_cost = float("inf")
        best_sol = None
        best_df = None
        best_costs_prof = None
        best_costs_room = None
        best_costs_time = None

        sol_actual = base_sol

        for run in range(1, n_runs + 1):
            print(f"\n===== RUN {run}/{n_runs} =====")
            print(f"Coste inicial: {self.evaluator.cost(sol_actual):.3f}")

            sol_fase1, costs_prof, times_prof = self.run_phase(
                sol_actual,
                self.intercambios_validos_prof,
                "profesor",
                **params_prof,
            )
            df_fase1 = self.evaluator.solution_to_df(sol_fase1)
            print(f"Coste tras fase 1 (profesores): {self.evaluator.cost(sol_fase1):.3f}")

            sol_fase2, costs_room, times_room = self.run_phase(
                sol_fase1,
                self.intercambios_validos_room,
                "aula",
                **params_room,
            )
            df_fase2 = self.evaluator.solution_to_df(sol_fase2)
            print(f"Coste tras fase 2 (aulas): {self.evaluator.cost(sol_fase2):.3f}")

            sol_fase3, costs_time, times_time = self.run_phase(
                sol_fase2,
                self.intercambios_validos_slot,
                "franja",
                **params_time,
            )
            df_fase3 = self.evaluator.solution_to_df(sol_fase3)
            coste_final = self.evaluator.cost(sol_fase3)
            print(f"Coste final ponderado (run {run}): {coste_final:.3f}")

            if coste_final < best_cost:
                best_cost = coste_final
                best_sol = sol_fase3
                best_df = df_fase3
                best_costs_prof = costs_prof
                best_costs_room = costs_room
                best_costs_time = costs_time
                print("  -> Nueva mejor solución encontrada.")

            sol_actual = sol_fase3

            if save_prefix is not None:
                df_fase3.to_csv(f"{save_prefix}_run_{run}.csv", index=False)

        print("\n===== RESUMEN FINAL =====")
        print(f"Mejor coste tras {n_runs} runs: {best_cost:.3f}")

        # --- Gráfica evolución de coste del MEJOR run ---
        if grafica and best_costs_prof is not None:
            plt.figure(figsize=(12, 5))

            iters_prof = np.arange(len(best_costs_prof))
            iters_room = np.arange(len(best_costs_room)) + len(best_costs_prof)
            iters_time = np.arange(len(best_costs_time)) + len(best_costs_prof) + len(best_costs_room)

            plt.plot(iters_prof, best_costs_prof, label="Profesores")
            plt.plot(iters_room, best_costs_room, label="Aulas")
            plt.plot(iters_time, best_costs_time, label="Franjas horarias")

            plt.xlabel("Iteración")
            plt.ylabel("Coste")
            plt.title("Evolución del coste vs iteración (mejor run)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return best_sol, best_df, best_cost, best_costs_prof, best_costs_room, best_costs_time

