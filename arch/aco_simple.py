import random
import copy
import numpy as np
import time
from collections import defaultdict

class SimpleACO:
    def __init__(self, datos, evaluator, validador_factible, n_ants=5, evaporation=0.1, alpha=1.0, beta=2.0):
        self.datos = datos
        self.evaluator = evaluator
        self.es_factible_fn = validador_factible
        
        # Parámetros
        self.n_ants = n_ants
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        self.perturbation_rate = 0.2  # Porcentaje de tareas a reconstruir por hormiga
        
        # Cache de datos
        self.tasks_ids = list(datos['tasks_df'].index)
        self.days = datos['days']
        self.rooms_ids = list(datos['rooms_df'].index)
        self.max_period = int(datos.get('max_period', 10))
        
        # Mapa de feromonas: {(task_id, dia, hora, aula): valor}
        self.pheromones = defaultdict(lambda: 0.1)
        
        self.best_global_sol = None
        self.best_global_cost = float('inf')

        # Cache de asignaturas y profes para validaciones rápidas
        self.prof_subject = datos.get('prof_subject', {})

    def _es_factible_completo(self, sol):
        """Verificación estricta de factibilidad."""
        res = self.es_factible_fn(
            sol,
            self.datos['tasks_df'],
            self.datos['prof_subject'],
            self.datos['prof_max_hours'],
            self.datos['valid_rooms'],
            self.datos['rooms_df'],
            self.datos['Hmax_subject_day'],
            self.datos['Hmax_prof_day']
        )
        if isinstance(res, bool): return res
        if isinstance(res, (int, np.integer)): return res == 0
        if isinstance(res, dict) and 'total_violations' in res: return res['total_violations'] == 0
        return bool(res)

    def _inicializar_con_solucion(self, sol_inicial):
        """Inicializa feromonas basándose en la solución inicial."""
        # Verificar si la inicial es válida
        if self._es_factible_completo(sol_inicial):
            self.best_global_sol = copy.deepcopy(sol_inicial)
            self.best_global_cost = self.evaluator.cost(sol_inicial)
            print(f"[ACO] Arrancando con solución válida. Coste: {self.best_global_cost:.2f}")
        else:
            print("[ACO] ⚠️ La solución inicial no es factible. Intentando repararla...")
            self.best_global_sol = self._intentar_reparar(sol_inicial)
            self.best_global_cost = self.evaluator.cost(self.best_global_sol)
            print(f"[ACO] Solución reparada. Coste: {self.best_global_cost:.2f}")

        # Sembrar feromonas iniciales
        for t_id, (p, (d, h), r) in self.best_global_sol.items():
            self.pheromones[(t_id, d, h, r)] = 5.0

    def _intentar_reparar(self, sol):
        """Intenta arreglar una solución inválida moviendo tareas conflictivas al azar."""
        mejor_reparada = copy.deepcopy(sol)
        
        # Si ya es factible, devolver
        if self._es_factible_completo(mejor_reparada): 
            return mejor_reparada

        # Intentos de fuerza bruta para encontrar factibilidad
        for _ in range(100):
            temp_sol = copy.deepcopy(sol)
            # Mover 5 tareas aleatorias a huecos aleatorios
            for _ in range(5):
                tid = random.choice(self.tasks_ids)
                p = temp_sol[tid][0]
                nd = random.choice(self.days)
                nh = random.randint(1, self.max_period)
                valid_rooms = self.datos['valid_rooms'].get(tid, self.rooms_ids)
                nr = random.choice(valid_rooms) if valid_rooms else random.choice(self.rooms_ids)
                temp_sol[tid] = (p, (nd, nh), nr)
            
            if self._es_factible_completo(temp_sol):
                return temp_sol
        
        return sol # Si falla, devuelve la original (el algoritmo tratará de mejorarla luego)

    def _construir_hormiga_inteligente(self):
        """
        Estrategia LNS: Toma la mejor solución, destruye una parte y la reconstruye usando feromonas.
        """
        # 1. Copiar la mejor base
        sol = copy.deepcopy(self.best_global_sol)
        
        # 2. Seleccionar tareas a re-planificar (Perturbación)
        n_tasks_to_change = int(len(self.tasks_ids) * self.perturbation_rate)
        # Seleccionamos tareas aleatorias para quitar
        tasks_to_move = random.sample(self.tasks_ids, max(1, n_tasks_to_change))
        
        # 3. Estructuras de ocupación para el resto de tareas fijas
        occupied = set()
        for tid, (p, (d, h), r) in sol.items():
            if tid not in tasks_to_move:
                grp = self.datos['tasks_df'].loc[tid, 'group']
                occupied.add((p, d, h)) # Profe ocupado
                occupied.add((r, d, h)) # Aula ocupada
                occupied.add((grp, d, h)) # Grupo ocupado

        # 4. Reconstruir las tareas eliminadas usando Feromonas + Heurística
        for tid in tasks_to_move:
            prof = sol[tid][0]
            grupo = self.datos['tasks_df'].loc[tid, 'group']
            valid_rooms = self.datos['valid_rooms'].get(tid, self.rooms_ids)
            if not valid_rooms: valid_rooms = self.rooms_ids

            # Generar candidatos
            candidates = []
            
            # Intentar encontrar X slots libres
            intentos = 0
            while len(candidates) < 5 and intentos < 30:
                d = random.choice(self.days)
                h = random.randint(1, self.max_period)
                r = random.choice(valid_rooms)
                
                # Chequeo rápido de colisión
                if (prof, d, h) not in occupied and \
                   (r, d, h) not in occupied and \
                   (grupo, d, h) not in occupied:
                    candidates.append((d, h, r))
                intentos += 1
            
            # Si no hay candidatos libres, forzamos uno aleatorio (permitimos infactibilidad temporal)
            if not candidates:
                d = random.choice(self.days)
                h = random.randint(1, self.max_period)
                r = random.choice(valid_rooms)
                candidates.append((d, h, r))

            # Elección por Feromona
            probs = []
            for (d, h, r) in candidates:
                # Valor feromona
                tau = self.pheromones[(tid, d, h, r)]
                # Heurística: Preferir huecos libres (10.0) vs ocupados (0.1)
                is_free = 1.0
                if (prof, d, h) in occupied or (r, d, h) in occupied:
                    is_free = 0.1
                
                prob = (tau ** self.alpha) * (is_free ** self.beta)
                probs.append(prob)
            
            # Normalizar
            total = sum(probs)
            if total == 0: probs = [1/len(probs)] * len(probs)
            else: probs = [p/total for p in probs]
            
            # Ruleta
            idx = np.random.choice(len(candidates), p=probs)
            d_sel, h_sel, r_sel = candidates[idx]
            
            # Asignar
            sol[tid] = (prof, (d_sel, h_sel), r_sel)
            
            # Actualizar ocupación
            occupied.add((prof, d_sel, h_sel))
            occupied.add((r, d, h))
            occupied.add((grupo, d, h))
            
        return sol

    def ejecutar(self, sol_inicial, n_iters=20, max_time=60):
        start_time = time.time()
        self._inicializar_con_solucion(sol_inicial)
        
        history = [self.best_global_cost]
        
        print(f"--- Inicio ACO (Modo LNS) | Coste Inicial: {self.best_global_cost:.2f} ---")
        
        sin_mejora = 0
        
        for it in range(n_iters):
            if time.time() - start_time > max_time:
                print(f"[ACO] 🛑 Tiempo límite alcanzado en iter {it}")
                break
                
            iteration_best_sol = None
            iteration_best_cost = float('inf')
            
            # --- Fase de Hormigas ---
            for _ in range(self.n_ants):
                # Construir/Modificar solución
                ant_sol = self._construir_hormiga_inteligente()
                
                # Validar
                if self._es_factible_completo(ant_sol):
                    cost = self.evaluator.cost(ant_sol)
                    
                    # Guardar la mejor de esta iteración
                    if cost < iteration_best_cost:
                        iteration_best_cost = cost
                        iteration_best_sol = ant_sol
            
            # --- Actualización Global ---
            if iteration_best_sol:
                # Si encontramos algo mejor que el global
                if iteration_best_cost < self.best_global_cost:
                    print(f"[ACO] ⭐ ¡Mejora en iter {it}! {self.best_global_cost:.2f} -> {iteration_best_cost:.2f}")
                    self.best_global_cost = iteration_best_cost
                    self.best_global_sol = copy.deepcopy(iteration_best_sol)
                    sin_mejora = 0
                    
                    # Refuerzo de feromona fuerte (Elite)
                    reward = 5.0
                    for tid, (p, (d, h), r) in self.best_global_sol.items():
                        self.pheromones[(tid, d, h, r)] += reward
                else:
                    sin_mejora += 1
            else:
                sin_mejora += 1

            # --- Evaporación ---
            for k in list(self.pheromones.keys()):
                self.pheromones[k] *= (1.0 - self.evaporation)
                # Limpieza para ahorrar memoria
                if self.pheromones[k] < 0.05:
                    del self.pheromones[k]

            # --- Mecanismo de Escape (Si nos atascamos) ---
            # Si llevamos X iters sin mejora, aumentamos la tasa de destrucción
            if sin_mejora > 5:
                self.perturbation_rate = min(0.5, self.perturbation_rate + 0.05)
            else:
                self.perturbation_rate = 0.2 # Reset a base

            history.append(self.best_global_cost)
            
            # Feedback simple
            if it % 2 == 0:
                print(f"[ACO] Iter {it}: Mejor Coste = {self.best_global_cost:.2f} (Perturbación: {self.perturbation_rate:.2f})")

        print(f"--- Fin ACO | Coste Final: {self.best_global_cost:.2f} ---")
        return self.best_global_sol, self.best_global_cost, history