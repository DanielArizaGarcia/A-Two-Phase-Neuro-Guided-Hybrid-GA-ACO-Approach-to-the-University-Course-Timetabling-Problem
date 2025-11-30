import random
from collections import defaultdict
from src.factible import es_factible
from myutils._ga_functions import generar_horario_aleatorio, generar_horario_greedy
from myutils.dataloader import normalizar_texto


class OptimizadorHorarios:
    def __init__(self, datos, seed=42):
        """
        Inicializa el optimizador con los datos y una semilla.
        Crea un generador aleatorio interno (self.rng) para aislar la aleatoriedad.
        La semilla SOLO se fija aquí, no en cada llamada, para que el GA pueda explorar.
        """
        self.datos = datos
        self.seed = seed

        # RNG interno
        self.rng = random.Random(seed)

        # Opcional: fijar semilla global y de numpy una vez (para reproducibilidad del run completo)
        import numpy as _np
        
        self.np = _np
        random.seed(seed)
        self.np.random.seed(seed)

    # =========================================================
    # CREACIÓN DE INDIVIDUOS
    # =========================================================
    def crear_individuo(self):
        """
        Decide si usa Greedy o Aleatorio.
        Ahora NO reseedea cada vez -> los individuos son distintos.
        Priorizamos greedy para converger rápido, pero dejamos algo de aleatorio para diversidad.
        """
        # 5% aleatorios, 95% greedy
        if self.rng.random() < 0.05:
            return generar_horario_aleatorio(self.datos)
        else:
            # si has puesto num_intentos en generar_horario_greedy, aquí lo aprovecha
            return generar_horario_greedy(self.datos)

    # =========================================================
    # EVALUACIÓN
    # =========================================================
    def evaluar_horario(self, solucion):
        """
        Evalúa el fitness (determinista).
        NO usa random ni reseedea.
        """
        tasks_df         = self.datos["tasks_df"]
        prof_subject     = self.datos["prof_subject"]
        prof_max_hours   = self.datos["prof_max_hours"]
        valid_rooms_for  = self.datos["valid_rooms"]
        rooms_df         = self.datos["rooms_df"]
        Hmax_subject_day = self.datos["Hmax_subject_day"]
        Hmax_prof_day    = self.datos["Hmax_prof_day"]

        res = es_factible(
            solucion,
            tasks_df,
            prof_subject,
            prof_max_hours,
            valid_rooms_for,
            rooms_df,
            Hmax_subject_day,
            Hmax_prof_day,
        )

        if res is True:
            # premio extra a factibles para que el GA las prefiera fuerte
            return 2.0, 0, True
        else:
            nviol = int(res)
            fitness = 1.0 / (1.0 + nviol)
            return fitness, nviol, False

    # =========================================================
    # CRUCE
    # =========================================================
    def crossover_horarios(self, ind1, ind2):
        """
        Cruce uniforme usando self.rng.
        NO reseedea -> hijos distintos en cada llamada.
        """
        hijo1 = {}
        hijo2 = {}

        for task_id in ind1.keys():
            if self.rng.random() < 0.5:
                hijo1[task_id] = ind1[task_id]
                hijo2[task_id] = ind2[task_id]
            else:
                hijo1[task_id] = ind2[task_id]
                hijo2[task_id] = ind1[task_id]

        ind1.clear()
        ind1.update(hijo1)

        ind2.clear()
        ind2.update(hijo2)

        return ind1, ind2

    # =========================================================
    # MUTACIÓN + REPARACIÓN
    # =========================================================
    def mutar_horario(self, solucion, 
                      rate_teacher=0.5, 
                      rate_slot=0.7, 
                      rate_room=0.5, 
                      max_conflict_tasks=30, 
                      max_tries=6, 
                      random_mut_prob=0.01):
        """
        Mutación usando self.rng para todas las decisiones aleatorias.
        Hace:
        1) snapshot de ocupaciones + detección de conflictos
        2) intenta reparar tareas conflictivas
        3) pequeña mutación aleatoria sobre el resto
        """
        tasks_df            = self.datos["tasks_df"]
        prof_max_hours      = self.datos["prof_max_hours"]
        Hmax_prof_day       = self.datos["Hmax_prof_day"]
        valid_rooms_for     = self.datos["valid_rooms"]
        subject_to_teachers = self.datos["subject_to_teachers"]
        professors          = self.datos["professors"]
        timeslots           = self.datos["timeslots"]
        rooms_df            = self.datos["rooms_df"]
        prof_subject        = self.datos["prof_subject"]
        task_ids            = self.datos["task_ids"]

        # 1) Snapshot de ocupación
        prof_total = defaultdict(int)
        prof_daily = defaultdict(int)
        occ_prof   = defaultdict(list)
        occ_group  = defaultdict(list)
        occ_room   = defaultdict(list)

        for t, (p, (day, period), r) in solucion.items():
            row   = tasks_df.loc[t]
            group = row["group"]

            prof_total[p]       += 1
            prof_daily[(p, day)] += 1
            occ_prof[(p, day, period)].append(t)
            occ_group[(group, day, period)].append(t)
            occ_room[(r, day, period)].append(t)

        conflict_tasks = set()

        # Detección de conflictos por solape
        for lst in occ_prof.values():
            if len(lst) > 1:
                conflict_tasks.update(lst)
        for lst in occ_group.values():
            if len(lst) > 1:
                conflict_tasks.update(lst)
        for lst in occ_room.values():
            if len(lst) > 1:
                conflict_tasks.update(lst)

        # Exceso de horas semanales
        for p, total in prof_total.items():
            if total > prof_max_hours[p]:
                for t, (pp, (_, _), _) in solucion.items():
                    if pp == p:
                        conflict_tasks.add(t)

        # Exceso de horas diarias
        for (p, day), dcount in prof_daily.items():
            if dcount > Hmax_prof_day[(p, day)]:
                for t, (pp, (dd, _), _) in solucion.items():
                    if pp == p and dd == day:
                        conflict_tasks.add(t)

        conflict_tasks = list(conflict_tasks)
        self.rng.shuffle(conflict_tasks)
        conflict_tasks = conflict_tasks[:max_conflict_tasks]

        # 2) Precalcular profes relativamente libres
        underloaded_profs = {
            p for p in professors if prof_total[p] < prof_max_hours[p]
        }

        prof_free_slots = {}
        for p in professors:
            free = []
            for (day, period) in timeslots:
                if (p, day, period) in occ_prof:
                    continue
                if prof_daily[(p, day)] >= Hmax_prof_day[(p, day)]:
                    continue
                free.append((day, period))
            prof_free_slots[p] = free

        # 3) Reparar tareas conflictivas
        for t in conflict_tasks:
            old_p, (old_day, old_period), old_r = solucion[t]
            row   = tasks_df.loc[t]
            group = row["group"]
            subj  = row["subject"]
            subj_key = normalizar_texto(subj)
            n_est = int(row["n_students"])

            base_profs = list(subject_to_teachers.get(subj_key, professors))
            if not base_profs:
                base_profs = list(professors)

            candidate_profs = [p for p in base_profs if p in underloaded_profs]
            if not candidate_profs:
                candidate_profs = base_profs

            for _ in range(max_tries):
                # Elegir profesor
                if self.rng.random() < rate_teacher and candidate_profs:
                    p = self.rng.choice(candidate_profs)
                else:
                    p = old_p

                # Elegir timeslot
                free_slots_p = prof_free_slots.get(p, [])
                if free_slots_p and self.rng.random() < rate_slot:
                    day, period = self.rng.choice(free_slots_p)
                else:
                    day, period = (old_day, old_period)
                    if self.rng.random() < rate_slot:
                        day, period = self.rng.choice(timeslots)

                # Verificar solape grupo (usar snapshot actualizado)
                other_group_tasks = [tt for tt in occ_group[(group, day, period)] if tt != t]
                if other_group_tasks:
                    continue

                # Elegir aula
                rooms_candidates = valid_rooms_for[t]
                free_rooms = [
                    rr for rr in rooms_candidates
                    if (len(occ_room[(rr, day, period)]) == 0) or
                       (len(occ_room[(rr, day, period)]) == 1 and t in occ_room[(rr, day, period)])
                ]

                if free_rooms and self.rng.random() < rate_room:
                    r = self.rng.choice(free_rooms)
                else:
                    r = self.rng.choice(rooms_candidates)

                # Validaciones básicas
                if subj not in prof_subject.get(p, []):
                    continue
                if rooms_df.loc[r, "capacity"] < n_est:
                    continue

                # Chequear horas semanales / diarias con los contadores
                weekly_new = prof_total[p] + (0 if p == old_p else 1)
                weekly_old_p = prof_total[old_p] - (1 if p != old_p else 0)
                if weekly_new > prof_max_hours[p]:
                    continue
                if weekly_old_p < 0:
                    weekly_old_p = 0

                daily_new = prof_daily[(p, day)] + (0 if (p == old_p and day == old_day) else 1)
                daily_old = prof_daily[(old_p, old_day)] - (1 if (p != old_p or day != old_day) else 0)
                if daily_new > Hmax_prof_day[(p, day)]:
                    continue
                if daily_old < 0:
                    daily_old = 0

                # Chequear solape profe
                other_prof_tasks = [tt for tt in occ_prof[(p, day, period)] if tt != t]
                if other_prof_tasks:
                    continue

                # === ACEPTAMOS EL NUEVO SITIO ===
                # Actualizar contadores y ocupaciones
                if p != old_p:
                    prof_total[old_p] -= 1
                    prof_total[p]     += 1
                if (day, old_day, p, old_p) is not None:  # solo para dejar claro que actualizamos
                    prof_daily[(old_p, old_day)] = max(0, prof_daily[(old_p, old_day)] - 1)
                    prof_daily[(p, day)]         += 1

                # Actualizar occ_prof
                if t in occ_prof[(old_p, old_day, old_period)]:
                    occ_prof[(old_p, old_day, old_period)].remove(t)
                occ_prof[(p, day, period)].append(t)

                # Actualizar occ_group
                if t in occ_group[(group, old_day, old_period)]:
                    occ_group[(group, old_day, old_period)].remove(t)
                occ_group[(group, day, period)].append(t)

                # Actualizar occ_room
                if t in occ_room[(old_r, old_day, old_period)]:
                    occ_room[(old_r, old_day, old_period)].remove(t)
                occ_room[(r, day, period)].append(t)

                # Guardar nueva asignación
                solucion[t] = (p, (day, period), r)
                break  # dejamos de intentar reparar esta tarea

        # 4) Mutación aleatoria ligera sobre tareas NO conflictivas
        non_conflict_tasks = [t for t in task_ids if t not in conflict_tasks]
        for t in non_conflict_tasks:
            if self.rng.random() < random_mut_prob:
                p, (day, period), r = solucion[t]
                row = tasks_df.loc[t]
                subj_key = normalizar_texto(row["subject"])

                candidate_profs = list(subject_to_teachers.get(subj_key, professors))
                if not candidate_profs:
                    candidate_profs = list(professors)

                if self.rng.random() < rate_teacher and candidate_profs:
                    p = self.rng.choice(candidate_profs)
                if self.rng.random() < rate_slot:
                    day, period = self.rng.choice(timeslots)
                if self.rng.random() < rate_room:
                    r = self.rng.choice(valid_rooms_for[t])

                solucion[t] = (p, (day, period), r)

        return (solucion,)
