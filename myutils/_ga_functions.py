# arch.py
import random
from collections import defaultdict
from myutils.dataloader import normalizar_texto

# ==========================================
# PARÁMETROS / CONSTANTES
# ==========================================
PENALIZACION_CHOQUE_FISICO = 10
PENALIZACION_MISMA_ASIG    = 5
PUNTUACION_INFINITA        = 10**9


# ==========================================
# FUNCIONES AUXILIARES BÁSICAS
# ==========================================
def calcular_dificultad(id_tarea, datos):
    """
    Calcula cuán difícil es programar una tarea.
    Devuelve una tupla para ordenar.
    Criterio: Menos aulas disponibles, menos profes y más alumnos = Más difícil.
    """
    tasks_df            = datos["tasks_df"]
    valid_rooms         = datos["valid_rooms"]
    subject_to_teachers = datos["subject_to_teachers"]
    professors          = datos["professors"]

    fila      = tasks_df.loc[id_tarea]
    asig      = fila["subject"]
    asig_key  = normalizar_texto(asig)
    n_alumnos = int(fila["n_students"])

    # Contar opciones
    aulas_posibles      = valid_rooms[id_tarea]
    num_opciones_aula   = len(aulas_posibles)
    lista_profes        = subject_to_teachers.get(asig_key, professors)
    num_opciones_profe  = len(lista_profes)

    # Orden ascendente: primero las tareas más limitadas (más “difíciles”)
    return (num_opciones_aula, num_opciones_profe, -n_alumnos)


def _evaluar_conflictos(
    profe, grupo, asig, aula, dia, hora,
    ocupacion_profe, ocupacion_grupo, ocupacion_aula, ocupacion_asig_ts,
    mejor_puntuacion_actual
):
    """
    Calcula la puntuación de conflicto para un (profe, grupo, asig, aula, dia, hora).
    Aplica podas tempranas: si ya supera la mejor puntuación conocida, corta.
    """
    puntos = 0

    # Choques físicos
    if (profe, dia, hora) in ocupacion_profe:
        puntos += PENALIZACION_CHOQUE_FISICO
        if puntos >= mejor_puntuacion_actual:
            return puntos

    if (grupo, dia, hora) in ocupacion_grupo:
        puntos += PENALIZACION_CHOQUE_FISICO
        if puntos >= mejor_puntuacion_actual:
            return puntos

    if (aula, dia, hora) in ocupacion_aula:
        puntos += PENALIZACION_CHOQUE_FISICO
        if puntos >= mejor_puntuacion_actual:
            return puntos

    # Misma asignatura para el mismo grupo a la vez
    if ((grupo, asig), dia, hora) in ocupacion_asig_ts:
        puntos += PENALIZACION_MISMA_ASIG

    return puntos


def _evaluar_solucion(solucion, datos):
    """
    Coste rápido de un horario:
    - Penaliza conflictos duros (profe/grupo/aula/asig en el mismo slot).
    - Penaliza superar horas máximas por profe / día / asignatura.
    Solo se usa para elegir el mejor greedy entre varios intentos.
    """
    tasks_df       = datos["tasks_df"]
    task_ids       = datos["task_ids"]
    prof_max_hours = datos["prof_max_hours"]
    Hmax_prof_day  = datos["Hmax_prof_day"]
    Hmax_subject_day = datos["Hmax_subject_day"]

    # Contadores
    horas_totales_profe = defaultdict(int)
    horas_dia_profe     = defaultdict(int)
    horas_dia_asig      = defaultdict(int)

    # Ocupaciones
    ocupacion_profe   = set()
    ocupacion_grupo   = set()
    ocupacion_aula    = set()
    ocupacion_asig_ts = set()

    coste_total = 0

    for tarea_id in task_ids:
        p, (d, h), r = solucion[tarea_id]
        fila  = tasks_df.loc[tarea_id]
        grupo = fila["group"]
        asig  = fila["subject"]

        # Conflictos duros
        coste_total += _evaluar_conflictos(
            p, grupo, asig, r, d, h,
            ocupacion_profe, ocupacion_grupo, ocupacion_aula, ocupacion_asig_ts,
            PUNTUACION_INFINITA
        )

        # Actualizar ocupaciones
        ocupacion_profe.add((p, d, h))
        ocupacion_grupo.add((grupo, d, h))
        ocupacion_aula.add((r, d, h))
        ocupacion_asig_ts.add(((grupo, asig), d, h))

        # Soft: horas máximas
        horas_totales_profe[p] += 1
        if horas_totales_profe[p] > prof_max_hours.get(p, PUNTUACION_INFINITA):
            coste_total += 1

        horas_dia_profe[(p, d)] += 1
        if horas_dia_profe[(p, d)] > Hmax_prof_day.get((p, d), PUNTUACION_INFINITA):
            coste_total += 1

        horas_dia_asig[(asig, d)] += 1
        if horas_dia_asig[(asig, d)] > Hmax_subject_day.get((asig, d), PUNTUACION_INFINITA):
            coste_total += 1

    return coste_total


# ==========================================
# GREEDY "SIMPLE" (UN INTENTO)
# ==========================================
def _generar_horario_greedy_single(datos):
    """
    Greedy de construcción de un único horario.
    Se centra en minimizar conflictos al construir.
    """
    tasks_df            = datos["tasks_df"]
    task_ids            = datos["task_ids"]
    professors          = datos["professors"]
    timeslots           = datos["timeslots"]
    subject_to_teachers = datos["subject_to_teachers"]
    prof_max_hours      = datos["prof_max_hours"]
    Hmax_prof_day       = datos["Hmax_prof_day"]
    Hmax_subject_day    = datos["Hmax_subject_day"]
    valid_rooms         = datos["valid_rooms"]

    solucion = {}

    # Contadores mientras construimos
    horas_totales_profe = defaultdict(int)
    horas_dia_profe     = defaultdict(int)  # (profe, dia)
    horas_dia_asig      = defaultdict(int)  # (asig, dia)

    # Ocupaciones: (Entidad, Dia, Hora)
    ocupacion_profe   = set()
    ocupacion_grupo   = set()
    ocupacion_aula    = set()
    ocupacion_asig_ts = set()  # ((grupo, asig), dia, hora)

    # Ordenar tareas por dificultad
    tareas_ordenadas = sorted(task_ids, key=lambda t: calcular_dificultad(t, datos))

    for tarea_id in tareas_ordenadas:
        fila      = tasks_df.loc[tarea_id]
        grupo     = fila["group"]
        asig      = fila["subject"]
        asig_norm = normalizar_texto(asig)

        # Candidatos a profesor para la asignatura
        candidatos_profe = subject_to_teachers.get(asig_norm, professors)
        candidatos_profe = list(candidatos_profe)
        random.shuffle(candidatos_profe)

        mejor_eleccion   = None
        mejor_puntuacion = PUNTUACION_INFINITA

        for profe in candidatos_profe:
            # Filtro semanal
            if horas_totales_profe[profe] >= prof_max_hours.get(profe, PUNTUACION_INFINITA):
                continue

            # Shuffle de timeslots para variedad
            huecos_mezclados = list(timeslots)
            random.shuffle(huecos_mezclados)

            for (dia, hora) in huecos_mezclados:
                # Filtro diario profe
                if horas_dia_profe[(profe, dia)] >= Hmax_prof_day.get((profe, dia), PUNTUACION_INFINITA):
                    continue
                # Filtro diario asignatura
                if horas_dia_asig[(asig, dia)] >= Hmax_subject_day.get((asig, dia), PUNTUACION_INFINITA):
                    continue

                aulas_posibles = valid_rooms[tarea_id]
                if not aulas_posibles:
                    continue
                aulas_posibles = list(aulas_posibles)

                # Probar más aulas (mejor calidad, algo más caro)
                if len(aulas_posibles) > 10:
                    aulas_a_probar = random.sample(aulas_posibles, 10)
                else:
                    aulas_a_probar = aulas_posibles

                for aula in aulas_a_probar:
                    puntos_conflicto = _evaluar_conflictos(
                        profe, grupo, asig, aula, dia, hora,
                        ocupacion_profe, ocupacion_grupo, ocupacion_aula, ocupacion_asig_ts,
                        mejor_puntuacion
                    )

                    # Perfecto -> nos quedamos con esto y salimos
                    if puntos_conflicto == 0:
                        mejor_eleccion   = (profe, (dia, hora), aula)
                        mejor_puntuacion = 0
                        break

                    # Menos malo que lo que llevamos
                    if puntos_conflicto < mejor_puntuacion:
                        mejor_puntuacion = puntos_conflicto
                        mejor_eleccion   = (profe, (dia, hora), aula)

                if mejor_puntuacion == 0:
                    break
            if mejor_puntuacion == 0:
                break

        # Asignación final para esta tarea
        if mejor_eleccion:
            p, (d, h), r = mejor_eleccion
        else:
            # Fallback muy raro: algo al azar pero válido en tipo
            p = random.choice(candidatos_profe) if candidatos_profe else random.choice(list(professors))
            d, h = random.choice(list(timeslots))
            aulas_fallback = list(valid_rooms[tarea_id])
            r = random.choice(aulas_fallback)

        solucion[tarea_id] = (p, (d, h), r)

        # Actualizar contadores y ocupaciones
        horas_totales_profe[p]    += 1
        horas_dia_profe[(p, d)]   += 1
        horas_dia_asig[(asig, d)] += 1

        ocupacion_profe.add((p, d, h))
        ocupacion_grupo.add((grupo, d, h))
        ocupacion_aula.add((r, d, h))
        ocupacion_asig_ts.add(((grupo, asig), d, h))

    return solucion


# ==========================================
# GREEDY "MEJORADO" (MULTI-START)
# ==========================================
def generar_horario_greedy(datos, num_intentos=3):
    """
    Genera un horario con un greedy multi-start:
      - Lanza 'num_intentos' construcciones greedy (con aleatoriedad).
      - Evalúa cada solución con un coste rápido.
      - Devuelve el mejor horario encontrado.
    Con esto los individuos iniciales del GA suelen tener muchos menos conflictos.
    """
    mejor_sol   = None
    mejor_coste = PUNTUACION_INFINITA

    for _ in range(num_intentos):
        sol   = _generar_horario_greedy_single(datos)
        coste = _evaluar_solucion(sol, datos)

        if coste < mejor_coste:
            mejor_coste = coste
            mejor_sol   = sol

    return mejor_sol


# ==========================================
# HORARIO ALEATORIO (DIVERSIDAD)
# ==========================================
def generar_horario_aleatorio(datos):
    """
    Genera un horario 100% al azar (útil para diversidad de población en el GA).
    No intenta respetar restricciones; solo devuelve una asignación completa.
    """
    solucion = {}

    task_ids            = datos["task_ids"]
    tasks_df            = datos["tasks_df"]
    subject_to_teachers = datos["subject_to_teachers"]
    professors          = datos["professors"]
    timeslots           = datos["timeslots"]
    valid_rooms         = datos["valid_rooms"]

    for t_id in task_ids:
        fila      = tasks_df.loc[t_id]
        asig_norm = normalizar_texto(fila["subject"])

        # Elegir profe válido para la asignatura
        profes_validos = subject_to_teachers.get(asig_norm, professors)
        profes_validos = list(profes_validos)
        p = random.choice(profes_validos)

        # Elegir hueco y aula
        d, h = random.choice(list(timeslots))
        aulas_posibles = list(valid_rooms[t_id])
        r = random.choice(aulas_posibles)

        solucion[t_id] = (p, (d, h), r)

    return solucion
