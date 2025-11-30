def es_factible(sol,
                tasks_df,
                prof_subject,
                prof_max_hours,
                valid_rooms_for,
                rooms_df,
                Hmax_subject_day,
                Hmax_prof_day):
    """
    Comprueba las restricciones R1–R11.

    Nuevo contrato:
    - Devuelve True si NO hay violaciones.
    - Devuelve un entero > 0 con el número de violaciones si las hay.
    """
    from collections import defaultdict

    # --- Contador de violaciones ---
    violations = 0

    # R1: cada tarea aparece exactamente una vez
    sol_tasks   = set(sol.keys())
    real_tasks  = set(tasks_df.index)
    if sol_tasks != real_tasks:
        # Contamos cuántas faltan y cuántas sobran
        missing = len(real_tasks - sol_tasks)
        extra   = len(sol_tasks - real_tasks)
        return missing + extra  # aquí sí podemos devolver ya (es grave)

    prof_total   = defaultdict(int)
    prof_daily   = defaultdict(int)
    subj_daily   = defaultdict(int)
    occ_prof     = set()
    occ_group    = set()
    occ_room     = set()
    occ_subj_ts  = set()

    for t, (p, (day, period), r) in sol.items():
        grupo = tasks_df.loc[t, "group"]
        subj  = tasks_df.loc[t, "subject"]
        n_est = tasks_df.loc[t, "n_students"]

        # R11: profesor capacitado para la asignatura
        if subj not in prof_subject.get(p, []):
            violations += 1

        # R7: aula del tipo requerido
        if r not in valid_rooms_for[t]:
            violations += 1

        # R10: capacidad del aula ≥ número de alumnos
        if rooms_df.loc[r, "capacity"] < n_est:
            violations += 1

        # R2: no superar horas semanales por profesor
        prof_total[p] += 1
        if prof_total[p] > prof_max_hours[p]:
            violations += 1

        # R9: no superar horas diarias por profesor
        prof_daily[(p, day)] += 1
        if prof_daily[(p, day)] > Hmax_prof_day[(p, day)]:
            violations += 1

        # R8: no superar horas diarias por asignatura
        subj_daily[(subj, day)] += 1
        if subj_daily[(subj, day)] > Hmax_subject_day[(subj, day)]:
            violations += 1

        # R3: un profesor no imparte dos clases simultáneas
        if (p, day, period) in occ_prof:
            violations += 1
        else:
            occ_prof.add((p, day, period))

        # R4: un grupo no tiene dos clases simultáneas
        if (grupo, day, period) in occ_group:
            violations += 1
        else:
            occ_group.add((grupo, day, period))

        # R6: un aula no tiene dos clases simultáneas
        if (r, day, period) in occ_room:
            violations += 1
        else:
            occ_room.add((r, day, period))

        # R5: misma asignatura y grupo no en varias aulas a la vez
        key_subj_group_ts = ((grupo, subj), day, period)
        if key_subj_group_ts in occ_subj_ts:
            violations += 1
        else:
            occ_subj_ts.add(key_subj_group_ts)

    # --- Resultado final ---
    if violations == 0:
        return True
    else:
        return violations
