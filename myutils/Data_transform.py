import pandas as pd
import re
import os

def data_transform(path_asignaturas, path_aulas, path_clases, path_profesores):
    """
    Transforma, anonimiza y SOBRESCRIBE los archivos CSV originales 
    simulando un departamento de ingeniería.
    """
    
    # 1. Cargar los datos
    df_asig = pd.read_csv(path_asignaturas)
    df_aulas = pd.read_csv(path_aulas)
    df_clases = pd.read_csv(path_clases)
    df_prof = pd.read_csv(path_profesores)

    # Listas de transformación
    GRADOS_ING = [
        "Grado en Ingeniería Informática",
        "Grado en Ingeniería del Software",
        "Grado en Ingeniería Mecánica",
        "Grado en Ingeniería Eléctrica",
        "Grado en Ingeniería Industrial",
        "Grado en Ingeniería de Telecomunicaciones",
        "Grado en Ingeniería Civil",
        "Grado en Ingeniería Aeroespacial",
    ]

    ASIGNATURAS_ING = [
        "Cálculo I", "Cálculo II", "Álgebra Lineal", "Física I",
        "Arquitectura de Computadores", "Programación I", "Programación II",
        "Estadística", "Señales y Sistemas", "Bases de Datos",
        "Sistemas Operativos", "Algoritmos y Estructuras de Datos",
        "Redes de Comunicaciones", "Gestión de Proyectos", "Sistemas de Control",
    ]

    # --- 2. Crear Diccionarios de Mapeo ---

    # Mapeo de Asignaturas
    original_subjects = df_asig['Asignatura'].astype(str).str.strip().tolist()
    subject_mapping = {}
    for i, subj in enumerate(original_subjects):
        if i < len(ASIGNATURAS_ING):
            subject_mapping[subj] = ASIGNATURAS_ING[i]
        else:
            subject_mapping[subj] = f"Asignatura_Optativa_{i+1}"
    
    # Mapeo de Clases (Grados)
    original_classes = df_clases['Clase'].unique()
    class_mapping = {}
    for cls in original_classes:
        match = re.match(r"(\d+)([A-Za-z]*)", str(cls))
        if match:
            level = int(match.group(1))
            suffix = match.group(2)
            degree_idx = (level - 1) % len(GRADOS_ING)
            new_name = GRADOS_ING[degree_idx]
            if suffix:
                new_name = f"{new_name} - Grupo {suffix}"
            class_mapping[cls] = new_name
        else:
            class_mapping[cls] = f"Ingeniería General - {cls}"

    # --- 3. Aplicar Transformaciones ---

    # -- Asignaturas --
    df_asig_anon = df_asig.copy()
    df_asig_anon['Asignatura'] = df_asig_anon['Asignatura'].astype(str).str.strip().map(subject_mapping)

    # -- Clases --
    df_clases_anon = df_clases.copy()
    df_clases_anon.columns = df_clases_anon.columns.str.strip()
    df_clases_anon.rename(columns=subject_mapping, inplace=True)
    df_clases_anon['Clase'] = df_clases_anon['Clase'].map(class_mapping)

    # -- Profesores --
    df_prof_anon = df_prof.copy()
    df_prof_anon.columns = df_prof_anon.columns.str.strip()
    df_prof_anon.rename(columns=subject_mapping, inplace=True)
    # Generar nombres anonimizados
    df_prof_anon['Nombre'] = [f"Profesor {i+1}" for i in range(len(df_prof_anon))]

    # -- Aulas --
    df_aulas_anon = df_aulas.copy() # Sin cambios estructurales, pero la guardamos igual por consistencia

    mappings = {
        "asignaturas": subject_mapping,
        "clases": class_mapping
    }

    # --- 4. GUARDAR CAMBIOS EN LOS PATHS ORIGINALES ---
    # index=False evita que se guarde el índice numérico de pandas como columna nueva
    df_asig_anon.to_csv(path_asignaturas, index=False)
    df_aulas_anon.to_csv(path_aulas, index=False)
    df_clases_anon.to_csv(path_clases, index=False)
    df_prof_anon.to_csv(path_profesores, index=False)

    print(f"Archivos transformados y guardados exitosamente en:\n - {path_asignaturas}\n - {path_aulas}\n - {path_clases}\n - {path_profesores}")

    return df_asig_anon, df_aulas_anon, df_clases_anon, df_prof_anon, mappings