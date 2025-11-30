# dataloader.py
import pandas as pd
from collections import defaultdict

# ==========================================
# FUNCIONES DE AYUDA (LIMPIEZA DE TEXTO)
# ==========================================

def normalizar_texto(texto):
    """
    Convierte cualquier texto a mayúsculas y quita espacios extra.
    Ejemplo: "  Matemáticas " -> "MATEMÁTICAS"
    """
    return str(texto).strip().upper()

# ==========================================
# 1. CARGA DE DATOS
# ==========================================

def cargar_archivos_csv(asignaturas_path, aulas_path, clases_path, profesores_path):
    """
    Lee los 4 archivos CSV necesarios.
    Asegúrate de que los archivos estén en la misma carpeta.
    """
    asignaturas_df = pd.read_csv(asignaturas_path, encoding='utf-8-sig')
    aulas_df       = pd.read_csv(aulas_path,       encoding='utf-8-sig')
    clases_df      = pd.read_csv(clases_path,      encoding='utf-8-sig')
    profesores_df  = pd.read_csv(profesores_path,  encoding='utf-8-sig')
    
    return asignaturas_df, aulas_df, clases_df, profesores_df

# ==========================================
# 2. DEFINIR TIEMPO (DÍAS Y HORAS)
# ==========================================

def obtener_estructura_tiempo():
    """
    Define los días de la semana y las franjas horarias (1 a 10).
    """
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    franjas = list(range(1, 11)) # De 1 a 10
    
    # Creamos una lista de tuplas: [('Lunes', 1), ('Lunes', 2)...]
    timeslots = []
    for dia in dias:
        for franja in franjas:
            timeslots.append((dia, franja))
            
    return dias, franjas, timeslots

# ==========================================
# 3. PROCESAR AULAS
# ==========================================

def procesar_aulas(aulas_df):
    """
    Desglosa las aulas. Si una fila dice "3 Laboratorios", 
    crea 3 filas individuales: Laboratorio_1, Laboratorio_2, etc.
    """
    lista_aulas = []
    
    for _, fila in aulas_df.iterrows():
        cantidad = int(fila["Nº de aulas"])
        tipo_aula = fila['Tipo de aula']
        
        for i in range(cantidad):
            lista_aulas.append({
                "room_id":  f"{tipo_aula}_{i+1}", # ID único
                "type":     tipo_aula,
                "type_norm": normalizar_texto(tipo_aula), # Para comparar fácil
                "capacity": int(fila["Capacidad"]),
                "floor":    int(fila["Planta"])
            })
            
    # Convertimos la lista en un DataFrame y usamos room_id como índice
    rooms_df = pd.DataFrame(lista_aulas).set_index('room_id')
    return rooms_df

# ==========================================
# 4. PROCESAR TAREAS (CLASES INDIVIDUALES)
# ==========================================

def procesar_tareas(clases_df, asignaturas_df):
    """
    Convierte la tabla de Clases (donde dice cuántas horas totales hay)
    en una lista de tareas individuales de 1 hora.
    """
    lista_tareas = []
    
    # Recorremos cada grupo (cada fila del excel Clases)
    for _, fila in clases_df.iterrows():
        grupo = fila["Clase"]
        
        # Obtenemos número de alumnos, si está vacío ponemos 0
        n_alumnos = fila["Numero de alumnos"]
        if pd.isna(n_alumnos):
            n_alumnos = 0
        else:
            n_alumnos = int(n_alumnos)
            
        # Para este grupo, miramos todas las asignaturas posibles
        for asig in asignaturas_df["Asignatura"]:
            
            # Obtenemos cuántas horas tiene esa asignatura (si no existe, es 0)
            horas_totales = int(fila.get(asig, 0))
            
            # Creamos una tarea por cada hora necesaria
            for hora in range(horas_totales):
                lista_tareas.append({
                    "task_id":    f"{grupo}_{asig}_{hora}", # ID único: 1A_Mates_0
                    "group":      grupo,
                    "subject":    asig,
                    "n_students": n_alumnos
                })
                
    tasks_df = pd.DataFrame(lista_tareas).set_index("task_id")
    return tasks_df

# ==========================================
# 5. PROCESAR PROFESORES
# ==========================================

def procesar_profesores(profesores_df, asignaturas_df):
    """
    Analiza qué profesor puede dar qué asignatura y sus horas máximas.
    Devuelve diccionarios fáciles de consultar.
    """
    # Limpiamos nombres de columnas para evitar errores por espacios
    df = profesores_df.copy()
    df.columns = df.columns.str.strip()
    
    # Buscamos dinámicamente la columna que tenga "nombre" y la que tenga "hora"
    col_nombre = next(c for c in df.columns if "nombre" in c.lower())
    col_horas  = next(c for c in df.columns if "hora"   in c.lower())
    
    # Catálogo para buscar nombres normalizados: { "MATEMATICAS": "Matemáticas" }
    catalogo_asignaturas = {normalizar_texto(s): s for s in asignaturas_df["Asignatura"]}

    # Diccionarios de salida
    prof_subject = {}      # {Profesor -> [Lista Nombres Reales Asignaturas]}
    prof_subject_norm = {} # {Profesor -> {CONJUNTO ASIGNATURAS NORMALIZADAS}}
    prof_max_hours = {}    # {Profesor -> Horas Máximas}
    subject_to_teachers = defaultdict(list) # {ASIG_NORMALIZADA -> [Lista Profesores]}

    for _, fila in df.iterrows():
        nombre_prof = str(fila[col_nombre]).strip()
        prof_max_hours[nombre_prof] = int(fila[col_horas])
        
        pueden_dar_real = []
        pueden_dar_norm = set()
        
        # Revisamos cada asignatura del catálogo
        for asig_norm, asig_real in catalogo_asignaturas.items():
            columna_en_excel = asig_real.strip()
            
            # Si la columna existe en el excel de profes y dice "SI"
            if columna_en_excel in df.columns:
                valor = str(fila.get(columna_en_excel, "")).strip().upper()
                if valor == "SI":
                    pueden_dar_real.append(asig_real)
                    pueden_dar_norm.add(asig_norm)
                    subject_to_teachers[asig_norm].append(nombre_prof)
        
        prof_subject[nombre_prof] = pueden_dar_real
        prof_subject_norm[nombre_prof] = pueden_dar_norm
        
    return prof_subject, prof_subject_norm, prof_max_hours, subject_to_teachers

# ==========================================
# 6. REGLAS Y RESTRICCIONES (AULAS Y LÍMITES)
# ==========================================

def construir_aulas_validas(tasks_df, rooms_df, asignaturas_df):
    """
    Determina qué aulas sirven para cada tarea basándose en:
    1. El tipo de aula requerido por la asignatura.
    2. La capacidad (que quepan los alumnos).
    """
    # Paso 1: Mapa de requerimientos {MATEMATICAS -> 'Aula Normal', FISICA -> 'Laboratorio'}
    mapa_requerimientos = {}
    df_asig = asignaturas_df.copy()
    
    # Buscar columnas clave
    col_asig_nombre = next(c for c in df_asig.columns if "asignatura" in c.lower())
    col_requisito   = next(c for c in df_asig.columns if "espacio"    in c.lower())
    
    for _, fila in df_asig.iterrows():
        clave = normalizar_texto(fila[col_asig_nombre])
        valor = str(fila[col_requisito]).strip().upper()
        
        if valor in ["", "NO", "N/A", "NA", "NAN"]:
            mapa_requerimientos[clave] = None # Cualquier aula vale (si cabe gente)
        else:
            mapa_requerimientos[clave] = normalizar_texto(valor)

    # Paso 2: Agrupar IDs de aulas por su tipo para buscar rápido
    aulas_por_tipo = defaultdict(list)
    todas_las_aulas = rooms_df.index.tolist()
    
    for id_aula, fila in rooms_df.iterrows():
        tipo_norm = fila["type_norm"]
        aulas_por_tipo[tipo_norm].append(id_aula)

    # Paso 3: Asignar aulas válidas a cada tarea
    valid_rooms = {}
    
    for id_tarea, fila in tasks_df.iterrows():
        asig_norm = normalizar_texto(fila["subject"])
        n_alumnos = int(fila["n_students"])
        
        # ¿Qué tipo de aula necesita?
        tipo_req = mapa_requerimientos.get(asig_norm)
        
        # Seleccionar el grupo de aulas candidatas
        if tipo_req:
            candidatas = aulas_por_tipo.get(tipo_req, []) # Aulas del tipo especifico
            if not candidatas:
                candidatas = todas_las_aulas # Fallback: si no hay de ese tipo, usa todas
        else:
            candidatas = todas_las_aulas # Si no requiere tipo, todas valen
            
        # Filtrar por capacidad (Alumnos <= Capacidad Aula)
        candidatas_capacidad = []
        for aula_id in candidatas:
            cap_aula = rooms_df.loc[aula_id, "capacity"]
            if cap_aula >= n_alumnos:
                candidatas_capacidad.append(aula_id)
        
        # Si no hay ninguna con capacidad suficiente, devolvemos las del tipo correcto (aunque no quepan)
        # para evitar errores vacíos, pero lo ideal es que quepan.
        if candidatas_capacidad:
            valid_rooms[id_tarea] = candidatas_capacidad
        else:
            valid_rooms[id_tarea] = candidatas
            
    return valid_rooms

def construir_limites_diarios(tasks_df, prof_max_hours, dias, franjas):
    """
    Define cuántas horas máximas puede haber por día para profes y asignaturas.
    """
    # 1. Límite asignatura/día (muy generoso para no bloquear)
    limite_asignatura_dia = {}
    tope_generoso = len(franjas) * 10 # Un número alto
    
    mis_asignaturas = tasks_df["subject"].unique()
    for asig in mis_asignaturas:
        for dia in dias:
            limite_asignatura_dia[(asig, dia)] = tope_generoso
            
    # 2. Límite profesor/día (para repartir la carga semanal)
    limite_profe_dia = {}
    num_dias = len(dias)
    min_horas_dia = 3
    max_horas_dia = 7
    
    for profe, horas_semanales in prof_max_hours.items():
        # Cálculo simple: horas semanales entre 5 dias, redondeado hacia arriba
        horas_dia = (horas_semanales + num_dias - 1) // num_dias
        
        # Ajustes para que no sea ni muy poco ni demasiado
        horas_dia = max(min_horas_dia, horas_dia)
        horas_dia = min(max_horas_dia, horas_dia)
        
        # Nunca puede ser mayor que sus horas totales
        horas_dia = min(horas_dia, horas_semanales)
        
        for dia in dias:
            limite_profe_dia[(profe, dia)] = horas_dia
            
    return limite_asignatura_dia, limite_profe_dia

# ==========================================
# FUNCIÓN PRINCIPAL ("WRAPPER")
# ==========================================

def cargar_todo_el_sistema(asignaturas_path='data/Asignaturas.csv', aulas_path='data/Aulas.csv',
                            clases_path='data/Clases.csv', profesores_path='data/Profesores.csv'):
    """
    Ejecuta todo el proceso en orden y devuelve un diccionario 'data' 
    con todo lo necesario para el algoritmo genético.
    """
    print("--- Iniciando Dataloader ---")
    
    # 1. Cargar CSVs
    asig_df, aulas_df, clases_df, profes_df = cargar_archivos_csv(asignaturas_path, aulas_path, clases_path, profesores_path)
    print(f"Datos cargados: {len(asig_df)} asig, {len(aulas_df)} tipos aula, {len(clases_df)} clases, {len(profes_df)} profes.")
    
    # 2. Tiempo
    dias, franjas, timeslots = obtener_estructura_tiempo()
    
    # 3. Aulas expandidas
    rooms_df = procesar_aulas(aulas_df)
    
    # 4. Tareas (Tasks)
    tasks_df = procesar_tareas(clases_df, asig_df)
    print(f"Total de tareas (horas lectivas) a programar: {len(tasks_df)}")
    
    # 5. Profesores
    prof_subj, prof_subj_norm, prof_max_h, subj_to_prof = procesar_profesores(profes_df, asig_df)
    
    # 6. Restricciones (Valid rooms y límites)
    valid_rooms = construir_aulas_validas(tasks_df, rooms_df, asig_df)
    h_max_asig, h_max_prof = construir_limites_diarios(tasks_df, prof_max_h, dias, franjas)
    
    # Empaquetamos todo en un diccionario
    data = {
        "days": dias,
        "periods": franjas,
        "timeslots": timeslots,
        "rooms_df": rooms_df,
        "tasks_df": tasks_df,
        "prof_subject": prof_subj,           # Diccionario simple
        "prof_subject_norm": prof_subj_norm, # Diccionario con sets normalizados
        "prof_max_hours": prof_max_h,
        "subject_to_teachers": subj_to_prof, # Quién da qué
        "valid_rooms": valid_rooms,          # Qué aulas valen para cada tarea
        "Hmax_subject_day": h_max_asig,
        "Hmax_prof_day": h_max_prof,
        "task_ids": list(tasks_df.index),    # Lista simple de IDs de tareas
        "professors": list(prof_subj.keys()) # Lista simple de nombres de profes
    }
    
    print("--- Dataloader Finalizado Correctamente ---")
    return data
