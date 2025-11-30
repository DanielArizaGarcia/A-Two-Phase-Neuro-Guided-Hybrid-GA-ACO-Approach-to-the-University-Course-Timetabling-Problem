import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import random
from copy import deepcopy
import pickle
import os


class NeuroOptimizador:
    def __init__(
        self,
        datos,
        aco_helper,
        evaluator,
        validador_factible,
        ruta_pesos="pesos_neuroopt.pkl",
        cargar_pesos=False,
        batch_size=32,
        max_memory=300,
        mlp_params=None
    ):
        """
        datos: diccionario con, al menos:
            - 'tasks_df'
            - 'rooms_df'
            - 'days'
            - 'prof_subject', 'prof_max_hours', 'valid_rooms',
              'Hmax_subject_day', 'Hmax_prof_day' (para es_factible)
        evaluator: objeto/función con método .cost(sol)
        validador_factible: es_factible(sol, tasks_df, prof_subject, prof_max_hours,
                                         valid_rooms, rooms_df, Hmax_subject_day, Hmax_prof_day)
        """
        default_mlp_params = dict(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.01,
            alpha=0.0001,
            random_state=42
        )
        if mlp_params is not None:
            default_mlp_params.update(mlp_params)
        self.datos = datos
        self.aco = aco_helper
        self.evaluator = evaluator
        self.es_factible_fn = validador_factible



        # --- CEREBRO: Perceptrón Multicapa ---
        self.model = MLPRegressor(**default_mlp_params)
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Memoria de experiencias
        self.memory = []
        self.batch_size = batch_size
        self.max_memory = max_memory

        # Mejor solución factible conocida
        self.mejor_sol_factible = None
        self.mejor_coste_factible = np.inf

        # Gestión de pesos
        self.ruta_pesos = ruta_pesos
        if cargar_pesos and os.path.exists(self.ruta_pesos):
            self._cargar_pesos()

    # ==========================================================
    #  Conversión de solución a DataFrame de contexto
    # ==========================================================
    def _convertir_sol_a_df(self, sol):
        """
        sol: dict {task_id: (prof, (day, period), room)}
        -> DataFrame con columnas:
        ['task_id','professor','group','day','period','room','planta']
        """
        rows = []
        tasks_df = self.datos['tasks_df']
        rooms_df = self.datos['rooms_df']

        for task_id, (p, (d, h), r) in sol.items():
            grupo = tasks_df.loc[task_id, 'group'] if 'group' in tasks_df.columns else None
            planta = rooms_df.loc[r, 'planta'] if r in rooms_df.index and 'planta' in rooms_df.columns else 0

            rows.append({
                'task_id': task_id,
                'professor': p,
                'group': grupo,
                'day': d,
                'period': h,
                'room': r,
                'planta': planta
            })

        df = pd.DataFrame(rows)
        return df

    # ==========================================================
    #  Función maestra de features (C1–C18)
    # ==========================================================
    def _extraer_features_complejas(self, sol, task_id, df_contexto=None):
        """
        Vector numérico que representa el estado de esa tarea (basado en C1–C18).
        """
        # Estado de la tarea
        p, (d, h), r = sol[task_id]

        # Contexto global
        if df_contexto is None:
            df_contexto = self._convertir_sol_a_df(sol)

        tasks_df = self.datos['tasks_df']
        rooms_df = self.datos['rooms_df']

        # Clases del profesor / grupo ese día
        classes_prof_day = df_contexto[
            (df_contexto['professor'] == p) &
            (df_contexto['day'] == d)
        ]

        grupo_tarea = tasks_df.loc[task_id, 'group'] if 'group' in tasks_df.columns else None
        if grupo_tarea is not None:
            classes_group_day = df_contexto[
                (df_contexto['group'] == grupo_tarea) &
                (df_contexto['day'] == d)
            ]
        else:
            classes_group_day = pd.DataFrame(columns=df_contexto.columns)

        features = []

        # [Time Info] Normalizados
        days_list = self.datos['days']
        d_idx = days_list.index(d) if d in days_list else 0
        features.append(d_idx / max(len(days_list), 1))  # Día normalizado
        features.append(h / 10.0)                        # Hora normalizada (asumiendo 1–10)

        # [C1] Preferencia mañana vs tarde
        features.append(1.0 if h <= 5 else 0.0)

        # [C2 & C4] Huecos PROFESOR
        periods_prof = set(classes_prof_day['period'])
        is_isolated_prof = (
            (h - 1 not in periods_prof) and
            (h + 1 not in periods_prof) and
            (len(periods_prof) > 1)
        )
        features.append(1.0 if is_isolated_prof else 0.0)

        # Huecos GRUPO (opcional)
        periods_group = set(classes_group_day['period'])
        is_isolated_group = (
            (h - 1 not in periods_group) and
            (h + 1 not in periods_group) and
            (len(periods_group) > 1)
        )
        features.append(1.0 if is_isolated_group else 0.0)

        # [C5] Carga total del profesor
        carga_prof = len(df_contexto[df_contexto['professor'] == p])
        features.append(carga_prof / 20.0)  # normalización aproximada

        # [C8 & C14] Coherencia de planta (profesor)
        planta_actual = rooms_df.loc[r, 'planta'] if r in rooms_df.index and 'planta' in rooms_df.columns else 0
        plantas_prof = classes_prof_day['planta'].unique()
        mezcla_plantas = 1.0 if len(plantas_prof) > 1 else 0.0
        if len(plantas_prof) > 0:
            moda_planta = pd.Series(plantas_prof).mode()[0]
            planta_distinta = 1.0 if planta_actual != moda_planta else 0.0
        else:
            planta_distinta = 0.0
        features.append(mezcla_plantas)
        features.append(planta_distinta)

        # [C11] Tardes aisladas
        is_afternoon = h > 5
        count_afternoon = classes_prof_day[classes_prof_day['period'] > 5].shape[0]
        features.append(1.0 if is_afternoon and count_afternoon == 1 else 0.0)

        # [C12 & C13] Carga diaria profe y grupo
        load_day_prof = len(classes_prof_day)
        load_day_group = len(classes_group_day)
        features.append(load_day_prof / 8.0)   # normalización aprox.
        features.append(load_day_group / 8.0)

        # [C15] Franjas malas
        bad_slots = [6, 7, 8, 9, 10]
        features.append(1.0 if h in bad_slots else 0.0)

        # [C18] Saturación del aula
        room_usage = len(df_contexto[df_contexto['room'] == r])
        features.append(room_usage / 40.0)  # normalización aprox.

        return np.array(features, dtype=float)

    def _extraer_features_tarea(self, sol, task_id, df_contexto=None):
        """Wrapper para mantener el nombre 'oficial'."""
        return self._extraer_features_complejas(sol, task_id, df_contexto=df_contexto)

    # ==========================================================
    #  Memoria y entrenamiento de la red
    # ==========================================================
    def registrar_experiencia(self, sol, task_id, target):
        """
        target = recompensa / valor que quieres que aprenda el MLP.
        """
        experiencia = (deepcopy(sol), task_id, float(target))
        self.memory.append(experiencia)
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]

    def _preparar_batch_red(self, experiencias):
        X, y = [], []
        for sol, task_id, target in experiencias:
            df_contexto = self._convertir_sol_a_df(sol)
            feats = self._extraer_features_tarea(sol, task_id, df_contexto=df_contexto)
            X.append(feats)
            y.append(target)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        return X, y

    def _actualizar_red(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        X, y = self._preparar_batch_red(batch)

        if not self.is_fitted:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y)

    # ==========================================================
    #  Predicción de calidad
    # ==========================================================
    def predecir_calidad_tarea(self, sol, task_id):
        df_contexto = self._convertir_sol_a_df(sol)
        x = self._extraer_features_tarea(sol, task_id, df_contexto=df_contexto).reshape(1, -1)

        if not self.is_fitted:
            return 0.0

        x_scaled = self.scaler.transform(x)
        pred = self.model.predict(x_scaled)[0]
        return float(pred)

    # ==========================================================
    #  Factibilidad (wrapper) — aquí se arregla lo del 83
    # ==========================================================
    def _es_factible(self, sol):
        """
        Usa es_factible(...) pero interpretando correctamente:
        - bool  -> se respeta
        - int   -> 0 = factible, >0 = no factible
        - dict  -> si trae 'total_violations', se usa
        """
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

        # Caso 1: ya es booleano
        if isinstance(res, bool):
            return res

        # Caso 2: entero = nº de violaciones
        if isinstance(res, (int, np.integer)):
            return (res == 0)

        # Caso 3: dict con info
        if isinstance(res, dict):
            if 'total_violations' in res:
                return (res['total_violations'] == 0)

        # Fallback
        return bool(res)

    # ==========================================================
    #  Gestión de mejor solución factible global
    # ==========================================================
    def _actualizar_mejor_factible(self, sol, coste=None):
        """
        Si 'sol' es factible y mejora el coste, actualiza el mejor conocido.
        """
        if not self._es_factible(sol):
            return

        if coste is None:
            coste = self.evaluator.cost(sol)

        if coste < self.mejor_coste_factible:
            self.mejor_coste_factible = coste
            self.mejor_sol_factible = deepcopy(sol)

    def get_mejor_solucion_factible(self):
        """
        Devuelve la mejor solución factible conocida (o None si no hay).
        """
        return deepcopy(self.mejor_sol_factible)

    # ==========================================================
    #  Movimiento local guiado por la red
    # ==========================================================
    def _vecino_basico(self, sol, task_id):
        """
        Genera un vecino modificando día, hora y aula de una tarea,
        respetando las aulas válidas definidas en datos['valid_rooms'].
        """
        new_sol = deepcopy(sol)
        p, (d, h), r = new_sol[task_id]

        days = self.datos['days']
        rooms_ids = list(self.datos['rooms_df'].index)

        # Horas: asumimos 1..10 si no hay info
        max_period = 10
        if 'max_period' in self.datos:
            max_period = int(self.datos['max_period'])

        # Aulas válidas para esta tarea (si existen)
        if 'valid_rooms' in self.datos and task_id in self.datos['valid_rooms']:
            rooms_valid = list(self.datos['valid_rooms'][task_id])
            if len(rooms_valid) == 0:
                rooms_valid = rooms_ids
        else:
            rooms_valid = rooms_ids

        new_d = random.choice(days)
        new_h = random.randint(1, max_period)
        new_r = random.choice(rooms_valid)

        new_sol[task_id] = (p, (new_d, new_h), new_r)
        return new_sol

    def entrenar_paso(self, sol_actual, n_intentos=20, exploracion=0.3):
        """
        Un paso de búsqueda local:
        - Explora vecinos de sol_actual.
        - Actualiza SIEMPRE la mejor solución factible global.
        - Devuelve SIEMPRE la mejor solución factible conocida (si existe).
        """
        coste_actual = self.evaluator.cost(sol_actual)

        # Actualizamos mejor factible con la solución actual (si lo es)
        self._actualizar_mejor_factible(sol_actual, coste=coste_actual)

        tasks_ids = list(self.datos['tasks_df'].index)

        for _ in range(n_intentos):
            # 1) Elegir tarea a mover
            if self.is_fitted and random.random() > exploracion:
                # Exploit: elegir la peor según la red entre una muestra pequeña
                muestra = random.sample(tasks_ids, min(10, len(tasks_ids)))
                pred_scores = []
                for t_id in muestra:
                    score = self.predecir_calidad_tarea(sol_actual, t_id)
                    pred_scores.append((score, t_id))
                # Suponemos que score alto = peor (ajusta si decides lo contrario)
                _, task_id = max(pred_scores, key=lambda x: x[0])
            else:
                # Explore: tarea al azar
                task_id = random.choice(tasks_ids)

            # 2) Generar vecino
            candidato = self._vecino_basico(sol_actual, task_id)

            # 3) Comprobar factibilidad del vecino (ya usando el fix de 0/83)
            if not self._es_factible(candidato):
                # Penalizamos ese movimiento en la memoria (reward negativo)
                self.registrar_experiencia(sol_actual, task_id, target=-1.0)
                continue

            # 4) Calcular coste del vecino factible
            coste_candidato = self.evaluator.cost(candidato)

            # Reward = mejora respecto a sol_actual (puedes cambiar a respecto a mejor_factible si quieres)
            reward = coste_actual - coste_candidato
            self.registrar_experiencia(sol_actual, task_id, target=reward)

            # 5) Actualizar mejor factible global
            self._actualizar_mejor_factible(candidato, coste=coste_candidato)

        # 6) Actualizar la red tras este paso
        self._actualizar_red()

        # 7) Devolver SIEMPRE la mejor factible conocida (si la hay)
        if self.mejor_sol_factible is not None:
            return deepcopy(self.mejor_sol_factible), self.mejor_coste_factible
        else:
            # No se ha encontrado ninguna solución factible todavía
            print("[NeuroOpt] WARNING: aún no se ha encontrado ninguna solución factible. Devuelvo sol_actual tal cual.")
            return sol_actual, coste_actual

    # ==========================================================
    #  Gestión de pesos (guardar / cargar)
    # ==========================================================
    def _guardar_pesos(self):
        payload = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        with open(self.ruta_pesos, 'wb') as f:
            pickle.dump(payload, f)

    def _cargar_pesos(self):
        with open(self.ruta_pesos, 'rb') as f:
            payload = pickle.load(f)
        self.model = payload['model']
        self.scaler = payload['scaler']
        self.is_fitted = payload['is_fitted']

    # Métodos públicos para que tu script los use
    def guardar_cerebro(self, ruta=None):
        if ruta is not None:
            self.ruta_pesos = ruta
        self._guardar_pesos()

    def cargar_cerebro(self, ruta=None):
        if ruta is not None:
            self.ruta_pesos = ruta
        if os.path.exists(self.ruta_pesos):
            self._cargar_pesos()
        else:
            print(f"[NeuroOpt] No se encontró el fichero de pesos: {self.ruta_pesos}")
