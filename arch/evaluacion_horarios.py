# evaluacion_horarios.py
# ======================
# Envoltorio OO de la evaluación:
#  - convertir sol -> DataFrame
#  - evaluar_horario (C1–C18, coste)
#  - función cargar_y_evaluar

from typing import Dict, Tuple
import pandas as pd
from myutils.datos_horarios import TimetablingData
from pathlib import Path
import itertools


def evaluar_horario(schedule_df, H_ideal=1, Us=3):
    """
    Calcula las penalizaciones C1–C18 y el coste total de un horario dado.

    Parámetros:
    - schedule_df: DataFrame con columnas ['task_id','group','subject','professor','day','period','room']
    - H_ideal:     Umbral ideal de clases por asignatura/día (B6), por defecto 1
    - Us:          Umbral máximo de separación en días entre sesiones mismas (B16), por defecto 1
    
    Devuelve:
    - pen_series:  pd.Series con índices ['C1',..., 'C18']
    - total_cost:  float con el coste ponderado
    """

    # ========= Helpers internos =========
    def _load_rooms_catalog():
        """
        Lee data/Aulas.csv y genera un DataFrame con la planta de cada aula (room_id).
        Si el archivo no existe o falla, devuelve un DataFrame vacío con la columna esperada.
        """
        try:
            aulas_path = Path(__file__).resolve().parents[1] / "data" / "Aulas.csv"
        except NameError:
            # Por si __file__ no está definido (entorno interactivo)
            return pd.DataFrame(columns=['room_id', 'planta']).set_index('room_id')

        if not aulas_path.exists():
            return pd.DataFrame(columns=['room_id', 'planta']).set_index('room_id')

        aulas_df = pd.read_csv(aulas_path, encoding='utf-8-sig')
        rooms = []
        for _, row in aulas_df.iterrows():
            tipo = str(row['Tipo de aula']).strip()
            planta = int(row['Planta'])
            for i in range(int(row["Nº de aulas"])):
                rooms.append({"room_id": f"{tipo}_{i+1}", "planta": planta})
        if not rooms:
            return pd.DataFrame(columns=['room_id', 'planta']).set_index('room_id')
        return pd.DataFrame(rooms).set_index('room_id')

    ROOMS_DF = _load_rooms_catalog()

    def _ensure_floor_column(df_in):
        """
        Devuelve una copia del schedule con la columna 'planta'.
        Si ya existe la columna, solo la normaliza; de lo contrario, la completa usando data/Aulas.csv.
        """
        df = df_in.copy()
        if 'planta' in df.columns:
            df['planta'] = df['planta'].astype(int)
            return df

        if ROOMS_DF.empty:
            df['planta'] = 0
            return df

        merged = df.merge(
            ROOMS_DF[['planta']],
            left_on='room',
            right_index=True,
            how='left'
        )
        if merged['planta'].isna().any():
            default_floor = int(ROOMS_DF['planta'].mode().iloc[0])
            merged['planta'] = merged['planta'].fillna(default_floor)
        merged['planta'] = merged['planta'].astype(int)
        return merged

    def _count_holes(plist):
        """Cuenta huecos en una lista de periodos."""
        if len(plist) <= 1:
            return 0
        mn, mx = min(plist), max(plist)
        return (mx - mn + 1) - len(plist)

    # ========= 1) Definición de días, franjas, etc. =========
    days      = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    periods   = list(range(1, 11))
    timeslots = [(d, p) for d in days for p in periods]
    preferred = set(range(1, 6))   # mañanas
    afternoon = set(range(6, 11))  # tardes

    # ========= 2) Garantizar columna 'planta' =========
    df = _ensure_floor_column(schedule_df)

    # ========= 3) C1–C10 =========
    C1 = df[~df['period'].isin(preferred)].shape[0]

    C2 = int(df.groupby(['professor', 'day'])['period']
             .apply(lambda x: _count_holes(sorted(x))).sum())

    C3 = C1

    C4 = int(df.groupby(['group', 'day'])['period']
             .apply(lambda x: _count_holes(sorted(x))).sum())

    prof_counts = df['professor'].value_counts()
    H = prof_counts.mean()
    C5 = int((prof_counts - H).abs().sum())

    C6 = sum(
        max(0, df[(df['subject'] == subj) & (df['day'] == d)].shape[0] - H_ideal)
        for subj in df['subject'].unique() for d in days
    )

    C7 = int((df.groupby('group')['room'].nunique() - 1).clip(lower=0).sum())

    C8 = int(df.groupby(['professor', 'day'])['planta']
             .nunique().sub(1).clip(lower=0).sum())

    C9 = int(df.groupby(['group', 'day'])['planta']
             .nunique().sub(1).clip(lower=0).sum())

    total_slots  = len(timeslots)
    usage_counts = df['room'].value_counts()
    rooms_catalog = set(ROOMS_DF.index) if not ROOMS_DF.empty else set()
    rooms_catalog.update(df['room'].unique())
    availability  = {r: total_slots - usage_counts.get(r, 0) for r in rooms_catalog}
    C10 = df['room'].map(lambda r: 1 / max(availability.get(r, total_slots), 1)).sum()

    # ========= 4) C11: tardes aisladas =========
    C11_prof = sum(
        1
        for (_, d), grp in df.groupby(['professor', 'day'])
        if grp['period'].isin(afternoon).sum() == 1
    )
    C11_group = sum(
        1
        for (_, d), grp in df.groupby(['group', 'day'])
        if grp['period'].isin(afternoon).sum() == 1
    )
    C11 = C11_prof + C11_group

    # ========= 5) C12: desbalance diario por grupo =========
    C12 = 0
    for _, grp in df.groupby('group'):
        hours = grp['day'].value_counts().reindex(days, fill_value=0)
        for d1, d2 in itertools.combinations(days, 2):
            C12 += abs(int(hours[d1]) - int(hours[d2]))

    # ========= 6) C13: desbalance diario por profesor =========
    C13 = 0
    for _, grp in df.groupby('professor'):
        hours = grp['day'].value_counts().reindex(days, fill_value=0)
        for d1, d2 in itertools.combinations(days, 2):
            C13 += abs(int(hours[d1]) - int(hours[d2]))

    # ========= 7) C14: desplazamientos contiguos de planta =========
    C14 = 0
    for (_, d), grp in df.groupby(['professor', 'day']):
        pf = dict(zip(grp['period'], grp['planta']))
        for t in range(1, 10):
            if t in pf and (t + 1) in pf and pf[t] != pf[t + 1]:
                C14 += 1

    # ========= 8) C15: franjas malas =========
    day_to_idx = {d: i for i, d in enumerate(days)}
    df['slot_id'] = (df['day'].astype(str).map(day_to_idx) * 10 + df['period']).astype(int)
    bad = set(range(6, 11)) | set(range(45, 51))
    C15 = df['slot_id'].isin(bad).sum()

    # ========= 9) C16: separación excesiva entre días por asignatura =========
    counts_sd = df.groupby(['subject', 'day']).size().unstack(fill_value=0)
    C16 = 0
    for subj in counts_sd.index:
        for d1, d2 in itertools.combinations(days, 2):
            c1 = counts_sd.loc[subj, d1]
            c2 = counts_sd.loc[subj, d2]
            dist = day_to_idx[d2] - day_to_idx[d1]
            if c1 >= 1 and c2 >= 1 and dist > Us:
                C16 += (dist - Us)

    # ========= 10) C17: concentración en bloques (plantas) =========
    total_classes = len(df)
    n_blocks = df['planta'].nunique()
    ideal_block = total_classes / n_blocks if n_blocks > 0 else 0
    counts_block = df['planta'].value_counts()
    C17 = int(sum(abs(count - ideal_block) for count in counts_block))

    # ========= 11) C18: saturación de franjas =========
    x = total_classes / total_slots if total_slots > 0 else 0
    counts_t = df['slot_id'].value_counts()
    C18 = sum(abs(counts_t.get(slot, 0) - x) for slot in range(1, total_slots + 1))

    # ========= 12) Series de penalizaciones =========
    penalties = [C1, C2, C3, C4, C5, C6, C7, C8, C9,
                 C10, C11, C12, C13, C14, C15, C16, C17, C18]
    labels = [f"C{i}" for i in range(1, 19)]
    pen_series = pd.Series(penalties, index=labels, dtype=float)

    # ========= 13) Cálculo del coste total (pesos) =========
    importance_weights = {
        "Muy importante": 15,
        "Importante":     10,
        "Deseable":        5,
        "Mejor":           1
    }
    groups = {
        "Muy importante": ["C2", "C4", "C5", "C11", "C12"],
        "Importante":     ["C18", "C17", "C16", "C13", "C10"],
        "Deseable":       ["C6", "C7", "C8", "C9", "C14"],
        "Mejor":          ["C1", "C3", "C15"]
    }

    label_to_weight = {}
    for grp, lbls in groups.items():
        w = importance_weights[grp]
        for lbl in lbls:
            label_to_weight[lbl] = w

    weight_series = pd.Series(label_to_weight)
    total_cost = (pen_series * weight_series).sum()

    return pen_series, float(total_cost)

class TimetablingEvaluator:
    """
    Envuélvelo todo para que sea fácil de usar desde cualquier notebook:

    - solution_to_df(sol)
    - evaluate_schedule_df(df) -> (pen_series, total_cost)
    - cost(sol) -> escalar
    - cargar_y_evaluar(path_horario)
    """

    def __init__(self, data: TimetablingData, H_ideal: int = 1, Us: int = 1):
        self.data = data
        self.H_ideal = H_ideal
        self.Us = Us

    # -----------------------------------------
    # Conversión sol -> DataFrame
    # -----------------------------------------
    def solution_to_df(
        self,
        sol: Dict[str, Tuple[str, Tuple[str, int], str]],
    ) -> pd.DataFrame:
        """
        Convierte un diccionario:
        sol[task_id] = (professor, (day, period), room)
        en un DataFrame con las columnas que espera evaluar_horario.
        """
        rows = []
        for t, (prof, (day, period), room) in sol.items():
            rows.append({
                "task_id":   t,
                "group":     self.data.tasks_df.loc[t, "group"],
                "subject":   self.data.tasks_df.loc[t, "subject"],
                "professor": prof,
                "day":       day,
                "period":    period,
                "room":      room,
            })
        df = pd.DataFrame(rows)
        df = df.astype({
            "task_id":   str,
            "group":     str,
            "subject":   str,
            "professor": str,
            "day":       str,
            "period":    int,
            "room":      str,
        })

        # Añadir planta de aula para coherencia con los horarios cargados
        df = df.merge(
            self.data.rooms_df[["floor"]],
            left_on="room",
            right_index=True,
            how="left",
        )
        return df

    # -----------------------------------------
    # Envoltorio de Evaluacion.evaluar_horario
    # -----------------------------------------
    def evaluate_schedule_df(self, schedule_df: pd.DataFrame):
        """
        Llama a Evaluacion.evaluar_horario(schedule_df, H_ideal, Us)
        y devuelve (pen_series, total_cost).
        """
        pen_series, total_cost = evaluar_horario(
            schedule_df,
            H_ideal=self.H_ideal,
            Us=self.Us,
        )
        return pen_series, float(total_cost)

    def cost(self, sol: Dict[str, Tuple[str, Tuple[str, int], str]]) -> float:
        """
        Devuelve solo el coste escalar de un diccionario sol.
        """
        df = self.solution_to_df(sol)
        _, total_cost = self.evaluate_schedule_df(df)
        return float(total_cost)

    # -----------------------------------------
    # Versión OO de tu cargar_y_evaluar
    # -----------------------------------------
    def cargar_y_evaluar(
        self,
        path_horario: str = "horario_factible.csv",
        encoding: str = "utf-8-sig",
    ):
        """
        Carga un horario desde CSV usando TimetablingData.cargar_horario_csv
        y lo evalúa. Muestra resumen por pantalla y devuelve:

        - schedule_df
        - sol
        - pen_series
        - total_cost
        """
        schedule_df, sol = self.data.cargar_horario_csv(
            path_horario=path_horario,
            encoding=encoding,
        )
        pen_series, total_cost = self.evaluate_schedule_df(schedule_df)

        print("\n==== PENALIZACIONES C1–C18 ====")
        print(pen_series)
        print(f"\nCoste total: {total_cost:.3f}")

        return schedule_df, sol, pen_series, total_cost
