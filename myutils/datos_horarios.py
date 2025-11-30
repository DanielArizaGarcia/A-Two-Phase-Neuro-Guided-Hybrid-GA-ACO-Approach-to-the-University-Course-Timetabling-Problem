# datos_horarios.py
# ==================
# Carga de CSV y construcción de:
#  - tasks_df
#  - rooms_df
#  - prof_subject, prof_max_hours
#  - req_type, valid_rooms_for
#  - Hmax_subject_day, Hmax_prof_day
#
# Además, función para cargar un horario factible desde CSV
# y reconstruir el diccionario sol.

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd


class TimetablingData:
    """
    Contiene todos los datos y estructuras necesarias para el problema
    de horarios (UCTP):

    - asignaturas_df, aulas_df, clases_df, profesores_df
    - rooms_df (todas las aulas físicas, con capacidad y planta)
    - tasks_df (todas las tareas grupo-asignatura-hora)
    - prof_subject, prof_max_hours
    - req_type, valid_rooms_for
    - Hmax_subject_day, Hmax_prof_day
    """

    def __init__(
        self,
        asignaturas_df: pd.DataFrame,
        aulas_df: pd.DataFrame,
        clases_df: pd.DataFrame,
        profesores_df: pd.DataFrame,
        days: Optional[List[str]] = None,
        max_subject_hours_per_day: int = 4,
        max_prof_hours_per_day: int = 8,
    ):
        self.asignaturas_df = asignaturas_df.copy()
        self.aulas_df = aulas_df.copy()
        self.clases_df = clases_df.copy()
        self.profesores_df = profesores_df.copy()

        self.days = days or ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

        # ---------- 1) Construir rooms_df ----------
        rooms = []
        for _, row in self.aulas_df.iterrows():
            for i in range(int(row["Nº de aulas"])):
                rooms.append({
                    "room_id": f"{row['Tipo de aula']}_{i+1}",
                    "capacity": int(row["Capacidad"]),
                    "floor":    int(row["Planta"]),
                })
        self.rooms_df = pd.DataFrame(rooms).set_index("room_id")

        # ---------- 2) Construir tasks_df ----------
        tasks = []
        for _, row in self.clases_df.iterrows():
            grp   = row["Clase"]
            n_est = int(row["Numero de alumnos"]) if not pd.isna(row["Numero de alumnos"]) else 0
            for subj in self.asignaturas_df["Asignatura"]:
                horas = int(row.get(subj, 0))
                for s in range(horas):
                    tasks.append({
                        "task_id":    f"{grp}_{subj}_{s}",
                        "group":      grp,
                        "subject":    subj,
                        "n_students": n_est,
                    })
        self.tasks_df = pd.DataFrame(tasks).set_index("task_id")

        # ---------- 3) Profesor -> asignaturas que puede impartir ----------
        self.prof_subject: Dict[str, List[str]] = {
            row["Nombre"]: [
                subj for subj in self.asignaturas_df["Asignatura"] if row.get(subj) == "SI"
            ]
            for _, row in self.profesores_df.iterrows()
        }

        # ---------- 4) Horas máximas semanales por profesor ----------
        self.prof_max_hours: Dict[str, int] = {
            row["Nombre"]: int(row["Nº horas"])
            for _, row in self.profesores_df.iterrows()
        }

        # ---------- 5) Tipo de aula requerida por asignatura ----------
        self.req_type: Dict[str, str] = dict(
            self.asignaturas_df[["Asignatura", "Espacio específico"]].values
        )

        # ---------- 6) Aulas válidas por tarea ----------
        self.valid_rooms_for: Dict[str, List[str]] = {}
        for t, info in self.tasks_df.iterrows():
            subj = info["subject"]
            tp = self.req_type.get(subj, "NO")
            if tp == "NO":
                self.valid_rooms_for[t] = list(self.rooms_df.index)
            else:
                self.valid_rooms_for[t] = [r for r in self.rooms_df.index if r.startswith(tp)]

        # ---------- 7) Límites diarios por asignatura y profesor ----------
        self.Hmax_subject_day = defaultdict(lambda: float("inf"))
        self.Hmax_prof_day    = defaultdict(lambda: float("inf"))

        for subj in self.asignaturas_df["Asignatura"]:
            for d in self.days:
                self.Hmax_subject_day[(subj, d)] = max_subject_hours_per_day

        for p in self.prof_max_hours:
            for d in self.days:
                self.Hmax_prof_day[(p, d)] = max_prof_hours_per_day

    # --------------------------------------------------
    # Métodos de construcción/carga desde CSV
    # --------------------------------------------------
    @classmethod
    def from_csv_folder(
        cls,
        data_folder: str = "data",
        encoding: str = "utf-8-sig",
        **kwargs,
    ) -> "TimetablingData":
        """
        Crea la instancia leyendo:
        - data/Asignaturas.csv
        - data/Aulas.csv
        - data/Clases.csv
        - data/Profesores.csv
        """
        base = Path(data_folder)
        asignaturas_df = pd.read_csv(base / "Asignaturas.csv", encoding=encoding)
        aulas_df       = pd.read_csv(base / "Aulas.csv",       encoding=encoding)
        clases_df      = pd.read_csv(base / "Clases.csv",      encoding=encoding)
        profesores_df  = pd.read_csv(base / "Profesores.csv",  encoding=encoding)
        return cls(asignaturas_df, aulas_df, clases_df, profesores_df, **kwargs)

    # --------------------------------------------------
    # Cargar horario factible desde CSV y reconstruir sol
    # --------------------------------------------------
    def cargar_horario_csv(
        self,
        path_horario: str = "horario_factible.csv",
        encoding: str = "utf-8-sig",
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, Tuple[str, int], str]]]:
        """
        Carga un horario desde CSV y devuelve:
        - schedule_df: DataFrame normalizado con columnas
          ['task_id','group','subject','professor','day','period','room','floor']
        - sol: dict task_id -> (professor, (day, period), room)
        """
        horario_df = pd.read_csv(path_horario, encoding=encoding)

        col_map = {
            "TaskID":     "task_id",
            "Grupo":      "group",
            "Asignatura": "subject",
            "Profesor":   "professor",
            "Día":        "day",
            "Hora":       "period",
            "Aula":       "room",
        }
        horario_df = horario_df.rename(columns=col_map)

        horario_df = horario_df.astype({
            "task_id":   str,
            "group":     str,
            "subject":   str,
            "professor": str,
            "day":       str,
            "period":    int,
            "room":      str,
        })

        horario_df["day"] = pd.Categorical(
            horario_df["day"],
            categories=self.days,
            ordered=True,
        )

        # Añadir 'floor' desde rooms_df (útil para evaluaciones que miran planta)
        horario_df = horario_df.merge(
            self.rooms_df[["floor"]],
            left_on="room",
            right_index=True,
            how="left",
        )

        # Reconstruir sol
        sol = {
            row["task_id"]: (
                row["professor"],
                (row["day"], row["period"]),
                row["room"],
            )
            for _, row in horario_df.iterrows()
        }

        return horario_df, sol
