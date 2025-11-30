import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go



def plot_horario_grupo(df_horario, grupo="1A"):
    # Filtrar solo el grupo que queremos
    df = df_horario[df_horario["Grupo"] == grupo].copy()

    # Orden de días (ajusta si usas otros nombres)
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    horas = sorted(df["Hora"].unique())  # p.ej. [1,2,3,4,5,6]

    dia_to_idx = {d: j for j, d in enumerate(dias)}
    hora_to_idx = {h: i for i, h in enumerate(horas)}

    # Cuadrícula de textos (asignatura + aula)
    grid = [["" for _ in dias] for _ in horas]
    for _, row in df.iterrows():
        i = hora_to_idx[row["Hora"]]
        j = dia_to_idx[row["Día"]]
        asignatura = str(row["Asignatura"])
        aula = str(row["Aula"])
        grid[i][j] = f"{asignatura}\n({aula})"

    fig, ax = plt.subplots(figsize=(len(dias)*2.5, len(horas)*1.2))

    # Dibujar las celdas como una tabla vacía
    for i in range(len(horas)):
        for j in range(len(dias)):
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                       fill=False, edgecolor="black"))
            texto = grid[i][j]
            if texto:
                ax.text(j + 0.5, i + 0.5, texto,
                        ha="center", va="center", fontsize=8, wrap=True)

    # Ejes y etiquetas
    ax.set_xticks(np.arange(len(dias)) + 0.5)
    ax.set_xticklabels(dias)
    ax.set_yticks(np.arange(len(horas)) + 0.5)
    ax.set_yticklabels(horas)

    ax.set_xlim(0, len(dias))
    ax.set_ylim(len(horas), 0)  # invertimos eje Y para que la hora 1 quede arriba

    ax.set_title(f"Horario del grupo {grupo}")
    plt.tight_layout()
    plt.show()

def abreviar_asignatura(nombre):
    """Crea una abreviatura cortita: primera palabra, máx 4 letras."""
    if not isinstance(nombre, str) or not nombre:
        return ""
    palabra = nombre.split()[0]  # primera palabra: LENGUA, MATEMÁTICAS, etc.
    return palabra[:4].upper()

def plot_horario_plotly(df_horario, grupo="1A", filename=None):
    # Filtrar grupo
    df = df_horario[df_horario["Grupo"] == grupo].copy()

    if df.empty:
        print(f"Advertencia: No hay datos para el grupo {grupo}")
        return

    # Orden de días
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    # Aseguramos que las horas estén ordenadas
    horas = sorted(df["Hora"].unique())

    dia_idx = {d: j for j, d in enumerate(dias)}
    hora_idx = {h: i for i, h in enumerate(horas)}

    # Matrices: None = celda vacía
    z     = [[None for _ in dias] for _ in horas]
    text  = [[""   for _ in dias] for _ in horas]
    hover = [[""   for _ in dias] for _ in horas]

    subj_to_id = {}
    next_id = 1

    for _, row in df.iterrows():
        # Validar que el día y hora existen en nuestros índices
        if row["Día"] not in dia_idx or row["Hora"] not in hora_idx:
            continue
            
        i = hora_idx[row["Hora"]]
        j = dia_idx[row["Día"]]
        asign = str(row["Asignatura"])
        aula  = str(row["Aula"])
        prof  = str(row["Profesor"])

        if asign not in subj_to_id:
            subj_to_id[asign] = next_id
            next_id += 1

        z[i][j] = subj_to_id[asign]
        text[i][j] = f"{asign}<br>({aula})"
        hover[i][j] = f"{asign}<br>{prof}<br>Aula: {aula}"

    # Escala de colores pastel
    pastel_scale = [
        [0.00, "#ffe082"], [0.16, "#ffccbc"], [0.33, "#c5e1a5"],
        [0.50, "#b3e5fc"], [0.66, "#f8bbd0"], [0.83, "#d1c4e9"],
        [1.00, "#fff9c4"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=dias, y=[f"Hora {h}" for h in horas],
        text=text, hovertext=hover, hoverinfo="text",
        texttemplate="%{text}", colorscale=pastel_scale,
        showscale=False, xgap=2, ygap=2, zauto=True,
    ))

    fig.update_layout(
        title=f"Horario del grupo {grupo}",
        xaxis_title="Día", yaxis_title="Hora",
        yaxis_autorange="reversed", width=1200, height=600,
        plot_bgcolor="white",
    )
    fig.update_traces(textfont=dict(size=11, color="black"))

    # --- CAMBIO PRINCIPAL: Guardado seguro y visualización opcional ---
    if filename is not None:
        try:
            fig.write_html(filename)
            print(f"Gráfico interactivo guardado correctamente en: {filename}")
        except Exception as e:
            print(f"Error guardando HTML: {e}")

    try:
        # Intentamos mostrarlo, pero si falta nbformat, no rompemos el programa
        fig.show()
    except ValueError as e:
        print(f"\nAviso: No se pudo mostrar el gráfico en el notebook (falta librería nbformat).")
        print(f"Sin embargo, el archivo '{filename}' sí debería haberse generado.")

import pandas as pd
import plotly.graph_objects as go

# --- 1. FUNCIÓN DE VISUALIZACIÓN ---
def plot_horario_interactivo(df_horario, filename="horario_final.html"):
    """
    Genera un Heatmap interactivo con Plotly.
    """
    dias_orden = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    
    # Asegurar que el DF no esté vacío
    if df_horario.empty:
        print("No hay datos para graficar.")
        return

    # Obtener rangos
    todas_horas = sorted(df_horario["Hora"].unique())
    grupos = sorted(df_horario["Grupo"].unique())
    
    # Mapeos de índices
    dia_idx = {d: i for i, d in enumerate(dias_orden)}
    hora_idx = {h: i for i, h in enumerate(todas_horas)}
    
    # Escala de colores pastel
    colors = [
        [0.0, "#ffffff"], # Fondo vacío
        [0.1, "#ffadad"], [0.2, "#ffd6a5"], [0.3, "#fdffb6"],
        [0.4, "#caffbf"], [0.5, "#9bf6ff"], [0.6, "#a0c4ff"],
        [0.7, "#bdb2ff"], [0.8, "#ffc6ff"], [1.0, "#fffffc"]
    ]

    traces = []
    # Diccionario para asignar un color único (ID numérico) a cada asignatura
    subj_map = {}
    next_color_id = 1

    for g in grupos:
        df_g = df_horario[df_horario["Grupo"] == g]
        
        # Grid vacío
        z_grid = [[None] * 5 for _ in range(len(todas_horas))]
        text_grid = [[""] * 5 for _ in range(len(todas_horas))]
        hover_grid = [[""] * 5 for _ in range(len(todas_horas))]
        
        for _, row in df_g.iterrows():
            d_str = row["Día"]
            h_val = row["Hora"]
            
            if d_str not in dia_idx or h_val not in hora_idx:
                continue
                
            c = dia_idx[d_str]
            r = hora_idx[h_val]
            
            asig = str(row["Asignatura"])
            
            # Asignar color
            if asig not in subj_map:
                subj_map[asig] = next_color_id
                next_color_id += 1
            
            # Rellenar celdas
            z_grid[r][c] = subj_map[asig]
            # Texto visible: Asignatura + Aula
            text_grid[r][c] = f"<b>{asig}</b><br>Aula: {row['Aula']}"
            # Texto hover: Detalles completos
            hover_grid[r][c] = (f"<b>{asig}</b><br>"
                                f"Prof: {row['Profesor']}<br>"
                                f"Grupo: {g}<br>"
                                f"Aula: {row['Aula']}")

        # Solo el primer grupo es visible por defecto
        is_visible = (g == grupos[0])
        
        traces.append(go.Heatmap(
            z=z_grid,
            x=dias_orden,
            y=[f"Hora {h}" for h in todas_horas],
            text=text_grid,
            texttemplate="%{text}",
            hovertext=hover_grid,
            hoverinfo="text",
            colorscale=colors,
            showscale=False,
            xgap=1, ygap=1,
            visible=is_visible,
            name=g
        ))

    # Crear botones del menú desplegable
    buttons = []
    for i, g in enumerate(grupos):
        # La máscara de visibilidad pone True solo en el índice i
        visibility = [False] * len(grupos)
        visibility[i] = True
        
        buttons.append(dict(
            label=f"Grupo {g}",
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"Horario - Grupo {g}"}
            ]
        ))

    layout = go.Layout(
        title=f"Horario - Grupo {grupos[0]}",
        xaxis_title="Día",
        yaxis_title="Franja Horaria",
        yaxis_autorange="reversed", # Hora 1 arriba
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.15, y=1,
            xanchor="left", yanchor="top"
        )],
        height=600,
        width=1000,
        template="plotly_white"
    )

    fig = go.Figure(data=traces, layout=layout)
    
    # Guardar HTML y mostrar en Notebook
    fig.write_html(filename)
    fig.show()

