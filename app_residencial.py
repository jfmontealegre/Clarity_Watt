# =====================================================================
# CLARITY WATT — APP COMPLETA UNIFICADA (OCR + DASHBOARD + PROYECCIÓN)
# =====================================================================

import io
import re
from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd
import numpy as np

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# OCR
from io import BytesIO
try:
    import pdfplumber
except:
    pdfplumber = None

try:
    from pdf2image import convert_from_bytes
except:
    convert_from_bytes = None

try:
    import pytesseract
except:
    pytesseract = None

from PIL import Image

# PDF REPORT
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except:
    canvas = None
    letter = None

# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from scipy import stats


# =====================================================================
# CONFIGURACIÓN DE MARCA Y UI
# =====================================================================
st.set_page_config(page_title="Clarity Watt", page_icon="⚡", layout="wide")

# Cargar estilo
try:
    with open("assets/css/estilos.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass


# =====================================================================
# UTILIDADES DE PARSEO Y CONVERSIÓN NUMÉRICA
# =====================================================================
def _safe_search(pattern: str, text: str, group: int = 1, flags=0, default=None):
    m = re.search(pattern, text, flags=flags | re.MULTILINE)
    return m.group(group).strip() if m else default


def _to_int_safe(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    v = str(value).strip()

    if "," in v and "." in v:
        last_sep = max(v.rfind(","), v.rfind("."))
        entero = v[:last_sep]
        entero = entero.replace(".", "").replace(",", "")
    else:
        if "," in v:
            entero = v.split(",")[0]
        else:
            entero = v.split(".")[0]
        entero = entero.replace(".", "").replace(",", "")
    try:
        return int(entero)
    except:
        try:
            return int(float(v.replace(",", ".")))
        except:
            return None


def _to_float_safe(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    v = str(value).strip()
    if "," in v and "." in v:
        v = v.replace(".", "").replace(",", ".")
    elif "," in v:
        v = v.replace(".", "").replace(",", ".")
    try:
        return float(v)
    except:
        return None


# =====================================================================
# OCR EMCALI
# =====================================================================
def _extract_text_pdf(file_bytes: bytes) -> str:
    if pdfplumber is None:
        return ""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _extract_text_ocr_from_pdf(file_bytes: bytes) -> str:
    if convert_from_bytes is None or pytesseract is None:
        return ""
    images = convert_from_bytes(file_bytes)
    texts = [pytesseract.image_to_string(img, lang="spa") for img in images]
    return "\n".join(texts)


def _extract_text_ocr_from_image(file_bytes: bytes) -> str:
    if pytesseract is None:
        return ""
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img, lang="spa")


def parse_emcali_invoice(file_bytes: bytes, file_type: str) -> Dict[str, Any]:
    """
    OCR + extracción de campos clave para facturas EMCALI.
    """
    if file_type == "pdf":
        text = _extract_text_pdf(file_bytes)
        if not text or len(text) < 200:
            text = _extract_text_ocr_from_pdf(file_bytes)
    else:
        text = _extract_text_ocr_from_image(file_bytes)

    if not text:
        return {}

    text = re.sub(r"[ \t]+", " ", text)

    consumo_kwh = _safe_search(r"Consumo Actual\s*[:\-]?\s*(\d+)\s*KWH", text)
    consumo_kwh = _to_int_safe(consumo_kwh)

    total_pagar_str = _safe_search(r"TOTAL A PAGAR\s*\$?\s*([\d\.,]+)", text)
    total_pagar = _to_int_safe(total_pagar_str)

    historico = []
    bloque = _safe_search(r"(Consumos Anteriores[\s\S]{0,400}PROM\s*-\s*[\d\.,]+)", text, flags=re.IGNORECASE)
    if bloque:
        pares = re.findall(r"([A-Za-z]{3,})\s*[-–]\s*([\d\.,]+)", bloque)
        for mes, val in pares:
            kv = _to_float_safe(val)
            if kv is not None:
                historico.append({"mes": mes, "kwh": float(kv)})

        proms = re.findall(r"PROM\s*-\s*([\d\.,]+)", bloque)
        consumo_prom_kwh = _to_float_safe(proms[-1]) if proms else None
    else:
        consumo_prom_kwh = None

    bloque_energia = re.search(r"(ENERG[ií]A[\s\S]{0,200}?TOTAL\s*\$?\s*([\d\.,]+))", text, re.IGNORECASE)
    total_energia = _to_int_safe(bloque_energia.group(2)) if bloque_energia else None

    nombre = _safe_search(r"^([A-ZÁÉÍÓÚÑ ]{3,})\s*\nC\.C\./Nit", text)
    contrato = _safe_search(r"CONTRATO\s*[:\-]?\s*(\d+)", text)

    return {
        "texto_crudo": text,
        "nombre_suscriptor": nombre,
        "numero_contrato": contrato,
        "consumo_kwh": consumo_kwh,
        "consumo_prom_kwh": consumo_prom_kwh,
        "total_pagar": total_pagar,
        "total_pagar_str": total_pagar_str,
        "total_energia": total_energia,
        "historico_kwh": historico,
    }


# =====================================================================
# RECOMENDACIONES
# =====================================================================
RECOMENDACIONES_BASE = [
    {"titulo": "Cambia bombillos a LED", "descripcion": "Reemplaza bombillos comunes por LED.", "costo": 120000, "fraccion": 0.10, "reduccion": 0.60},
    {"titulo": "Nevera eficiente", "descripcion": "Cambia tu nevera antigua por una A++.", "costo": 2500000, "fraccion": 0.25, "reduccion": 0.30},
    {"titulo": "Ajusta el aire a 24°C", "descripcion": "Configura tu aire acondicionado en 24°C.", "costo": 0, "fraccion": 0.35, "reduccion": 0.10},
    {"titulo": "Optimiza la lavadora", "descripcion": "Lava con cargas completas y agua fría.", "costo": 0, "fraccion": 0.10, "reduccion": 0.15},
]


def estimar_tarifa_kwh_desde_factura(consumo_kwh, total_energia_cop=None, total_pagar_cop=None):
    if consumo_kwh <= 0:
        return 0.0
    if total_energia_cop:
        return total_energia_cop / consumo_kwh
    if total_pagar_cop:
        return (total_pagar_cop * 0.7) / consumo_kwh
    return 0.0


def calcular_recomendaciones(consumo_kwh, tarifa, datos_hogar):
    recs = []
    for r in RECOMENDACIONES_BASE:
        ahorro_kwh = consumo_kwh * r["fraccion"] * r["reduccion"]
        ahorro_cop = ahorro_kwh * tarifa
        costo = r["costo"]

        if costo > 0 and ahorro_cop > 0:
            pay = costo / ahorro_cop
        elif costo == 0 and ahorro_cop > 0:
            pay = 0
        else:
            pay = None

        recs.append({
            "titulo": r["titulo"],
            "descripcion": r["descripcion"],
            "costo_cop": costo,
            "ahorro_kwh_mes": round(ahorro_kwh, 1),
            "ahorro_cop_mes": int(ahorro_cop),
            "payback_meses": round(pay, 1) if pay is not None else None
        })

    return sorted(recs, key=lambda x: x["payback_meses"] if x["payback_meses"] is not None else 9999)


# =====================================================================
# PDF REPORT
# =====================================================================
def build_markdown_report(context):
    md = []
    md.append("# Informe Clarity Watt")
    md.append(f"**Nombre:** {context.get('nombre','-')}")
    md.append(f"**Contrato:** {context.get('contrato','-')}")
    md.append("")
    md.append("## Consumo")
    md.append(f"- Mes: **{context.get('consumo_kwh',0)} kWh**")
    md.append(f"- Promedio: **{context.get('consumo_prom_kwh',0)} kWh**")
    md.append("")
    md.append("## Recomendaciones")
    for r in context["recomendaciones"]:
        md.append(f"### {r['titulo']}")
        md.append(f"- Ahorro: {r['ahorro_kwh_mes']} kWh/mes (~${r['ahorro_cop_mes']:,}/mes)")
        md.append(f"- Costo: ${r['costo_cop']:,}")
        md.append(f"- Payback: {r['payback_meses']} meses")
        md.append("")
    return "\n".join(md)


def crear_pdf_desde_markdown(md_text: str) -> bytes:
    if canvas is None:
        raise RuntimeError("Instala reportlab")
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    x, y = 50, h - 50
    for line in md_text.split("\n"):
        c.drawString(x, y, line[:120])
        y -= 14
        if y < 50:
            c.showPage()
            y = h - 50
    c.save()
    return buf.getvalue()


# =====================================================================
# FUNCIONES DE ANOMALÍAS Y PROYECCIÓN
# =====================================================================
def detect_anomalies_zscore(series: pd.Series, threshold=2.5):
    if len(series) < 2:
        return pd.Series([False] * len(series), index=series.index)
    z = np.abs(stats.zscore(series, nan_policy="omit"))
    return pd.Series(z > threshold, index=series.index)


def detect_anomalies_isolationforest(series: pd.Series, contamination=0.1):
    if len(series.dropna()) < 5:
        return detect_anomalies_zscore(series)
    try:
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(series.fillna(series.mean()).values.reshape(-1, 1))
        return pd.Series(preds == -1, index=series.index)
    except:
        return detect_anomalies_zscore(series)


def project_future_consumption(df_monthly, months_ahead=3):
    if df_monthly.empty or len(df_monthly) < 2:
        return pd.DataFrame(columns=["Mes", "Pred_kWh"])
    df = df_monthly.copy().reset_index()

    df["t"] = np.arange(len(df))
    X = df[["t"]].values
    y = df["Consumo_kWh"].values

    model = LinearRegression()
    model.fit(X, y)

    future_t = np.arange(len(df), len(df) + months_ahead).reshape(-1, 1)
    preds = model.predict(future_t)

    last_month = df_monthly["Mes"].max()
    new_dates = [last_month + pd.DateOffset(months=i + 1) for i in range(months_ahead)]

    return pd.DataFrame({"Mes": new_dates, "Pred_kWh": np.maximum(preds, 0)})


# =====================================================================
# INTERFAZ — TABS PRINCIPALES
# =====================================================================

tab_diag, tab_dash = st.tabs(["🧾 Diagnóstico", "📊 Dashboard energético"])

# =====================================================================
# TAB 1 — DIAGNÓSTICO
# =====================================================================
with tab_diag:

    col_a, col_b, _ = st.columns([1, 4, 1])
    with col_b:
        st.image("assets/img/logo.png", width=320)
        #st.markdown("<h2 style='text-align:center;'>Tu energía clara y bajo control</h2>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("1. Sube tu factura")

    archivo = st.file_uploader("Carga tu factura (PDF/JPG/PNG)")

    if "ocr_data" not in st.session_state:
        st.session_state["ocr_data"] = {}

    if archivo:
        file_bytes = archivo.read()
        ext = archivo.name.split(".")[-1].lower()
        tipo = "pdf" if ext == "pdf" else "image"

        with st.spinner("Procesando factura..."):
            try:
                data = parse_emcali_invoice(file_bytes, tipo)
                st.session_state["ocr_data"] = data
                st.success("Factura leída correctamente.")
            except:
                st.error("Error procesando factura.")
                st.session_state["ocr_data"] = {}

        with st.expander("Ver texto detectado"):
            st.text_area("OCR Detectado", st.session_state["ocr_data"].get("texto_crudo",""))

    ocr = st.session_state["ocr_data"]

    st.subheader("2. Datos detectados (editar si es necesario)")
    c1, c2, c3 = st.columns(3)
    with c1:
        nombre = st.text_input("Nombre", ocr.get("nombre_suscriptor",""))
        contrato = st.text_input("Contrato", ocr.get("numero_contrato",""))
    with c2:
        consumo_kwh = st.number_input("Consumo mes (kWh)", value=float(ocr.get("consumo_kwh") or 0))
        consumo_prom_kwh = st.number_input("Promedio (kWh)", value=float(ocr.get("consumo_prom_kwh") or 0))
    with c3:
        total_energia = st.number_input("Total energía (COP)", value=float(ocr.get("total_energia") or 0))
        total_fact = st.text_input("Total factura", ocr.get("total_pagar_str") or "")

    st.subheader("3. Resumen")
    if consumo_prom_kwh > 0:
        dif = consumo_kwh - consumo_prom_kwh
        pct = 100 * dif / consumo_prom_kwh
    else:
        dif = pct = 0

    tarifa_kwh = estimar_tarifa_kwh_desde_factura(
    consumo_kwh,
    total_energia_cop=int(total_energia) if total_energia > 0 else None,
    
    )

    st.session_state["tarifa_kwh"] = tarifa_kwh

    st.write(f"Tarifa estimada: **${tarifa_kwh:,.0f} COP/kWh**")



    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Consumo actual", f"{consumo_kwh:.0f} kWh")
    kpi2.metric("Promedio", f"{consumo_prom_kwh:.0f} kWh")
    kpi3.metric("Diferencia", f"{dif:+.0f} kWh", f"{pct:+.1f}%")

    st.markdown("---")

    # ===============================================================
    # 6. CONFIGURACIÓN DE ELECTRODOMÉSTICOS Y USO DIARIO
    # ===============================================================

    st.subheader("6. Equipos en tu hogar y uso diario")
    st.write("Selecciona los electrodomésticos que tienes y cuántas horas al día los usas. "
            "Clarity Watt calculará tu consumo real estimado.")

    # Base de electrodomésticos (potencia real promedio en Watts)
    ELECTRODOMESTICOS = {
        "Nevera (nueva)": 120,
        "Nevera (antigua)": 250,
        "Aire acondicionado (12000 BTU)": 1200,
        "Lavadora": 500,
        "Secadora": 2500,
        "Televisor LED": 100,
        "Computador": 150,
        "Cargadores / Celulares": 20,
        "Ventilador": 60,
        "Microondas": 1200,
        "Horno eléctrico": 1800,
        "Iluminación incandescente (60W)": 60,
        "Iluminación fluorescente (20W)": 20,
        "Iluminación LED (9W)": 9,
    }

    # Selector múltiple
    equipos_usuario = st.multiselect(
        "Selecciona tus electrodomésticos",
        list(ELECTRODOMESTICOS.keys()),
    )

    equipos_uso = {}

    # Para cada equipo, solicitar horas de uso
    for eq in equipos_usuario:
        colA, colB = st.columns([2, 1])
        with colA:
            h = colA.slider(f"Horas diarias — {eq}", 0.0, 24.0, 2.0, step=0.5)
        equipos_uso[eq] = h

    # CÁLCULO DEL CONSUMO REAL ESTIMADO
    consumo_equipos = []
    for eq, horas in equipos_uso.items():
        potencia = ELECTRODOMESTICOS[eq]  # Watts
        consumo_kwh_mes = (potencia / 1000) * horas * 30
        costo_mes = consumo_kwh_mes * tarifa_kwh

        consumo_equipos.append({
            "Equipo": eq,
            "Potencia_W": potencia,
            "Horas/día": horas,
            "kWh_mes": consumo_kwh_mes,
            "Costo_mes": costo_mes,
        })

    df_equipos = pd.DataFrame(consumo_equipos)

    # Mostrar tabla
    if not df_equipos.empty:
        st.markdown("### 📦 Consumo mensual por electrodoméstico")
        st.dataframe(
            df_equipos.assign(
                kWh_mes=lambda d: d["kWh_mes"].round(2),
                Costo_mes=lambda d: d["Costo_mes"].round(0)
            )
        )

        # Gráfico de barras — consumo por equipo
        st.markdown("### 🔥 ¿Qué equipo consume más?")
        fig_equipos = px.bar(
            df_equipos,
            x="Equipo",
            y="kWh_mes",
            title="Consumo mensual por electrodoméstico (kWh)",
            color="kWh_mes",
            color_continuous_scale="Bluered",
        )
        st.plotly_chart(fig_equipos, use_container_width=True)

        # Insight automático
        top = df_equipos.sort_values("kWh_mes", ascending=False).iloc[0]
        st.info(
            f"⚡ Tu equipo que más consume es **{top['Equipo']}**, con **{top['kWh_mes']:.1f} kWh/mes**, "
            f"equivalentes a **${top['Costo_mes']:.0f} COP/mes**."
        )

        # Integrar al contexto general
        consumo_total_equipos = df_equipos["kWh_mes"].sum()
        st.metric("Consumo mensual total estimado por equipos", f"{consumo_total_equipos:.1f} kWh")

        # Comparar OCR vs Equipos
        if consumo_kwh > 0:
            diferencia = consumo_total_equipos - consumo_kwh
            st.metric("Diferencia vs factura OCR", f"{diferencia:+.1f} kWh")

    st.subheader("4. Recomendaciones de ahorro")

    total_pagar_cop = _to_int_safe(total_fact) if total_fact else ocr.get("total_pagar")
    tarifa = estimar_tarifa_kwh_desde_factura(consumo_kwh, total_energia, total_pagar_cop)
    st.caption(f"Tarifa estimada: **${tarifa:,.0f} COP/kWh**")

    personas = st.slider("Personas", 1, 10, 3)
    datos_hogar = {"personas": personas}

    recomendaciones = []
    if tarifa > 0 and consumo_kwh > 0:
        recomendaciones = calcular_recomendaciones(consumo_kwh, tarifa, datos_hogar)

    if recomendaciones:
        st.markdown("### ⭐ Top 3 acciones")
        top3 = recomendaciones[:3]
        cols = st.columns(len(top3))

        for c, r in zip(cols, top3):
            color = "#22c55e" if r["payback_meses"] <= 12 else "#eab308" if r["payback_meses"] <= 36 else "#ef4444"
            box = f"""
            <div class='card' style='min-height:160px'>
                <h4>{r['titulo']}</h4>
                <p style='font-size:22px;'><b>${r['ahorro_cop_mes']:,}</b>/mes</p>
                <p>Ahorro: {r['ahorro_kwh_mes']} kWh/mes</p>
                <span style='background:{color}; color:white; padding:4px 10px; border-radius:20px; font-size:12px'>
                    Payback: {r['payback_meses']} meses
                </span>
                <p style='font-size:12px; margin-top:6px'>Inversión: ${r['costo_cop']:,}</p>
            </div>
            """
            c.markdown(box, unsafe_allow_html=True)

        st.markdown("### 📋 Detalle completo")
        for r in recomendaciones:
            html = f"""
            <div class='card' style='margin-bottom:12px'>
                <h4>{r['titulo']}</h4>
                <p>{r['descripcion']}</p>
                <p>Ahorro: ${r['ahorro_cop_mes']:,}/mes</p>
                <p>Inversión: ${r['costo_cop']:,}</p>
                <p>Payback: {r['payback_meses']} meses</p>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("5. Descargar reporte")

    email = st.text_input("Correo destino (opcional)")
    opciones = st.multiselect("Opciones", ["Descargar PDF"], default=["Descargar PDF"])

    if st.button("Generar reporte"):
        if not recomendaciones:
            st.error("No hay recomendaciones para generar el informe.")
        else:
            ctx = {
                "nombre": nombre,
                "contrato": contrato,
                "consumo_kwh": consumo_kwh,
                "consumo_prom_kwh": consumo_prom_kwh,
                "recomendaciones": recomendaciones
            }
            md = build_markdown_report(ctx)
            pdf = crear_pdf_desde_markdown(md)

            if "Descargar PDF" in opciones:
                st.download_button(
                    "⬇️ Descargar informe PDF",
                    pdf,
                    file_name="claritywatt_reporte.pdf",
                    mime="application/pdf"
                )


# =====================================================================
# TAB 2 — DASHBOARD ENERGÉTICO COMPLETO
# =====================================================================
with tab_dash:
    st.header("Dashboard energético (OCR + Insights + Proyección)")

    ocr_data = st.session_state.get("ocr_data", {})
    historico = ocr_data.get("historico_kwh", [])

    # --- SERIE HISTÓRICA ---
    if historico and len(historico) >= 3:
        df_hist = pd.DataFrame(historico)
        try:
            df_hist["Mes"] = pd.to_datetime(df_hist["mes"], format="%b")
            df_hist["Mes"] = df_hist["Mes"].apply(lambda d: d.replace(year=pd.Timestamp.today().year))
        except:
            df_hist["Mes"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_hist), freq="M")
        df_hist.rename(columns={"kwh": "Consumo_kWh"}, inplace=True)
    else:
        base = ocr_data.get("consumo_prom_kwh") or ocr_data.get("consumo_kwh") or 200
        meses = pd.date_range(end=pd.Timestamp.today(), periods=12, freq="M")
        valores = np.round(np.linspace(base * 0.9, base * 1.1, 12) + np.random.randint(-10, 10, 12))
        df_hist = pd.DataFrame({"Mes": meses, "Consumo_kWh": valores})

    df_hist = df_hist.sort_values("Mes")
    df_hist["mes_str"] = df_hist["Mes"].dt.strftime("%Y-%m")

    # --- FILTROS ---
    st.markdown("### Filtros")
    colf1, colf2 = st.columns(2)
    with colf1:
        personas_range = st.slider("Filtro personas (estimado)", 1, 6, (1, 6))
    with colf2:
        meses_list = list(df_hist["mes_str"])
        start_m, end_m = st.select_slider("Rango de meses", meses_list, (meses_list[0], meses_list[-1]))

    df_filt = df_hist[(df_hist["mes_str"] >= start_m) & (df_hist["mes_str"] <= end_m)]

    # --- KPIs ---
    st.markdown("### KPIs del rango filtrado")

    total_kwh = df_filt["Consumo_kWh"].sum()
    promedio_kwh = df_filt["Consumo_kWh"].mean()
    tarifa = estimar_tarifa_kwh_desde_factura(
        total_kwh,
        total_energia_cop=ocr_data.get("total_energia"),
        total_pagar_cop=ocr_data.get("total_pagar"),
    )
    costo = total_kwh * tarifa

    k1, k2, k3 = st.columns(3)
    k1.metric("Consumo total", f"{total_kwh:.0f} kWh")
    k2.metric("Costo estimado", f"$ {costo:,.0f}")
    k3.metric("Promedio mensual", f"{promedio_kwh:.1f} kWh")

    st.markdown("---")

    # --- ANOMALÍAS ---
    st.markdown("### 🔍 Detección de anomalías")

    metodo = st.selectbox("Método", ["Z-score", "IsolationForest"])

    if metodo == "Z-score":
        df_filt["anomaly"] = detect_anomalies_zscore(df_filt["Consumo_kWh"])
    else:
        df_filt["anomaly"] = detect_anomalies_isolationforest(df_filt["Consumo_kWh"])

    fig = px.line(df_filt, x="Mes", y="Consumo_kWh", markers=True)
    fig.add_scatter(
        x=df_filt.loc[df_filt["anomaly"], "Mes"],
        y=df_filt.loc[df_filt["anomaly"], "Consumo_kWh"],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Anomalía"
    )
    fig.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- INSIGHTS ---
    st.markdown("### 💡 Insights automáticos")

    insights = []
    if len(df_filt) > 1:
        last = df_filt["Consumo_kWh"].iloc[-1]
        prom = df_filt["Consumo_kWh"].mean()

        if last > prom * 1.2:
            insights.append(f"El último mes está {((last/prom)-1)*100:.0f}% por encima del promedio.")

        if df_filt["anomaly"].any():
            insights.append("Se detectaron meses anómalos que pueden indicar uso inusual de energía.")

    if insights:
        for ins in insights:
            st.info(ins)
    else:
        st.write("Sin insights especiales por ahora.")

    st.markdown("---")

    # --- PROYECCIÓN ---
    st.markdown("### 📈 Proyección de consumo (3 meses)")

    df_proj_in = df_hist[["Mes", "Consumo_kWh"]]
    df_proj = project_future_consumption(df_proj_in, 3)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_proj_in["Mes"], y=df_proj_in["Consumo_kWh"], name="Histórico"))
    fig2.add_trace(go.Scatter(x=df_proj["Mes"], y=df_proj["Pred_kWh"], name="Proyección", line=dict(color="#22c55e", dash="dash")))
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df_proj.assign(Pred_KWh=lambda d: d["Pred_kWh"].round(1)))

    st.markdown("---")

    st.subheader("Datos usados")
    st.dataframe(df_filt.drop(columns=["mes_str"]), use_container_width=True)

    # ============================================================
# 💸 SIMULADOR DE AHORRO — VERSIÓN PRO CLARITY WATT
# ============================================================
# Recuperar la tarifa detectada desde el diagnóstico
tarifa_kwh = st.session_state.get("tarifa_kwh", 0)
tarifa = tarifa_kwh   # tarifa unificada para cálculos

st.markdown("---")
st.header("💸 Simulador de ahorro PRO")

st.write(
    "Explora diferentes escenarios de ahorro ajustando tus hábitos y cambiando equipos "
    "por versiones eficientes. Comparamos tu situación actual vs escenarios mejorados."
)

if tarifa == 0:
    st.warning("Para simular ahorros necesitamos una tarifa estimada en COP/kWh (se calcula en el diagnóstico).")
else:
    # Necesitamos que arriba ya se haya configurado al menos un equipo
    if "df_equip_dash" not in locals():
        # Por si copiaste esto antes del módulo de electrodomésticos
        df_equip_dash = pd.DataFrame()

    if df_equip_dash.empty:
        st.info("Primero configura tus electrodomésticos en la sección anterior para usar el simulador PRO.")
    else:
        # -----------------------------------------------------------
        # ESCENARIO 0: SITUACIÓN ACTUAL
        # -----------------------------------------------------------
        consumo_esc0_kwh = df_equip_dash["kWh_mes"].sum()
        costo_esc0 = consumo_esc0_kwh * tarifa

        st.subheader("🎯 Escenario 0 — Tal como estás hoy")
        col0a, col0b = st.columns(2)
        with col0a:
            st.metric("Consumo mensual actual estimado", f"{consumo_esc0_kwh:.1f} kWh")
        with col0b:
            st.metric("Costo mensual actual estimado", f"${costo_esc0:,.0f}")

        # -----------------------------------------------------------
        # CONTROLES PRO — ESCENARIO 1 y 2
        # -----------------------------------------------------------
        st.markdown("---")
        st.subheader("⚙️ Ajustes para escenarios de ahorro")

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

        with col_ctrl1:
            reduccion_confort_pct = st.slider(
                "Reducir horas de equipos de confort (AA, secadora, horno, microondas)",
                0, 50, 15, step=5,
                help="Reducción porcentual de horas diarias en equipos de alto consumo."
            )

        with col_ctrl2:
            cambiar_inc_led = st.checkbox(
                "Cambiar toda iluminación incandescente a LED",
                value=True
            )
            cambiar_nev_ant = st.checkbox(
                "Cambiar nevera antigua por eficiente",
                value=False
            )

        with col_ctrl3:
            cambio_tarifa_pct = st.slider(
                "Cambio esperado en la tarifa de energía (%)",
                -20, 50, 0, step=5,
                help="Simula un aumento o disminución de la tarifa eléctrica."
            )

        factor_tarifa = 1 + cambio_tarifa_pct / 100.0
        tarifa_futura = tarifa * factor_tarifa

        # -----------------------------------------------------------
        # ESCENARIO 1: MENOS HORAS EN EQUIPOS DE CONFORT
        # -----------------------------------------------------------
        equipos_confort = [
            "Aire acondicionado (12000 BTU)",
            "Secadora",
            "Horno eléctrico",
            "Microondas",
        ]

        filas_esc1 = []
        for _, row in df_equip_dash.iterrows():
            eq = row["Equipo"]
            horas = row["Horas_día"]
            potencia = row["Potencia_W"]

            if eq in equipos_confort and reduccion_confort_pct > 0:
                horas_nuevas = horas * (1 - reduccion_confort_pct / 100.0)
            else:
                horas_nuevas = horas

            kwh_esc1 = (potencia / 1000) * horas_nuevas * 30

            filas_esc1.append({
                "Equipo": eq,
                "Potencia_W": potencia,
                "Horas_día": horas_nuevas,
                "kWh_mes": kwh_esc1,
            })

        df_esc1 = pd.DataFrame(filas_esc1)
        consumo_esc1_kwh = df_esc1["kWh_mes"].sum()
        costo_esc1 = consumo_esc1_kwh * tarifa

        # -----------------------------------------------------------
        # ESCENARIO 2: EQUIPOS EFICIENTES + HORAS REDUCIDAS
        # -----------------------------------------------------------
        filas_esc2 = []
        for _, row in df_esc1.iterrows():
            eq = row["Equipo"]
            horas = row["Horas_día"]
            potencia = row["Potencia_W"]

            # Cambiar incandescente a LED
            if cambiar_inc_led and eq == "Iluminación incandescente (60W)":
                potencia = 9  # LED

            # Cambiar nevera antigua a nueva
            if cambiar_nev_ant and eq == "Nevera (antigua)":
                potencia = 120  # nevera nueva eficiente

            kwh_esc2 = (potencia / 1000) * horas * 30

            filas_esc2.append({
                "Equipo": eq,
                "Potencia_W": potencia,
                "Horas_día": horas,
                "kWh_mes": kwh_esc2,
            })

        df_esc2 = pd.DataFrame(filas_esc2)
        consumo_esc2_kwh = df_esc2["kWh_mes"].sum()
        costo_esc2 = consumo_esc2_kwh * tarifa
        costo_esc2_futuro = consumo_esc2_kwh * tarifa_futura

        # -----------------------------------------------------------
        # RESUMEN DE ESCENARIOS
        # -----------------------------------------------------------
        st.markdown("### 📊 Comparación de escenarios")

        df_escenarios = pd.DataFrame([
            {
                "Escenario": "0 – Actual",
                "Consumo_kWh": consumo_esc0_kwh,
                "Costo_mensual_COP": costo_esc0,
                "Tarifa_COP_kWh": tarifa,
            },
            {
                "Escenario": "1 – Menos horas en confort",
                "Consumo_kWh": consumo_esc1_kwh,
                "Costo_mensual_COP": costo_esc1,
                "Tarifa_COP_kWh": tarifa,
            },
            {
                "Escenario": "2 – Equipos eficientes + menos horas",
                "Consumo_kWh": consumo_esc2_kwh,
                "Costo_mensual_COP": costo_esc2,
                "Tarifa_COP_kWh": tarifa,
            },
        ])

        st.dataframe(
            df_escenarios.assign(
                Consumo_kWh=lambda d: d["Consumo_kWh"].round(1),
                Costo_mensual_COP=lambda d: d["Costo_mensual_COP"].round(0),
            ),
            use_container_width=True,
        )

        # Gráfico de barras de costo por escenario
        fig_esc = px.bar(
            df_escenarios,
            x="Escenario",
            y="Costo_mensual_COP",
            title="Costo mensual por escenario (COP)",
            text_auto=".0f",
            color="Escenario",
            color_discrete_sequence=["#ef4444", "#eab308", "#22c55e"],
        )
        fig_esc.update_layout(
            yaxis_title="COP/mes",
            xaxis_title="Escenario",
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_esc, use_container_width=True)

        # -----------------------------------------------------------
        # AHORRO MENSUAL Y ANUAL
        # -----------------------------------------------------------
        ahorro_mensual_esc2 = costo_esc0 - costo_esc2
        ahorro_anual_esc2 = ahorro_mensual_esc2 * 12

        st.markdown("### 🏆 Ahorro total potencial")

        c_sav1, c_sav2, c_sav3 = st.columns(3)
        c_sav1.metric("Ahorro mensual estimado", f"${ahorro_mensual_esc2:,.0f}")
        c_sav2.metric("Ahorro anual estimado", f"${ahorro_anual_esc2:,.0f}")
        c_sav3.metric("Reducción de consumo", f"{(consumo_esc0_kwh - consumo_esc2_kwh):.1f} kWh/mes")

        st.success(
            f"Aplicando el escenario 2, podrías reducir alrededor de **${ahorro_mensual_esc2:,.0f} COP/mes**, "
            f"es decir, cerca de **${ahorro_anual_esc2:,.0f} COP/año**."
        )

        # -----------------------------------------------------------
        # IMPACTO DE CAMBIO DE TARIFA
        # -----------------------------------------------------------
        st.markdown("### ⚡ Si la tarifa cambia en el futuro...")

        c_tar1, c_tar2 = st.columns(2)
        with c_tar1:
            st.metric(
                "Tarifa actual",
                f"${tarifa:,.0f} COP/kWh"
            )
        with c_tar2:
            st.metric(
                "Tarifa futura simulada",
                f"${tarifa_futura:,.0f} COP/kWh",
                f"{cambio_tarifa_pct:+.0f}%"
            )

        costo_esc0_futuro = consumo_esc0_kwh * tarifa_futura

        st.info(
            f"Si la tarifa cambiara a **${tarifa_futura:,.0f} COP/kWh**, "
            f"tu escenario actual costaría **${costo_esc0_futuro:,.0f} COP/mes**, "
            f"mientras que el escenario 2 costaría **${costo_esc2_futuro:,.0f} COP/mes**."
        )

        # -----------------------------------------------------------
        # GRÁFICO DE PROYECCIÓN 12 MESES (ESC0 vs ESC2)
        # -----------------------------------------------------------
        st.markdown("### 📈 Proyección 12 meses — Actual vs Escenario 2")

        meses = list(range(1, 13))
        costos_esc0_12 = [costo_esc0_futuro] * 12
        costos_esc2_12 = [costo_esc2_futuro] * 12

        fig_proj_12 = go.Figure()
        fig_proj_12.add_trace(
            go.Scatter(
                x=meses,
                y=costos_esc0_12,
                mode="lines",
                name="Escenario 0 (actual)",
                line=dict(color="#ef4444"),
            )
        )
        fig_proj_12.add_trace(
            go.Scatter(
                x=meses,
                y=costos_esc2_12,
                mode="lines",
                name="Escenario 2 (ahorro)",
                line=dict(color="#22c55e", dash="dash"),
            )
        )

        fig_proj_12.update_layout(
            xaxis_title="Mes",
            yaxis_title="Costo mensual estimado (COP)",
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            title="Proyección de costo en 12 meses — Actual vs Escenario de ahorro"
        )

        st.plotly_chart(fig_proj_12, use_container_width=True)

