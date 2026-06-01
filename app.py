import io
import numpy as np
import pandas as pd
import streamlit as st
import CoolProp.CoolProp as cp
from scipy.optimize import brentq

st.set_page_config(page_title="Einfache CO2 Kreislaufberechnung", layout="wide", page_icon="logo.png")

APP_TITLE = "Einfache CO2 Kreislaufberechnung"
APP_VERSION = "0.9.12V"
FLUID = "CO2"


def getVi(di):
    return np.pi * (di) ** 2 * 2.5 * 1e-7


def getAM(da):
    return np.pi * da * 1e-3


cu_pipes = [
    ("6x1", 6, 1, 4, getVi(4), getAM(6)),
    ("8x1", 8, 1, 6, getVi(6), getAM(8)),
    ("10x1", 10, 1, 8, getVi(8), getAM(10)),
    ("12x1", 12, 1, 10, getVi(10), getAM(12)),
    ("15x1", 15, 1, 13, getVi(13), getAM(15)),
    ("16x1", 16, 1, 14, getVi(14), getAM(16)),
    ("18x1", 18, 1, 16, getVi(16), getAM(18)),
    ("22x1", 22, 1, 20, getVi(20), getAM(22)),
    ("28x1", 28, 1, 26, getVi(26), getAM(28)),
    ("28x1.5", 28, 1.5, 25, getVi(25), getAM(28)),
    ("35x1.5", 35, 1.5, 32, getVi(32), getAM(35)),
    ("42x1.5", 42, 1.5, 39, getVi(39), getAM(42)),
    ("54x1.5", 54, 1.5, 51, getVi(51), getAM(54)),
    ("54x2", 54, 2, 50, getVi(50), getAM(54)),
    ("64x2", 64, 2, 60, getVi(60), getAM(64)),
    ("76.1x2", 76.1, 2, 74.1, getVi(72.1), getAM(76.1)),
    ("88.9x2", 88.9, 2, 84.9, getVi(84.9), getAM(88.9)),
    ("108x2.5", 108, 2.5, 103, getVi(103), getAM(108)),
    ("133x3", 133, 3, 127, getVi(127), getAM(133)),
]


def friction_coefficient(di, w, nu):
    Re = di * w / nu
    k = 0.0015 / 1e3
    epsilon_k = k / di
    lambda_hagenpoiseulle = 64 / Re
    lambda_blasius = 0.3164 / Re**0.25
    lambda_nikuradse = (-2 * np.log10(k / 3.71 / di)) ** -2
    lambda_prandtl = 0.02
    for _ in range(10):
        lambda_prandtl = (2 * np.log10(Re * np.sqrt(lambda_prandtl))) ** -2
    lambda_colebrookwhite = 0.02
    for _ in range(10):
        lambda_colebrookwhite = (-2 * np.log10(2.51 / Re / np.sqrt(lambda_colebrookwhite) + k / 3.71 / di)) ** -2
    check_moody_diagram = Re * np.sqrt(lambda_nikuradse) * k / di
    if Re < 2320:
        return lambda_hagenpoiseulle
    if check_moody_diagram >= 200:
        return lambda_nikuradse
    if epsilon_k < 0.001 and Re < 10000:
        return lambda_blasius
    if epsilon_k < 0.0002 and Re < 100000:
        return lambda_blasius
    if epsilon_k < 0.00002 and Re < 1000000:
        return lambda_prandtl
    if epsilon_k < 0.00001:
        return lambda_prandtl
    return lambda_colebrookwhite


def find_next_bigger_pipe(di):
    di_values = [pipe[3] for pipe in cu_pipes]
    bigger = [x for x in di_values if x > di * 1e3]
    return min(bigger) * 1e-3 if bigger else None


def pipe_geometry(di_m, length_m):
    key = round(di_m * 1e3, 1)
    row = next(x for x in cu_pipes if abs(x[3] - key) < 1e-6)
    return row[0], round(row[5] * length_m, 3), round(row[4] * length_m * 1e3, 2)


def project_pipe(pre_di, length, V, nu, rho, dp_max, strategy="pressure"):
    di_values = sorted(pipe[3] * 1e-3 for pipe in cu_pipes)

    def calc(di):
        w = 4 * V / (np.pi * di**2)
        lam = float(friction_coefficient(di, w, nu))
        dp = lam * length * rho * w**2 * 0.5 / di
        if rho > 900:
            jg = np.sqrt(0.85 * 9.80665 * di * abs(rho * 1.1 - rho) / rho)
        else:
            jg = np.sqrt(0.85 * 9.80665 * di * abs(900 - rho) / rho)
        return w, jg, dp

    valid = []
    for di in di_values:
        w, jg, dp = calc(di)
        if w > jg and dp < dp_max:
            valid.append((di, w, jg, dp))

    if valid:
        return valid[-1] if strategy == "pressure" else valid[0]

    di = min(di_values, key=lambda x: abs(x - pre_di))
    w, jg, dp = calc(di)
    return di, w, jg, dp


def Q_brentq_h(p, h, x_min, x_max):
    def f(x):
        return cp.PropsSI("H", "P", p, "Q", x, FLUID) - h
    return brentq(f, x_min, x_max)


def T_brentq_s(p, s, T_min, T_max):
    def f(T):
        return cp.PropsSI("S", "T", T, "P", p, FLUID) - s
    return brentq(f, T_min, T_max)


def T_brentq_h(p, h, T_min, T_max):
    def f(T):
        return cp.PropsSI("H", "T", T, "P", p, FLUID) - h
    return brentq(f, T_min, T_max)


def T_0_X(T, T_0_original, h_4):
    p_0_dew = cp.PropsSI("P", "T", T, "Q", 1, FLUID)
    p_0_bubble = cp.PropsSI("P", "T", T, "Q", 0, FLUID)
    p_0 = (p_0_dew + p_0_bubble) / 2
    T_0_dew = cp.PropsSI("T", "P", p_0, "Q", 1, FLUID)
    T_0_bubble = cp.PropsSI("T", "P", p_0, "Q", 0, FLUID)
    h_0_dew = cp.PropsSI("H", "P", p_0, "Q", 1, FLUID)
    h_0_bubble = cp.PropsSI("H", "P", p_0, "Q", 0, FLUID)
    T_4 = -(h_0_dew - h_4) / (h_0_dew - h_0_bubble) * (T_0_dew - T_0_bubble) + T_0_dew
    T_0_neu = (T_4 + T_0_dew) / 2
    dt_0_X = T_0_neu - T_0_original
    T_0_Z = T - dt_0_X
    return T_0_Z, T_4, T_0_neu, T_0_dew, T_0_bubble, h_0_dew, h_0_bubble


def build_csv_bytes(df_results, df_points, df_pipes=None):
    csv_buffer = io.StringIO()
    csv_buffer.write(f"{APP_TITLE};{APP_VERSION}\n")
    csv_buffer.write("Systemdaten\n")
    df_results.to_csv(csv_buffer, index=False, sep=";")
    csv_buffer.write("\nKreislaufpunkte\n")
    df_points.to_csv(csv_buffer, index=False, sep=";")
    if df_pipes is not None and not df_pipes.empty:
        csv_buffer.write("\nRohrleitungsdimensionierung\n")
        df_pipes.to_csv(csv_buffer, index=False, sep=";")
    return csv_buffer.getvalue().encode("utf-8-sig")


def run_calculation(inputs):
    project = inputs["project"]
    mode = inputs["mode"]
    errors = []

    if mode == 0:
        Q_0 = inputs["Q_0"] * 1e3
        Q_c = 0.0
        V_com = 0.0
    elif mode == 1:
        Q_c = inputs["Q_c"] * 1e3
        Q_0 = 0.0
        V_com = 0.0
        if Q_c > 0:
            Q_c = -Q_c
    else:
        V_com = inputs["V_com"] / 3.6e3
        Q_0 = 0.0
        Q_c = 0.0

    T_gc_o_input = inputs["T_gc_o"]
    T_0_input = inputs["T_0"]
    p_gc_mode = inputs["p_gc_mode"]
    if p_gc_mode == 0:
        p_gc_input = 0.002 * T_gc_o_input**2 + 2.256 * T_gc_o_input - 0.17 * T_0_input + 4.9
    else:
        p_gc_input = inputs["p_gc"]

    dt_0h = inputs["dt_0h"]
    dt_sh = inputs["dt_sh"]

    if inputs["if_pipes"] == 0:
        l_hg = l_fl = l_sl = 1.0
    else:
        l_hg = inputs["l_hg"]
        l_fl = inputs["l_fl"]
        l_sl = inputs["l_sl"]

    T_gc_o = T_gc_o_input + 273.15
    T_0 = T_0_input + 273.15
    p_gc = p_gc_input * 1e5

    if T_gc_o < T_0:
        raise ValueError("Die Gaskühleraustrittstemperatur darf nicht unter der Verdampfungstemperatur liegen.")
    if (T_gc_o - T_0) < 20:
        errors.append("Die Differenz zwischen Gaskühleraustrittstemperatur und Verdampfungstemperatur ist sehr gering.")
    if Q_0 == 0 and mode == 0:
        Q_0 = 1
    if Q_c == 0 and mode == 1:
        Q_c = -1
    if V_com == 0 and mode == 2:
        V_com = 1 / 3.6e3

    cycle_mode = "transkritisch" if p_gc >= cp.PropsSI("PCRIT", FLUID) else "subkritisch"

    h_3 = cp.PropsSI("H", "T", T_gc_o, "P", p_gc, FLUID)
    h_4 = h_3
    T_2min = T_gc_o + 1
    T_2max = T_gc_o + 100

    p_0_dew_0100 = cp.PropsSI("P", "T", T_0, "Q", 1, FLUID)
    p_0_bubble_0100 = cp.PropsSI("P", "T", T_0, "Q", 0, FLUID)
    x_4min = 0.01
    x_4max = 0.99

    T_0_neu = 0.0
    T_0_iteration = T_0
    while abs(T_0_neu - T_0) > 0.01:
        T_0_Z, T_4, T_0_neu, T_0_dew, T_0_bubble, h_0_dew, h_0_bubble = T_0_X(T_0_iteration, T_0, h_4)
        T_0_iteration = T_0_Z
        p_0 = cp.PropsSI("P", "T", T_0_dew, "Q", 1, FLUID)
        x_4 = Q_brentq_h(p_0, h_4, x_4min, x_4max)

    T_0h = T_0_dew + dt_0h
    T_sh = T_0_dew + dt_0h + dt_sh

    pi = p_gc / p_0
    eta_is = (-0.020644445 + 0.68403852 * pi - 0.22147167 * pi**2 + 0.032145926 * pi**3 - 0.00178 * pi**4) if pi <= 5 else (0.821 - 0.0105 * pi)

    h_1 = cp.PropsSI("H", "P", p_0, "T", T_sh, FLUID)
    s_1 = cp.PropsSI("S", "P", p_0, "T", T_sh, FLUID)
    T_2s = T_brentq_s(p_gc, s_1, T_2min, T_2max)
    h_2s = cp.PropsSI("H", "P", p_gc, "T", T_2s, FLUID)
    h_2 = h_1 + (h_2s - h_1) / eta_is

    h_5 = cp.PropsSI("H", "P", p_0, "T", T_0h, FLUID)
    rho_com_in = cp.PropsSI("D", "P", p_0, "T", T_sh + 10, FLUID)
    q_c = h_3 - h_2
    q_0 = h_5 - h_4

    if mode == 0:
        m_R = Q_0 / q_0
        Q_c = m_R * q_c
        V_com = m_R / rho_com_in
    elif mode == 1:
        m_R = Q_c / q_c
        Q_0 = m_R * q_0
        V_com = m_R / rho_com_in
    else:
        m_R = V_com * rho_com_in
        Q_c = m_R * q_c
        Q_0 = m_R * q_0

    P_com = m_R * (h_2 - h_1)
    COP = -Q_c / P_com
    EER = Q_0 / P_com

    rho_1 = cp.PropsSI("D", "P", p_0, "T", T_sh, FLUID)
    T_end = T_brentq_h(p_gc, h_2, T_2min, T_2max)
    rho_2 = cp.PropsSI("D", "P", p_gc, "T", T_end, FLUID)
    rho_3 = cp.PropsSI("D", "P", p_gc, "T", T_gc_o, FLUID)

    V_1 = m_R / rho_1
    V_2 = m_R / rho_2
    V_3 = m_R / rho_3

    mu_1 = cp.PropsSI("V", "P", p_0, "T", T_sh, FLUID)
    mu_2 = cp.PropsSI("V", "P", p_gc, "T", T_end, FLUID)
    mu_3 = cp.PropsSI("V", "P", p_gc, "T", T_gc_o, FLUID)

    nu_1 = mu_1 / rho_1
    nu_2 = mu_2 / rho_2
    nu_3 = mu_3 / rho_3

    dp_max_hg_bar = 0.002 * (T_gc_o_input - 1) ** 2 + 2.256 * (T_gc_o_input - 1) - 0.17 * T_0_input + 4.9
    dp_max_hg = (p_gc_input - dp_max_hg_bar) * 1e5
    dp_max_fl = (p_gc_input - dp_max_hg_bar) * 1e5
    dp_max_sl = float(cp.PropsSI("P", "T", T_0_dew + 1, "Q", 1, FLUID) - p_0)

    pre_hg = float(find_next_bigger_pipe(np.sqrt(4 * V_2 / np.pi / 15)))
    pre_fl = float(find_next_bigger_pipe(np.sqrt(4 * V_3 / np.pi / 1)))
    pre_sl = float(find_next_bigger_pipe(np.sqrt(4 * V_1 / np.pi / 12)))

    di_hg_dp, w_hg_dp, jg_hg_dp, dp_hg_dp = project_pipe(pre_hg, l_hg * 1.4, V_2, nu_2, rho_2, dp_max_hg, strategy="pressure")
    di_hg_dm, w_hg_dm, jg_hg_dm, dp_hg_dm = project_pipe(pre_hg, l_hg * 1.4, V_2, nu_2, rho_2, dp_max_hg, strategy="diameter")
    di_fl_dp, w_fl_dp, jg_fl_dp, dp_fl_dp = project_pipe(pre_fl, l_fl * 1.4, V_3, nu_3, rho_3, dp_max_fl, strategy="pressure")
    di_fl_dm, w_fl_dm, jg_fl_dm, dp_fl_dm = project_pipe(pre_fl, l_fl * 1.4, V_3, nu_3, rho_3, dp_max_fl, strategy="diameter")
    di_sl_dp, w_sl_dp, jg_sl_dp, dp_sl_dp = project_pipe(pre_sl, l_sl * 1.4, V_1, nu_1, rho_1, dp_max_sl, strategy="pressure")
    di_sl_dm, w_sl_dm, jg_sl_dm, dp_sl_dm = project_pipe(pre_sl, l_sl * 1.4, V_1, nu_1, rho_1, dp_max_sl, strategy="diameter")

    d_hg_dp, A_M_hg_dp, V_i_hg_dp = pipe_geometry(di_hg_dp, l_hg)
    d_hg_dm, A_M_hg_dm, V_i_hg_dm = pipe_geometry(di_hg_dm, l_hg)
    d_fl_dp, A_M_fl_dp, V_i_fl_dp = pipe_geometry(di_fl_dp, l_fl)
    d_fl_dm, A_M_fl_dm, V_i_fl_dm = pipe_geometry(di_fl_dm, l_fl)
    d_sl_dp, A_M_sl_dp, V_i_sl_dp = pipe_geometry(di_sl_dp, l_sl)
    d_sl_dm, A_M_sl_dm, V_i_sl_dm = pipe_geometry(di_sl_dm, l_sl)

    df_points = pd.DataFrame({
        "Punkte": ["1", "2", "3", "4", "5", "0''", "0'"],
        "Temperatur [°C]": [round(T_sh - 273.15, 2), round(T_end - 273.15, 2), round(T_gc_o - 273.15, 2), round(T_4 - 273.15, 2), round(T_0h - 273.15, 2), round(T_0_dew - 273.15, 2), round(T_0_bubble - 273.15, 2)],
        "Druck [bar]": [round(p_0 / 1e5, 2), round(p_gc / 1e5, 2), round(p_gc / 1e5, 2), round(p_0 / 1e5, 2), round(p_0 / 1e5, 2), round(p_0 / 1e5, 2), round(p_0 / 1e5, 2)],
        "Spezifische Enthalpie [kJ/kg]": [round(h_1 / 1e3, 2), round(h_2 / 1e3, 2), round(h_3 / 1e3, 2), round(h_4 / 1e3, 2), round(h_5 / 1e3, 2), round(cp.PropsSI("H", "P", p_0, "Q", 1, FLUID) / 1e3, 2), round(cp.PropsSI("H", "P", p_0, "Q", 0, FLUID) / 1e3, 2)],
        "Dichte [kg/m3]": [round(rho_1, 2), round(rho_2, 2), round(rho_3, 2), round(cp.PropsSI("D", "P", p_0, "Q", x_4, FLUID), 2), round(cp.PropsSI("D", "P", p_0, "T", T_0h, FLUID), 2), round(cp.PropsSI("D", "P", p_0, "Q", 1, FLUID), 2), round(cp.PropsSI("D", "P", p_0, "Q", 0, FLUID), 2)],
        "Spezifische Entropie [kJ/kg/K]": [round(s_1 / 1e3, 4), round(cp.PropsSI("S", "P", p_gc, "T", T_end, FLUID) / 1e3, 4), round(cp.PropsSI("S", "P", p_gc, "T", T_gc_o, FLUID) / 1e3, 4), round(cp.PropsSI("S", "P", p_0, "Q", x_4, FLUID) / 1e3, 4), round(cp.PropsSI("S", "P", p_0, "T", T_0h, FLUID) / 1e3, 4), round(cp.PropsSI("S", "P", p_0, "Q", 1, FLUID) / 1e3, 2), round(cp.PropsSI("S", "P", p_0, "Q", 0, FLUID) / 1e3, 2)],
        "Dampfqualität [%]": ["", "", "", round(x_4 * 100, 2), "", 100, 0],
    })

    df_results = pd.DataFrame({
        "Parameter": [
            "Projekt",
            "Prozessbereich",
            "Leistungsaufnahme [kW]",
            "Wärmeleistung [kW]",
            "Kälteleistung [kW]",
            "Kältemittelmassenstrom [kg/s]",
            "Verdichtervolumenstrom [m3/h]",
            "Gaskühlerdruck [bar]",
            "COP",
            "EER",
        ],
        "Wert": [
            project,
            cycle_mode,
            round(P_com / 1e3, 2),
            round(Q_c / 1e3, 2),
            round(Q_0 / 1e3, 2),
            round(m_R, 6),
            round(V_com * 3.6e3, 2),
            round(p_gc / 1e5, 2),
            round(COP, 2),
            round(EER, 2),
        ],
    })

    df_pipes = pd.DataFrame([
        ["Heissgasleitung", "minimaler Druckverlust", d_hg_dp, l_hg, round(w_hg_dp, 2), round(jg_hg_dp, 2), round(dp_hg_dp / 1e5, 2), A_M_hg_dp, V_i_hg_dp],
        ["", "minimaler Durchmesser", d_hg_dm, l_hg, round(w_hg_dm, 2), round(jg_hg_dm, 2), round(dp_hg_dm / 1e5, 2), A_M_hg_dm, V_i_hg_dm],
        ["Flüssigkeitsleitung", "minimaler Druckverlust", d_fl_dp, l_fl, round(w_fl_dp, 2), round(jg_fl_dp, 2), round(dp_fl_dp / 1e5, 2), A_M_fl_dp, V_i_fl_dp],
        ["", "minimaler Durchmesser", d_fl_dm, l_fl, round(w_fl_dm, 2), round(jg_fl_dm, 2), round(dp_fl_dm / 1e5, 2), A_M_fl_dm, V_i_fl_dm],
        ["Saugleitung", "minimaler Druckverlust", d_sl_dp, l_sl, round(w_sl_dp, 2), round(jg_sl_dp, 2), round(dp_sl_dp / 1e5, 2), A_M_sl_dp, V_i_sl_dp],
        ["", "minimaler Durchmesser", d_sl_dm, l_sl, round(w_sl_dm, 2), round(jg_sl_dm, 2), round(dp_sl_dm / 1e5, 2), A_M_sl_dm, V_i_sl_dm],
    ], columns=[
        "Leitung",
        "Berechnungsmethode",
        "Durchmesser [mm]",
        "Länge [m]",
        "Strömungsgeschwindigkeit [m/s]",
        "Jacobs-Geschwindigkeit [m/s]",
        "Druckverlust [bar]",
        "Aussenmantelfläche [m2]",
        "Innenvolumen [dm3]",
    ])

    return df_points, df_results, df_pipes, errors


st.markdown(
    f"""
    <div style='display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:0.2rem;'>
        <div style='font-size:3rem; font-weight:700; line-height:1.1;'>{APP_TITLE}</div>
        <div style='color:#9ca3af; font-size:1rem; line-height:1.1;'>{APP_VERSION}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Berechnet die wichtigsten Zustände und Kennwerte eines CO2-Kreislaufs für subkritischen und transkritischen Betrieb.")

if "co2result" not in st.session_state:
    st.session_state.co2result = None

col1, col2 = st.columns([1.05, 1.3])

with col1:
    st.subheader("Eingabe")
    project = st.text_input("Projekt", value="Projekt")
    
    row2col1, row2col2 = st.columns(2)
    with row2col1:
        mode = st.selectbox("Eingabemodus", ["Kälteleistung", "Wärmeleistung", "Verdichtervolumenstrom"])
    with row2col2:
        if mode == "Kälteleistung":
            q0 = st.number_input("Kälteleistung [kW]", value=10.0)
            qc = 0.0
            vcom = 0.0
        elif mode == "Wärmeleistung":
            qc = st.number_input("Wärmeleistung [kW]", value=14.0)
            q0 = 0.0
            vcom = 0.0
        else:
            vcom = st.number_input("Verdichtervolumenstrom [m3/h]", value=15.0)
            q0 = 0.0
            qc = 0.0

    row3col1, row3col2 = st.columns(2)
    with row3col1:
        tgc = st.number_input("Gaskühleraustrittstemperatur [°C]", value=35.0)
    with row3col2:
        
        pgc_mode = st.selectbox("Gaskühlerdruck", ["Automatisch", "Manuell"])

    row4col1, row4col2 = st.columns(2)
    with row4col1:
        pgc = st.number_input("Gaskühlerdruck [bar]", value=90.0, disabled=(pgc_mode == "Automatisch"))
    with row4col2:
        t0 = st.number_input("Verdampfungstemperatur [°C]", value=-10.0)

    row5col1, row5col2 = st.columns(2)
    with row5col1:
        dt0h = st.number_input("Verdampferüberhitzung [K]", value=6.0)
    with row5col2:
        dtsh = st.number_input("Saugleitungsüberhitzung [K]", value=9.0)
    ifpipes = st.selectbox("Rohrleitungsdimensionierung", ["Nein", "Ja"])
    pipeinputs = ifpipes == "Ja"
    row5col1, row5col2, row5col3 = st.columns(3)
    with row5col1:
        lhg = st.number_input("Heissgasleitungslänge [m]", value=5.0) if pipeinputs else 5.0
    with row5col2:
        lfl = st.number_input("Flüssigkeitsleitungslänge [m]", value=2.5) if pipeinputs else 5.0
    with row5col3:
        lsl = st.number_input("Saugleitungslänge [m]", value=3.0) if pipeinputs else 5.0

    run = st.button("Berechnen", use_container_width=True)

data = None
if run:
    try:
        points, results, pipes, errors = run_calculation({
            "project": project,
            "mode": ["Kälteleistung", "Wärmeleistung", "Verdichtervolumenstrom"].index(mode),
            "Q_0": q0,
            "Q_c": qc,
            "V_com": vcom,
            "T_gc_o": tgc,
            "T_0": t0,
            "p_gc_mode": 0 if pgc_mode == "Automatisch" else 1,
            "p_gc": pgc,
            "dt_0h": dt0h,
            "dt_sh": dtsh,
            "if_pipes": 0 if ifpipes == "Nein" else 1,
            "l_hg": lhg,
            "l_fl": lfl,
            "l_sl": lsl,
        })
        st.session_state.co2result = {
            "project": project,
            "points": points,
            "results": results,
            "pipes": pipes,
            "errors": errors,
            "csv": build_csv_bytes(results, points, pipes if ifpipes == "Ja" else None),
        }
    except Exception as e:
        st.session_state.co2result = {"exception": str(e)}

    if "exception" in st.session_state.co2result:
        st.error(st.session_state.co2result["exception"])
    else:
        data = st.session_state.co2result




with col2:
    result_container = st.container()
    with result_container:
        st.subheader("Ergebnis")
        data = st.session_state.get("co2result")
        if data is not None:
            st.dataframe(data["results"], use_container_width=True, hide_index=True)
            st.download_button(
                "CSV herunterladen",
                data=data["csv"],
                file_name=f"{data['project'].replace(' ', '_')}_Einfache-CO2-Kreislaufberechnung.csv",
                mime="text/csv",
                use_container_width=True,
                )
        if st.session_state.co2result is None:
            st.info("Eingaben setzen und auf Berechnen klicken.")

if st.session_state.co2result is not None:
    if "exception" in st.session_state.co2result:
        st.error(st.session_state.co2result["exception"])
    else:
        data = st.session_state.co2result
        st.subheader("Kreislaufpunkte")
        st.dataframe(data["points"], use_container_width=True, hide_index=True)
        if ifpipes == "Ja":
            st.subheader("Rohrleitungsdimensionierung")
            st.dataframe(data["pipes"], use_container_width=True, hide_index=True)
        if data["errors"]:
            st.warning("\n".join(data["errors"]))

st.divider()
with st.expander("Anleitung"):
    st.markdown(
        """
Dieses Tool überführt die bestehende CO2-Kreislaufberechnung in den Streamlit-Stil der einfachen Kreislaufberechnung.

Berücksichtigt wird subkritischer und transkritischer CO2-Betrieb.

**Bedeutung der Kreislaufpunkte:**

        - **1:** Verdichtereingang nach Überhitzung von Verdampfer und Saugleitung
        - **2:** Verdichterausgang und Gaskühlereintritt
        - **3:** Gaskühleraustritt und Expansionsventileingang
        - **4:** Expansionsventilausgang und Verdampfereintritt
        - **5:** Verdampferausgang und Eingang Saugleitung
        - **0''**: Gesättigtes Gas bei Verdampfungsdruck
        - **0'**: Flüssigkeit bei Verdampfungsdruck

Repository: https://github.com/div-ne/Einfache-CO2-Kreislaufberechnung
        """
    )
