import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ============================================================
# OUTPUT
# ============================================================
OUT = "Spirograma_Dinamico_Tidal_Manobras_Forcadas_FIX.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 20
DURATION_S = 60

W, H = 12.8, 7.2
DPI = 100  # 12.8*100 x 7.2*100 => 1280x720

# ============================================================
# DIDACTIC VOLUMES (mL) — coerentes por construção
# ============================================================
TLC = 6000.0   # Capacidade Pulmonar Total (CPT)
RV  = 1200.0   # Volume Residual (VR)
ERV = 1100.0   # Volume de Reserva Expiratório (VRE)
VT  = 500.0    # Volume Corrente (VT)

FRC = RV + ERV                         # CRF = VR + VRE
IRV = TLC - (FRC + VT)                 # VRI coerente (do fim insp tranquila até TLC)
IC  = TLC - FRC                        # Capacidade Inspiratória (CI)
VC  = TLC - RV                         # Capacidade Vital (CV)

# Guardas de coerência (sem crash agressivo; só “clip” se algo estiver impossível)
if IRV < 0:
    IRV = 0.0

# ============================================================
# TIMING (s) — ciclo didáctico estável (sem scroll)
# ============================================================
T_TIDAL_1 = 12.0   # respiração tranquila
T_HOLD_1  = 1.0
T_FEXP    = 6.0    # expiração forçada até RV (rápida no início, depois abranda)
T_HOLD_2  = 1.0
T_FINS    = 6.0    # inspiração forçada até TLC
T_HOLD_3  = 1.0
T_TIDAL_2 = 3.0    # retoma tranquila (curto) antes de reiniciar

T_CYCLE = T_TIDAL_1 + T_HOLD_1 + T_FEXP + T_HOLD_2 + T_FINS + T_HOLD_3 + T_TIDAL_2

# Targets das manobras
FEXP_TARGET = RV
FINS_TARGET = TLC

def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5*np.cos(np.pi*x)

def ease_out_fast_then_slow(x: float) -> float:
    """
    Curva com queda/subida rápida inicial e desaceleração no fim.
    Mantém 0..1, mais 'fisiológica' para manobras forçadas.
    """
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - (1.0 - x)**2  # quadrática ease-out

def phase_in_cycle(tau):
    a = T_TIDAL_1
    b = a + T_HOLD_1
    c = b + T_FEXP
    d = c + T_HOLD_2
    e = d + T_FINS
    f = e + T_HOLD_3
    if tau < a:
        return "tidal1", tau / max(T_TIDAL_1, 1e-9)
    if tau < b:
        return "hold1", (tau - a) / max(T_HOLD_1, 1e-9)
    if tau < c:
        return "fexp", (tau - b) / max(T_FEXP, 1e-9)
    if tau < d:
        return "hold2", (tau - c) / max(T_HOLD_2, 1e-9)
    if tau < e:
        return "fins", (tau - d) / max(T_FINS, 1e-9)
    if tau < f:
        return "hold3", (tau - e) / max(T_HOLD_3, 1e-9)
    return "tidal2", (tau - f) / max(T_TIDAL_2, 1e-9)

def tidal_volume(xlocal: float, breaths: float) -> float:
    """
    Respiração tranquila "clássica" no spirograma:
    começa na CRF, sobe até CRF+VT, volta à CRF.
    Usa meia-coseno (não passa abaixo da CRF).
    """
    # 0..1 -> breaths ciclos
    theta = 2.0 * np.pi * breaths * xlocal
    return FRC + (VT / 2.0) * (1.0 - np.cos(theta))  # [FRC .. FRC+VT]

def volume_of_tau(tau: float) -> float:
    ph, x = phase_in_cycle(tau)

    # Tidal 1: ~3 ciclos em 12 s (15/min)
    if ph == "tidal1":
        return tidal_volume(x, breaths=3.0)

    # Hold 1: fim da expiração tranquila (CRF)
    if ph == "hold1":
        # termina em CRF (no fim de ciclos inteiros, volta a CRF)
        return FRC

    # Forced expiration: parte do fim da inspiração tranquila (CRF+VT)
    if ph == "fexp":
        v0 = FRC + VT
        k = ease_out_fast_then_slow(x)
        return v0 + (FEXP_TARGET - v0) * k

    if ph == "hold2":
        return FEXP_TARGET

    # Forced inspiration: do RV até TLC
    if ph == "fins":
        v0 = FEXP_TARGET
        k = ease_out_fast_then_slow(x)
        return v0 + (FINS_TARGET - v0) * k

    if ph == "hold3":
        return FINS_TARGET

    # Tidal 2 (curto): 1 ciclo para regressar à CRF e reiniciar
    if ph == "tidal2":
        # começa em TLC; queremos cair para CRF e depois fazer 1 ciclo pequeno CRF->CRF+VT->CRF.
        # Fazemos um “retorno” rápido a CRF na primeira metade, depois 1 ciclo.
        if x < 0.5:
            k = ease_out_fast_then_slow(x / 0.5)
            return TLC + (FRC - TLC) * k
        else:
            xx = (x - 0.5) / 0.5
            return tidal_volume(xx, breaths=1.0)

    return FRC

# ============================================================
# Precompute one-cycle curve (for stable plot)
# ============================================================
N_CURVE = 2000
t_curve = np.linspace(0.0, T_CYCLE, N_CURVE)
v_curve = np.array([volume_of_tau(tt) for tt in t_curve])

# ============================================================
# Render helpers
# ============================================================
def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

# ============================================================
# FIGURE + VIDEO WRITER
# ============================================================
fig = plt.figure(figsize=(W, H), dpi=DPI)

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=[
        "-preset", "ultrafast",
        "-crf", "24",
        "-r", str(FPS),          # força FPS real
        "-pix_fmt", "yuv420p"    # compatibilidade (iPhone/Keynote/etc.)
    ]
)

total_frames = int(DURATION_S * FPS)

# Layout constants (para evitar sobreposições)
LABEL_FS = 10
TITLE_FS = 13

for i in range(total_frames):
    t = i / FPS
    tau = t % T_CYCLE

    # progresso: até tau (num ciclo)
    mask = t_curve <= tau
    t_prog = t_curve[mask]
    v_prog = v_curve[mask]
    v_now = volume_of_tau(tau)

    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    # Zonas de fundo (correctas)
    ax.axhspan(0, RV, facecolor="#f5e6b8", alpha=0.78)             # VR
    ax.axhspan(RV, FRC, facecolor="#edd9a2", alpha=0.78)           # VRE (ERV)
    ax.axhspan(FRC, FRC + VT, facecolor="#efe3c3", alpha=0.62)     # VT
    ax.axhspan(FRC + VT, TLC, facecolor="#f1e7cc", alpha=0.48)     # VRI (IRV)

    # Curva completa (referência) e progresso (realce)
    ax.plot(t_curve, v_curve, lw=2.2, color="#6b7280", alpha=0.35)
    ax.plot(t_prog, v_prog, lw=3.4, color="#d4382c")

    # Ponto actual
    ax.scatter([tau], [v_now], s=85, color="#2563eb", zorder=6)

    # Eixos fixos e limpos
    ax.set_xlim(0, T_CYCLE)
    ax.set_ylim(0, TLC)
    ax.set_yticks(np.arange(0, TLC + 1, 1000))
    ax.set_ylabel("Volume pulmonar (mL)", fontsize=12, weight="bold")
    ax.set_xlabel("Tempo (s)", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.14)

    ax.set_title("Spirograma dinâmico (tidal + manobras forçadas) — loop didáctico",
                 fontsize=TITLE_FS, weight="bold", pad=10)

    # Linhas de referência principais
    ax.axhline(RV,  color="#111827", lw=2.2)                        # topo do VR
    ax.axhline(FRC, color="#111827", lw=1.8, ls="--", alpha=0.75)   # CRF
    ax.axhline(FRC + VT, color="#111827", lw=1.2, ls=":", alpha=0.75)  # topo do VT (fim insp tranquila)
    ax.axhline(TLC, color="#111827", lw=2.2)                        # TLC

    # Labels fixos (sem colisões)
    ax.text(0.35, TLC - 140, "CPT (TLC) — Capacidade pulmonar total",
            fontsize=LABEL_FS, color="#111827", weight="bold")

    ax.text(0.35, RV + 90, "VR (RV) — Volume residual",
            fontsize=LABEL_FS, color="#111827", weight="bold")

    ax.text(0.35, FRC + 70, "CRF (FRC) = VR + VRE",
            fontsize=LABEL_FS, color="#111827", weight="bold")

    # Setas de volumes (esquerda/médio)
    x_erv = 4.0
    ax.annotate("", xy=(x_erv, FRC), xytext=(x_erv, RV),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(x_erv + 0.5, (RV + FRC) / 2.0, "VRE (ERV)\nreserva expiratória",
            fontsize=LABEL_FS, va="center", color="#111827")

    x_vt = 7.0
    ax.annotate("", xy=(x_vt, FRC + VT), xytext=(x_vt, FRC),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(x_vt + 0.5, FRC + VT/2.0, "VT\narespiração tranquila",
            fontsize=LABEL_FS, va="center", color="#111827")

    x_irv = 10.0
    ax.annotate("", xy=(x_irv, TLC), xytext=(x_irv, FRC + VT),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(x_irv + 0.5, (TLC + (FRC + VT)) / 2.0, "VRI (IRV)\nreserva inspiratória",
            fontsize=LABEL_FS, va="center", color="#111827")

    # Capacidades à direita (bem espaçadas)
    xr = T_CYCLE - 1.3
    ax.annotate("", xy=(xr, TLC), xytext=(xr, FRC),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(xr + 0.35, (FRC + TLC) / 2.0, "CI (IC)\ncapacidade inspiratória",
            fontsize=LABEL_FS, va="center", color="#111827")

    xr2 = xr + 0.7
    ax.annotate("", xy=(xr2, TLC), xytext=(xr2, RV),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(xr2 + 0.35, (RV + TLC) / 2.0, "CV (VC)\ncapacidade vital",
            fontsize=LABEL_FS, va="center", color="#111827")

    xr3 = xr2 + 0.7
    ax.annotate("", xy=(xr3, TLC), xytext=(xr3, 0),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(xr3 + 0.35, TLC / 2.0, "CPT (TLC)",
            fontsize=LABEL_FS, va="center", color="#111827")

    # Legenda curta (sem “estragar” layout)
    ax.text(0.35, 120,
            "Correções: VR abaixo de VRE; CRF = VR + VRE; VT entre CRF e CRF+VT.",
            fontsize=10, color="#111827", alpha=0.92)

    fig.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
