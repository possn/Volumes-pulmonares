import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ============================================================
# OUTPUT
# ============================================================
OUT = "Spirograma_Dinamico_Tidal_Manobras_Forcadas_CORRIGIDO.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 20
DURATION_S = 60

W, H = 12.8, 7.2
DPI = 120

# ============================================================
# DIDACTIC VOLUMES (mL)
# Valores típicos adultos (aprox). Ajusta se quiseres.
# Importante: garantir coerência:
#   FRC = RV + ERV
# ============================================================
RV  = 1200.0   # Volume Residual
ERV = 1100.0   # Volume de Reserva Expiratório
FRC = RV + ERV # Capacidade Residual Funcional

VT  = 500.0    # Volume Corrente (amplitude do tidal em torno da FRC)
IRV = 3000.0   # Volume de Reserva Inspiratório (acima do topo do VT)
TLC = 6000.0   # Capacidade Pulmonar Total

# Derived (didactic)
IC = VT + IRV                 # Capacidade Inspiratória (do fim exp tranquila até topo máximo)
VC = TLC - RV                 # Capacidade Vital
# Nota: "Capacidade pulmonar total" = TLC (linha superior)

# ============================================================
# TIMING (s) - loop didáctico
# Tidal 10s -> pausa 1s -> exp forçada 4s -> pausa 1s -> insp forçada 4s -> pausa 1s -> tidal 9s
# Total ciclo = 30s; em 60s repete 2x
# ============================================================
T_TIDAL_1 = 10.0
T_HOLD_1  = 1.0
T_FEXP    = 4.0
T_HOLD_2  = 1.0
T_FINS    = 4.0
T_HOLD_3  = 1.0
T_TIDAL_2 = 9.0

T_CYCLE = T_TIDAL_1 + T_HOLD_1 + T_FEXP + T_HOLD_2 + T_FINS + T_HOLD_3 + T_TIDAL_2

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5*np.cos(np.pi*x)

def phase_in_cycle(tau):
    a = T_TIDAL_1
    b = a + T_HOLD_1
    c = b + T_FEXP
    d = c + T_HOLD_2
    e = d + T_FINS
    f = e + T_HOLD_3
    if tau < a:
        return "tidal1", tau / max(T_TIDAL_1, 1e-6)
    if tau < b:
        return "hold1", (tau - a) / max(T_HOLD_1, 1e-6)
    if tau < c:
        return "fexp", (tau - b) / max(T_FEXP, 1e-6)
    if tau < d:
        return "hold2", (tau - c) / max(T_HOLD_2, 1e-6)
    if tau < e:
        return "fins", (tau - d) / max(T_FINS, 1e-6)
    if tau < f:
        return "hold3", (tau - e) / max(T_HOLD_3, 1e-6)
    return "tidal2", (tau - f) / max(T_TIDAL_2, 1e-6)

# ============================================================
# VOLUME WAVEFORM (mL) -- didactic curve:
# - tidal: sinusoidal around FRC with amplitude VT/2
# - forced expiration: drops towards RV+small margin
# - forced inspiration: rises towards TLC - small margin
# ============================================================
FEXP_TARGET = RV + 150.0
FINS_TARGET = TLC - 200.0

def volume_of_tau(tau):
    ph, x = phase_in_cycle(tau)

    # baseline tidal waveform around FRC
    def tidal(xlocal, cycles=2.0):
        # around FRC: +/- VT/2
        return FRC + (VT/2.0)*np.sin(2*np.pi*cycles*xlocal)

    if ph in ("tidal1", "tidal2"):
        cycles = 2.5 if ph == "tidal1" else 2.0
        return tidal(x, cycles=cycles)

    if ph == "hold1":
        return tidal(1.0, cycles=2.5)  # end of tidal1

    if ph == "fexp":
        v0 = tidal(1.0, cycles=2.5)
        return v0 + (FEXP_TARGET - v0)*smoothstep(x)

    if ph == "hold2":
        return FEXP_TARGET

    if ph == "fins":
        v0 = FEXP_TARGET
        return v0 + (FINS_TARGET - v0)*smoothstep(x)

    if ph == "hold3":
        return FINS_TARGET

    return FRC

# ============================================================
# Render helpers
# ============================================================
def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def history(t, window, fps):
    t0 = max(0.0, t - window)
    n = int(max(120, min(int(window*fps), int((t - t0)*fps + 1))))
    return np.linspace(t0, t, n)

# ============================================================
# FIGURE LAYOUT
# ============================================================
fig = plt.figure(figsize=(W, H), dpi=DPI)

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "24"]
)

WINDOW = 30.0  # mostrar 30s (1 ciclo) no gráfico
total_frames = int(DURATION_S * FPS)

for i in range(total_frames):
    t = i / FPS
    tau = t % T_CYCLE

    th = history(t, WINDOW, FPS)
    vol = np.array([volume_of_tau(tt % T_CYCLE) for tt in th])

    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    # Background bands (VR / ERV / tidal zone / IRV)
    ax.axhspan(0, RV, facecolor="#f5e6b8", alpha=0.75)                 # VR zone
    ax.axhspan(RV, FRC, facecolor="#edd9a2", alpha=0.75)               # ERV zone (since FRC=RV+ERV)
    ax.axhspan(FRC - VT/2, FRC + VT/2, facecolor="#efe3c3", alpha=0.55) # tidal zone
    ax.axhspan(FRC + VT/2, TLC, facecolor="#f1e7cc", alpha=0.45)       # above tidal up to TLC

    # Main trace
    ax.plot(th, vol, lw=3.2, color="#d4382c")

    # Current point
    v_now = volume_of_tau(tau)
    ax.scatter([t], [v_now], s=80, color="#2563eb", zorder=5)

    # Axes formatting
    ax.set_xlim(th[0], th[-1])
    ax.set_ylim(0, TLC)
    ax.set_yticks(np.arange(0, TLC+1, 1000))
    ax.set_ylabel("Volume pulmonar (mL)", fontsize=12, weight="bold")
    ax.set_xlabel("Tempo", fontsize=12, weight="bold")
    ax.set_title("Spirograma dinâmico (tidal + manobras forçadas) — loop didáctico",
                 fontsize=13, weight="bold", pad=10)
    ax.grid(True, alpha=0.15)

    # Reference lines
    ax.axhline(RV,  color="#111827", lw=2.2)                 # top of RV
    ax.axhline(FRC, color="#111827", lw=1.8, ls="--", alpha=0.7)  # FRC dashed
    ax.axhline(TLC, color="#111827", lw=2.2)                 # TLC top

    # Labels (non-overlapping, corrected physiology)
    ax.text(th[0] + 0.5, TLC - 120, "Capacidade pulmonar total", fontsize=10, color="#111827", weight="bold")

    ax.text(th[0] + 0.5, RV + 90, "Volume residual", fontsize=10, color="#111827", weight="bold")
    ax.text(th[0] + 0.5, FRC + 60, "Capacidade residual funcional", fontsize=10, color="#111827", weight="bold")

    # Arrow: ERV (between RV and FRC)
    x_erv = th[0] + 7.5
    ax.annotate("", xy=(x_erv, FRC), xytext=(x_erv, RV),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(x_erv + 0.6, (RV+FRC)/2, "Volume de reserva\nexpiratório",
            fontsize=10, va="center", color="#111827")

    # Arrow: IRV (from top of tidal to near TLC)
    x_irv = th[0] + 13.5
    top_tidal = FRC + VT/2
    ax.annotate("", xy=(x_irv, TLC-200), xytext=(x_irv, top_tidal),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(x_irv + 0.6, (top_tidal + (TLC-200))/2, "Volume de reserva\ninspiratório",
            fontsize=10, va="center", color="#111827")

    # Arrow: VT (tidal)
    x_vt = th[0] + 18.5
    ax.annotate("", xy=(x_vt, FRC + VT/2), xytext=(x_vt, FRC - VT/2),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(x_vt + 0.6, FRC, "Volume\ncorrente", fontsize=10, va="center", color="#111827")

    # Right side capacities: IC, VC, TLC (stacked at far right)
    xr = th[-1] - 1.0

    # IC: from FRC to TLC-200
    ax.annotate("", xy=(xr, TLC-200), xytext=(xr, FRC),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(xr + 0.4, (FRC + (TLC-200))/2, "Capacidade\ninspiratória",
            fontsize=10, va="center", color="#111827")

    # VC: from RV to TLC
    xr2 = xr + 0.8
    ax.annotate("", xy=(xr2, TLC), xytext=(xr2, RV),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(xr2 + 0.4, (RV + TLC)/2, "Capacidade\nvital",
            fontsize=10, va="center", color="#111827")

    # TLC: from 0 to TLC
    xr3 = xr2 + 0.8
    ax.annotate("", xy=(xr3, TLC), xytext=(xr3, 0),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax.text(xr3 + 0.4, TLC/2, "Capacidade\npulmonar\ntotal",
            fontsize=10, va="center", color="#111827")

    # Footer note (short)
    ax.text(th[0] + 0.6, 120, "Correção-chave: CRF = VR + VRE (ERV)", fontsize=10,
            color="#111827", alpha=0.9)

    fig.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
