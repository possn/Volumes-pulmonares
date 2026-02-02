import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ============================================================
# OUTPUT
# ============================================================
OUT = "Spirograma_Dinamico_Tidal_Manobras_Forcadas_AULA_LIMPA.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 20
DURATION_S = 60
W, H = 12.8, 7.2
DPI = 100  # 1280x720

# ============================================================
# DIDACTIC VOLUMES (mL) — coerentes por construção
# ============================================================
TLC = 6000.0   # CPT (TLC)
RV  = 1200.0   # VR (RV)
ERV = 1100.0   # VRE (ERV)
VT  = 500.0    # VT (VT)

FRC = RV + ERV                 # CRF (FRC)
IRV = TLC - (FRC + VT)         # VRI (IRV) coerente
IC  = TLC - FRC                # CI (IC)
VC  = TLC - RV                 # CV (VC)

IRV = max(IRV, 0.0)

# ============================================================
# TIMING (s) — ciclo fixo 30 s (repete 2x em 60 s)
# ============================================================
T_TIDAL_1 = 12.0
T_HOLD_1  = 1.0
T_FEXP    = 6.0
T_HOLD_2  = 1.0
T_FINS    = 6.0
T_HOLD_3  = 1.0
T_TIDAL_2 = 3.0

T_CYCLE = T_TIDAL_1 + T_HOLD_1 + T_FEXP + T_HOLD_2 + T_FINS + T_HOLD_3 + T_TIDAL_2  # 30s

FEXP_TARGET = RV
FINS_TARGET = TLC

def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5*np.cos(np.pi*x)

def ease_out_fast_then_slow(x: float) -> float:
    # rápido no início, abranda no fim
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - (1.0 - x)**2

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
    VT correcto: vai de CRF -> CRF+VT -> CRF (não desce para VRE).
    """
    theta = 2.0 * np.pi * breaths * xlocal
    return FRC + (VT / 2.0) * (1.0 - np.cos(theta))  # [FRC .. FRC+VT]

def volume_of_tau(tau: float) -> float:
    ph, x = phase_in_cycle(tau)

    if ph == "tidal1":
        return tidal_volume(x, breaths=3.0)

    if ph == "hold1":
        return FRC

    if ph == "fexp":
        v0 = FRC + VT
        k = ease_out_fast_then_slow(x)
        return v0 + (FEXP_TARGET - v0) * k

    if ph == "hold2":
        return FEXP_TARGET

    if ph == "fins":
        v0 = FEXP_TARGET
        k = ease_out_fast_then_slow(x)
        return v0 + (FINS_TARGET - v0) * k

    if ph == "hold3":
        return FINS_TARGET

    if ph == "tidal2":
        # 1ª metade: volta de TLC -> CRF
        if x < 0.5:
            k = ease_out_fast_then_slow(x / 0.5)
            return TLC + (FRC - TLC) * k
        # 2ª metade: 1 tidal
        xx = (x - 0.5) / 0.5
        return tidal_volume(xx, breaths=1.0)

    return FRC

# ============================================================
# Precompute cycle curve
# ============================================================
N_CURVE = 2400
t_curve = np.linspace(0.0, T_CYCLE, N_CURVE)
v_curve = np.array([volume_of_tau(tt) for tt in t_curve])

PHASE_LABEL = {
    "tidal1": "Respiração tranquila (VT)",
    "hold1":  "Pausa em CRF",
    "fexp":   "Expiração forçada até VR",
    "hold2":  "Pausa em VR",
    "fins":   "Inspiração forçada até CPT",
    "hold3":  "Pausa em CPT",
    "tidal2": "Retorno a CRF + retoma VT",
}

# ============================================================
# Render helpers
# ============================================================
def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

# ============================================================
# STYLE HELPERS
# ============================================================
def label_box(ax, x, y, text, ha="left", va="center", fs=11, weight="bold", alpha=0.92):
    ax.text(
        x, y, text,
        ha=ha, va=va, fontsize=fs, weight=weight, color="#111827",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#e5e7eb", alpha=alpha),
        zorder=20
    )

# ============================================================
# FIGURE + VIDEO WRITER
# ============================================================
fig = plt.figure(figsize=(W, H), dpi=DPI)
writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p"]
)

total_frames = int(DURATION_S * FPS)

for i in range(total_frames):
    t = i / FPS
    tau = t % T_CYCLE
    ph, _ = phase_in_cycle(tau)

    # progresso até tau
    mask = t_curve <= tau
    t_prog = t_curve[mask]
    v_prog = v_curve[mask]
    v_now = volume_of_tau(tau)

    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    # =========================
    # BACKGROUND BANDS (claras e correctas)
    # =========================
    ax.axhspan(0, RV, facecolor="#f1d9a6", alpha=0.75)                   # VR
    ax.axhspan(RV, FRC, facecolor="#edd09a", alpha=0.70)                 # VRE
    ax.axhspan(FRC, FRC + VT, facecolor="#f2e6c8", alpha=0.65)           # VT
    ax.axhspan(FRC + VT, TLC, facecolor="#f6f0df", alpha=0.85)           # VRI

    # =========================
    # CURVA (ref cinza + progresso vermelho)
    # =========================
    ax.plot(t_curve, v_curve, lw=2.2, color="#9ca3af", alpha=0.40, zorder=2)
    ax.plot(t_prog, v_prog, lw=3.6, color="#d4382c", zorder=4)
    ax.scatter([tau], [v_now], s=85, color="#2563eb", zorder=6)

    # =========================
    # AXES / GRID
    # =========================
    ax.set_xlim(0, T_CYCLE)
    ax.set_ylim(0, TLC)
    ax.set_yticks(np.arange(0, TLC + 1, 1000))
    ax.set_ylabel("Volume pulmonar (mL)", fontsize=13, weight="bold")
    ax.set_xlabel("Tempo (s)", fontsize=13, weight="bold")
    ax.grid(True, alpha=0.15)

    ax.set_title(
        "Spirograma dinâmico (tidal + manobras forçadas) — loop didáctico",
        fontsize=15, weight="bold", pad=12
    )

    # =========================
    # LINHAS DE REFERÊNCIA (CRF tem de saltar à vista)
    # =========================
    ax.axhline(RV, color="#111827", lw=2.2, zorder=3)
    ax.axhline(FRC, color="#111827", lw=2.6, ls="--", alpha=0.85, zorder=3)
    ax.axhline(FRC + VT, color="#111827", lw=1.6, ls=":", alpha=0.70, zorder=3)
    ax.axhline(TLC, color="#111827", lw=2.2, zorder=3)

    # =========================
    # RÓTULOS PRINCIPAIS (sem poluição)
    # =========================
    # VR bem evidente na faixa inferior
    label_box(ax, 0.6, RV * 0.45, "VR (RV)\nVolume residual", fs=12)

    # VRE no meio (RV->CRF)
    label_box(ax, 0.6, RV + (FRC - RV) * 0.55, "VRE (ERV)\nReserva expiratória", fs=12)

    # CRF como âncora (seta para a linha tracejada)
    ax.annotate(
        "CRF (FRC) = VR + VRE",
        xy=(2.5, FRC),
        xytext=(0.7, FRC + 260),
        fontsize=12, weight="bold", color="#111827",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#e5e7eb", alpha=0.92),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#111827"),
        zorder=20
    )

    # VT (CRF -> CRF+VT)
    x_vt = 8.0
    ax.annotate("", xy=(x_vt, FRC + VT), xytext=(x_vt, FRC),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    label_box(ax, x_vt + 0.4, FRC + VT/2, "VT\nRespiração tranquila", fs=11)

    # VRI (CRF+VT -> TLC)
    x_irv = 11.2
    ax.annotate("", xy=(x_irv, TLC), xytext=(x_irv, FRC + VT),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    label_box(ax, x_irv + 0.4, (TLC + (FRC + VT)) / 2, "VRI (IRV)\nReserva inspiratória", fs=11)

    # CPT label no topo (limpo)
    label_box(ax, 0.6, TLC - 140, "CPT (TLC)\nCapacidade pulmonar total", fs=12)

    # =========================
    # CAPACIDADES À DIREITA (separadas, claras, sem colisões)
    # =========================
    xr = T_CYCLE - 1.7

    # CI: FRC -> TLC
    ax.annotate("", xy=(xr, TLC), xytext=(xr, FRC),
                arrowprops=dict(arrowstyle="<->", lw=2.4, color="#111827"))
    ax.text(xr + 0.35, (FRC + TLC) / 2,
            "CI (IC)\ncapacidade\ninspiratória",
            fontsize=11, va="center", color="#111827",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#e5e7eb", alpha=0.92),
            zorder=20)

    # CV: RV -> TLC (um pouco mais à direita)
    xr2 = xr + 0.6
    ax.annotate("", xy=(xr2, TLC), xytext=(xr2, RV),
                arrowprops=dict(arrowstyle="<->", lw=2.4, color="#111827"))
    ax.text(xr2 + 0.35, (RV + TLC) / 2,
            "CV (VC)\ncapacidade\nvital",
            fontsize=11, va="center", color="#111827",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#e5e7eb", alpha=0.92),
            zorder=20)

    # CPT: 0 -> TLC (mais à direita ainda, só a seta + rótulo curto)
    xr3 = xr2 + 0.6
    ax.annotate("", xy=(xr3, TLC), xytext=(xr3, 0),
                arrowprops=dict(arrowstyle="<->", lw=2.4, color="#111827"))
    ax.text(xr3 + 0.32, TLC * 0.50, "CPT\n(TLC)",
            fontsize=11, va="center", color="#111827",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#e5e7eb", alpha=0.92),
            zorder=20)

    # =========================
    # BADGE DE FASE (top-left)
    # =========================
    ax.text(
        0.02, 0.98,
        f"Fase: {PHASE_LABEL.get(ph, ph)}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12.5, weight="bold", color="#111827",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#e5e7eb", alpha=0.95),
        zorder=30
    )

    fig.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
