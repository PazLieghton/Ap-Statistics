# -*- coding: utf-8 -*-
"""
TP Espectrometría – Hidrógeno y Mercurio
Ajuste lineal  1/λ = a1 + R_H · (1/4 – 1/n²)  con bandas de confianza σ1 y σ2.

Sigue exactamente la misma estructura que la Entrega 3 (Paz Leighton):
  – Mínimos cuadrados analíticos (mismas fórmulas S, Sx, Sxx, Delta)
  – Matriz de covarianza 2x2
  – chi² y p-valor
  – Bandas con covarianza completa vs solo diagonal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist

# ══════════════════════════════════════════════════════════════════
#   PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════

R_H_teo = 1.0973731568e-2   # Constante de Rydberg teórica [nm⁻¹]
                              # (= 1.0973731568e7 m⁻¹ convertida a nm⁻¹)

# Términos de Balmer: x = 1/4 – 1/n²  (n = número cuántico del nivel superior)
x_n3 = 1/4 - 1/9            # Hα  rojo     ≈ 0.13889  → λ_teo ≈ 656 nm
x_n4 = 1/4 - 1/16           # Hβ  cián     = 0.18750  → λ_teo ≈ 486 nm
x_n5 = 1/4 - 1/25           # Hγ  violeta  = 0.21000  → λ_teo ≈ 434 nm

# ══════════════════════════════════════════════════════════════════
#   FUNCIONES (mismo estilo que TP3)
# ══════════════════════════════════════════════════════════════════

def linear_fit(x, y, sigma):
    """
    Mínimos cuadrados lineales: y = a1 + a2·x   con error uniforme sigma.
    Devuelve a1, a2 y la matriz de covarianza 2×2.
    Las fórmulas son las de Cowan 7.3 / las mismas del TP3.
    """
    w     = 1.0 / sigma**2
    n     = len(x)
    S     = n   * w
    Sx    = w   * np.sum(x)
    Sy    = w   * np.sum(y)
    Sxx   = w   * np.sum(x**2)
    Sxy   = w   * np.sum(x * y)
    Delta = S * Sxx - Sx**2       # det de la matriz normal 2×2
    a1    = (Sxx * Sy  - Sx * Sxy) / Delta   # intercepto (debería ≈ 0)
    a2    = (S   * Sxy - Sx * Sy ) / Delta   # pendiente = R_H estimado
    cov   = (1 / Delta) * np.array([[Sxx, -Sx],
                                     [-Sx,   S]])
    return a1, a2, cov


def variance_ya(xa, cov):
    """
    Var(ya) = cov[0,0] + xa²·cov[1,1] + 2·xa·cov[0,1]   (covarianza completa)
    Var_diag(ya) = cov[0,0] + xa²·cov[1,1]               (solo diagonal, incorrecto)
    """
    var_full = cov[0,0] + xa**2 * cov[1,1] + 2 * xa * cov[0,1]
    var_diag = cov[0,0] + xa**2 * cov[1,1]
    return np.sqrt(np.abs(var_full)), np.sqrt(var_diag)


# ══════════════════════════════════════════════════════════════════
#   DATOS – HIDRÓGENO
# ══════════════════════════════════════════════════════════════════
#  Longitudes de onda medidas [nm] del espectrómetro
#  (orden de difracción 1 izq/der y 2do orden cuando estaba disponible)

lambda_H_nm = np.array([
    648.88, 648.35,            # Rojo   orden 1-  /  1+       → n_q = 3
    485.61, 485.06, 476.31,    # Cián   orden 1-  /  1+ / 2+  → n_q = 4
    429.92, 427.36, 402.71,    # Violeta orden 1- /  2+ / 1+  → n_q = 5
])

x_H = np.array([
    x_n3, x_n3,
    x_n4, x_n4, x_n4,
    x_n5, x_n5, x_n5,
])

# 1/λ directamente en nm⁻¹  (λ ya está en nm)
y_H = 1.0 / lambda_H_nm

# Error en 1/λ estimado a partir de incertidumbre angular ~0.5°
# Δ(1/λ) ≈ d·cos(θ)·Δθ / λ²  ≈ 5e-5 nm⁻¹  (error promedio tabla: 1.85%)
sigma_H = 5e-5     # [nm⁻¹]

n_H  = len(x_H)
nu_H = n_H - 2    # grados de libertad (2 parámetros: a1, a2)

a1_H, a2_H, cov_H = linear_fit(x_H, y_H, sigma_H)
sigma_a1_H = np.sqrt(cov_H[0, 0])
sigma_a2_H = np.sqrt(cov_H[1, 1])
cov_a1a2_H = cov_H[0, 1]

chi2_H  = np.sum(((y_H - (a1_H + a2_H * x_H)) / sigma_H)**2)
p_val_H = 1 - chi2_dist.cdf(chi2_H, nu_H)

print("=" * 62)
print("  HIDRÓGENO  –  1/λ = a1 + R_H · (1/4 – 1/n²)")
print("=" * 62)
print(f"  a1 (intercepto)  = {a1_H:+.6f}  ±  {sigma_a1_H:.6f}  [nm⁻¹]  (debería ≈ 0)")
print(f"  a2 = R_H medido  = {a2_H:.6f}  ±  {sigma_a2_H:.6f}  [nm⁻¹]")
print(f"  R_H teórico      = {R_H_teo:.6f}  [nm⁻¹]")
print(f"  Diferencia       = {abs(a2_H - R_H_teo)/R_H_teo*100:.2f} %")
print(f"  Cov(a1, a2)      = {cov_a1a2_H:.2e}  (anticorrelación → banda más angosta)")
print(f"  chi² = {chi2_H:.2f}   (ν = {nu_H})")
print(f"  p-valor = {p_val_H:.3f}  {'✓ buen ajuste' if p_val_H > 0.05 else '✗ ajuste deficiente'}")

# Mínimo de Var(ya): donde d/dxa = 0  →  xa_min = –Cov/Var(a2) = x̄
xa_min_H = -cov_a1a2_H / cov_H[1, 1]
print(f"  Mínimo de Var(ya) en x = {xa_min_H:.4f}   (x̄ = {np.mean(x_H):.4f})")

# ══════════════════════════════════════════════════════════════════
#   DATOS – MERCURIO
# ══════════════════════════════════════════════════════════════════
#  Mercurio NO sigue la serie de Balmer.
#  Asignación de n cuántico: Violeta→5, Verde→5, Naranja→4
#  (misma tabla de la planilla) → veremos chi² >> ν

lambda_Hg_nm = np.array([
    569.11, 567.21,            # Naranja orden 1-, 1+     → n_q = 4
    565.53,                    # Naranja orden 2- derecha → n_q = 4
    541.81, 539.07,            # Verde   orden 1-, 1+     → n_q = 5
    555.37, 543.98,            # Verde   orden 2-, 2+     → n_q = 5
    431.03, 428.24,            # Violeta orden 1-, 1+     → n_q = 5
    426.86,                    # Violeta orden 2+         → n_q = 5
])

x_Hg = np.array([
    x_n4, x_n4,
    x_n4,
    x_n5, x_n5,
    x_n5, x_n5,
    x_n5, x_n5,
    x_n5,
])

y_Hg     = 1.0 / lambda_Hg_nm
sigma_Hg = 5e-5     # misma incertidumbre angular [nm⁻¹]

n_Hg  = len(x_Hg)
nu_Hg = n_Hg - 2

a1_Hg, a2_Hg, cov_Hg = linear_fit(x_Hg, y_Hg, sigma_Hg)
sigma_a1_Hg = np.sqrt(cov_Hg[0, 0])
sigma_a2_Hg = np.sqrt(cov_Hg[1, 1])

chi2_Hg  = np.sum(((y_Hg - (a1_Hg + a2_Hg * x_Hg)) / sigma_Hg)**2)
p_val_Hg = 1 - chi2_dist.cdf(chi2_Hg, nu_Hg)

print("\n" + "=" * 62)
print("  MERCURIO  –  Modelo Balmer aplicado (referencia)")
print("=" * 62)
print(f"  a1 (intercepto)  = {a1_Hg:+.6f}  ±  {sigma_a1_Hg:.6f}  [nm⁻¹]")
print(f"  a2 = R_eff       = {a2_Hg:.6f}  ±  {sigma_a2_Hg:.6f}  [nm⁻¹]")
print(f"  chi² = {chi2_Hg:.2f}   (ν = {nu_Hg})  ← chi² >> ν : el modelo Balmer NO aplica")
print(f"  p-valor = {p_val_Hg:.2e}  ✗  (Hg no tiene estructura hidrógeno-like)")

# ══════════════════════════════════════════════════════════════════
#   FIGURA 1 – Ajuste lineal + bandas σ1 y σ2
# ══════════════════════════════════════════════════════════════════

xa = np.linspace(0.09, 0.26, 400)

# Colores por línea espectral
C_ROJO    = "#e53935"
C_CIAN    = "#00bcd4"
C_VIOLETA = "#8e24aa"
C_VERDE   = "#43a047"
C_NARANJA = "#fb8c00"

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("TP Espectrometría  –  Ajuste lineal  $1/\\lambda = a_1 + R_H \\cdot (1/4 - 1/n^2)$\n"
             "Bandas de confianza ±1σ y ±2σ (covarianza completa)",
             fontsize=13, fontweight="bold")

# Configuración de cada panel
paneles = [
    dict(
        titulo  = "Hidrógeno – Serie de Balmer",
        x_data  = x_H,   y_data  = y_H,
        a1=a1_H, a2=a2_H, cov=cov_H,
        sigma=sigma_H,   nu=nu_H,
        chi2=chi2_H,     p=p_val_H,
        fit_color = "navy",
        grupos = [
            dict(mask=x_H == x_n3, color=C_ROJO,
                 label=r"Rojo – H$\alpha$  $(n=3)$"),
            dict(mask=x_H == x_n4, color=C_CIAN,
                 label=r"Cián – H$\beta$   $(n=4)$"),
            dict(mask=x_H == x_n5, color=C_VIOLETA,
                 label=r"Violeta – H$\gamma$ $(n=5)$"),
        ],
    ),
    dict(
        titulo  = "Mercurio – Modelo Balmer (no aplica)",
        x_data  = x_Hg,  y_data  = y_Hg,
        a1=a1_Hg, a2=a2_Hg, cov=cov_Hg,
        sigma=sigma_Hg,  nu=nu_Hg,
        chi2=chi2_Hg,    p=p_val_Hg,
        fit_color = "saddlebrown",
        grupos = [
            dict(mask=x_Hg == x_n4, color=C_NARANJA,
                 label="Naranja  $(n_{{\\rm asig}}=4)$"),
            dict(mask=(x_Hg == x_n5) & (y_Hg < 0.002), color=C_VERDE,
                 label="Verde    $(n_{{\\rm asig}}=5)$"),
            dict(mask=(x_Hg == x_n5) & (y_Hg >= 0.002), color=C_VIOLETA,
                 label="Violeta  $(n_{{\\rm asig}}=5)$"),
        ],
    ),
]

for ax, p in zip(axes, paneles):
    a1, a2, cov = p['a1'], p['a2'], p['cov']
    ya_fit = a1 + a2 * xa
    band1, band1d = variance_ya(xa, cov)     # σ full / σ diagonal
    band2, _      = variance_ya(xa, cov)     # reusar: band2 = 2·band1
    band2 = 2 * band1

    # ── Bandas de confianza ──────────────────────────────────────
    ax.fill_between(xa, ya_fit - band2, ya_fit + band2,
                    alpha=0.12, color=p['fit_color'], label="±2σ (cov completa)")
    ax.fill_between(xa, ya_fit - band1, ya_fit + band1,
                    alpha=0.30, color=p['fit_color'], label="±1σ (cov completa)")

    # ── Línea de ajuste ─────────────────────────────────────────
    ax.plot(xa, ya_fit, color=p['fit_color'], lw=2.2, zorder=4,
            label=f"Ajuste: $R_{{\\rm eff}}$ = {a2:.4f} nm$^{{-1}}$")

    # ── R_H teórico ─────────────────────────────────────────────
    ax.plot(xa, R_H_teo * xa, color="gray", lw=1.4, ls="--", zorder=3,
            label=f"$R_H$ teórico = {R_H_teo:.4f} nm$^{{-1}}$")

    # ── Datos por color espectral ────────────────────────────────
    for g in p['grupos']:
        mask = g['mask']
        if mask.any():
            ax.errorbar(p['x_data'][mask], p['y_data'][mask],
                        yerr=p['sigma'], fmt="o", color=g['color'],
                        capsize=4, elinewidth=1.5, markersize=7,
                        zorder=5, label=g['label'])

    # ── Caja de resultados ───────────────────────────────────────
    txt = (f"$a_1$ = {a1:+.2e} ± {np.sqrt(cov[0,0]):.2e} nm$^{{-1}}$\n"
           f"$R_{{\\rm eff}}$ = {a2:.4f} ± {np.sqrt(cov[1,1]):.4f} nm$^{{-1}}$\n"
           f"$\\chi^2$ = {p['chi2']:.1f}  (ν = {p['nu']})\n"
           f"p = {p['p']:.3f}")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes,
            fontsize=8.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.85))

    ax.set_xlabel(r"$f(n) = \dfrac{1}{4} - \dfrac{1}{n^2}$", fontsize=12)
    ax.set_ylabel(r"$1/\lambda \ [\mathrm{nm}^{-1}]$", fontsize=12)
    ax.set_title(p['titulo'], fontsize=12, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/espectrometria_ajuste.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Figura 1 guardada")

# ══════════════════════════════════════════════════════════════════
#   FIGURA 2 – Varianza del ajuste (como B.2 del TP3)
#   Solo para Hidrógeno donde el modelo tiene sentido físico
# ══════════════════════════════════════════════════════════════════

var_full_H = cov_H[0,0] + xa**2 * cov_H[1,1] + 2 * xa * cov_H[0,1]
var_diag_H = cov_H[0,0] + xa**2 * cov_H[1,1]

plt.figure(figsize=(9, 4.5))
plt.plot(xa, var_full_H, lw=2.2, color="navy",
         label="Var$(1/\\lambda)$  covarianza completa (correcto)")
plt.plot(xa, var_diag_H, lw=2, color="navy", ls="--",
         label="Var$(1/\\lambda)$  solo diagonal (incorrecto, siempre mayor)")
plt.axvline(xa_min_H, color="green", ls=":", lw=1.8,
            label=f"Mínimo en $f(n)$ = {xa_min_H:.4f}  ($\\bar{{x}}$ = {np.mean(x_H):.4f})")

# Marcar las tres líneas de Balmer observadas
for xv, col, lbl in [
    (x_n3, C_ROJO,    r"Rojo   $n=3$"),
    (x_n4, C_CIAN,    r"Cián   $n=4$"),
    (x_n5, C_VIOLETA, r"Violeta $n=5$"),
]:
    plt.axvline(xv, color=col, ls=":", lw=1.3, alpha=0.8, label=lbl)

plt.xlabel(r"$f(n) = \frac{1}{4} - \frac{1}{n^2}$", fontsize=12)
plt.ylabel(r"Var$(1/\lambda)$ [nm$^{-2}$]", fontsize=12)
plt.title("Varianza del ajuste – Hidrógeno\n"
          "Mínimo en $\\bar{x}$ = punto medio de los datos (misma conclusión que B.2 del TP3)",
          fontsize=11)
plt.legend(fontsize=8.5)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/varianza_hidrogeno.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Figura 2 guardada")