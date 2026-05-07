"""Shared plotting utilities and constants for paper figure notebooks.

Importing these into each notebook ensures visual consistency across all figures
and eliminates duplicated helper code.
"""

# ── Color scheme ──────────────────────────────────────────────────────────────
TCT_Z_COLOR = '#2166ac'   # Zeta-2 schedule
TCT_S_COLOR = '#1a9641'   # Stitched schedule
IDP_COLOR   = '#d6604d'   # Standard IDP / Standard DP
SDP_COLOR   = IDP_COLOR   # alias used in hit-rate sensitivity notebook

# ── Matplotlib style ──────────────────────────────────────────────────────────
PLOT_STYLE = {
    'figure.dpi':       150,
    'font.size':        13,
    'axes.labelsize':   13,
    'axes.titlesize':   14,
    'legend.fontsize':  11,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'lines.linewidth':  2.0,
    'axes.spines.top':  False,
    'axes.spines.right': False,
}


# ── Curve utilities ───────────────────────────────────────────────────────────

def half_life(frac, threshold=0.5):
    """Round at which active fraction first drops to or below `threshold`."""
    for r, f in enumerate(frac):
        if f <= threshold:
            return r
    return len(frac)


def pad(curve, n, fill=None):
    """Right-pad `curve` to length `n`, repeating the last value."""
    fill = (curve[-1] if curve else 0.0) if fill is None else fill
    return list(curve) + [fill] * (n - len(curve))


def hl_label(hl, n):
    """Format half-life label; use '≥hl' when curve never crossed threshold."""
    return f'\u2265{hl}' if hl >= n - 1 else str(hl)


# ── Single-axis panel functions (used by ablation notebooks) ──────────────────

def plot_panel(ax, zf, sf, idpf, hl_z, hl_s, hl_i, leg_loc, tag):
    """Retention panel: Zeta-2, Stitched, and Standard RDP on one axis."""
    n = max(len(zf), len(sf), len(idpf)) + 1
    rnds = list(range(n))
    za = pad(zf, n); sa = pad(sf, n); ia = pad(idpf, n)
    ax.plot(rnds, ia, color=IDP_COLOR,   lw=2.0, linestyle=':',
            label=f'Std. RDP\n(HL={hl_i})')
    ax.plot(rnds, za, color=TCT_Z_COLOR, lw=2.0,
            label=f'A-TCT Zeta-2\n(HL={hl_label(hl_z, n)})')
    ax.plot(rnds, sa, color=TCT_S_COLOR, lw=2.0, linestyle='--',
            label=f'A-TCT Stitched\n(HL={hl_label(hl_s, n)})')
    ax.fill_between(rnds, ia, za, alpha=0.10, color=TCT_Z_COLOR)
    ax.axhline(0.5, color='#aaa', lw=0.75, linestyle=':', zorder=0)
    if hl_i < n - 2:
        ax.axvline(hl_i, color=IDP_COLOR, lw=0.9, linestyle=':', alpha=0.6)
    ax.set_xlabel('Query round', fontsize=16)
    ax.set_xlim(0, n - 1); ax.set_ylim(-0.02, 1.08)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=leg_loc, fontsize=12, frameon=True, framealpha=0.88,
              borderpad=0.45, labelspacing=0.55, handlelength=2.2)
    ax.text(0.5, 1.02, tag, transform=ax.transAxes,
            fontsize=13, ha='center', va='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.88,
                      ec='#cccccc', lw=0.8))


def plot_panel_budget(ax, zb, sb, ib, leg_loc, tag):
    """Budget panel: mean remaining ε_RDP for Zeta-2, Stitched, Standard RDP."""
    n = max(len(zb), len(sb), len(ib)) + 1
    rnds = list(range(n))
    zb_ = pad(zb, n); sb_ = pad(sb, n); ib_ = pad(ib, n)
    ax.plot(rnds, ib_, color=IDP_COLOR,   lw=2.0, linestyle=':', label='Std. RDP')
    ax.plot(rnds, zb_, color=TCT_Z_COLOR, lw=2.0,               label='A-TCT Zeta-2')
    ax.plot(rnds, sb_, color=TCT_S_COLOR, lw=2.0, linestyle='--', label='A-TCT Stitched')
    ax.axhline(0, color='#888', lw=0.5, zorder=0)
    ax.set_xlabel('Query round', fontsize=16)
    ax.set_xlim(0, n - 1); ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc=leg_loc, fontsize=12, frameon=True, framealpha=0.88,
              borderpad=0.45, labelspacing=0.55, handlelength=2.2)
    ax.text(0.5, 1.02, tag, transform=ax.transAxes,
            fontsize=13, ha='center', va='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.88,
                      ec='#cccccc', lw=0.8))
