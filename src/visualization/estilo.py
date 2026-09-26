"""Estilo unico dos graficos e exportacao para images/.

Paleta sobria, com uma cor de destaque para o recorte de interesse (avaliacao
negativa, entrega atrasada) e cinza para o restante. Todos os graficos dos
notebooks passam por `save`, que grava o PNG com o mesmo nome em images/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

IMAGES_DIR = Path("images")

BLUE = "#1f4e79"
ORANGE = "#d9822b"
GRAY = "#8c8c8c"
GREEN = "#3b7a57"
RED = "#b23a48"
PURPLE = "#6c5b7b"
PALETTE = [BLUE, ORANGE, GREEN, RED, GRAY, PURPLE]

# nota 1 a 5: do vermelho (insatisfeito) ao verde (satisfeito)
SCORE_COLORS = {1: RED, 2: "#d4776b", 3: "#c9c9c9", 4: "#7fae8f", 5: GREEN}

DELIVERY_COLORS = {
    "no prazo": GREEN,
    "atrasada": ORANGE,
    "nao entregue": RED,
    "cancelada": GRAY,
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette=PALETTE)
    mpl.rcParams.update(
        {
            "figure.figsize": (10, 5.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig: plt.Figure, name: str, images_dir: Path = IMAGES_DIR) -> Path:
    """Grava a figura em images/<name>.png com layout ajustado."""
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path


def fmt_pct(ax: plt.Axes, axis: str = "x", decimals: int = 0) -> None:
    """Formata o eixo como percentual (valores ja em 0 a 100)."""
    formatter = mpl.ticker.FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)


def fmt_thousands(ax: plt.Axes, axis: str = "y") -> None:
    """Separador de milhar no padrao brasileiro."""
    formatter = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)
