"""
Визуализация одной скважины: кривые SP и GK с наложенными отбивками
и колонкой интервалов (коллектор / неколлектор).

Панели (слева направо):
    1. Шкала глубин (TVDSS)
    2. SP + SP_norm
    3. GK + GK_norm
    4. Колонка интервалов

Сохраняем в PNG.
"""

from pathlib import Path

import matplotlib.pyplot as plt


# Цвета для интервалов
RESERVOIR_COLOR = "#4CAF50"       # зелёный
NON_RESERVOIR_COLOR = "#E0E0E0"   # светло-серый

# Порог SP_norm для коллектора — для отображения вертикальной линии
SP_RESERVOIR_THRESHOLD = 0.6


def plot_well(df, title, output_path, tvdss_range=None, layers=None,
              intervals_df=None):
    """
    Рисует разрез. Ось Y — TVDSS (абсолютные отметки).

    Параметры:
        df           : pd.DataFrame из build_dataset
                       (колонки depth_tvdss, SP, SP_norm, GK, GK_norm)
        title        : заголовок
        output_path  : куда сохранить PNG
        tvdss_range  : (top_tvdss, bottom_tvdss) или None.
                       Фильтр видимого диапазона по АО.
        layers       : list[dict] из df.attrs["layers"]
        intervals_df : pd.DataFrame из segment_all_layers
                       (колонки top_tvdss, bottom_tvdss, kind,
                       interval_index). Может быть None.
    """
    # --- Фильтрация по TVDSS ----------------------------------------
    if tvdss_range is not None:
        top_tvdss, bottom_tvdss = tvdss_range
        df = df[(df["depth_tvdss"] <= top_tvdss) &
                (df["depth_tvdss"] >= bottom_tvdss)]

    if layers is None:
        layers = df.attrs.get("layers", [])

    depth_tvdss = df["depth_tvdss"].to_numpy()

    # --- 4 панели ---------------------------------------------------
    fig, axes = plt.subplots(
        nrows=1, ncols=4,
        figsize=(12, 14),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 2, 2, 1]},
    )

    # --- Панель 0: шкала глубин ------------------------------------
    axes[0].set_ylabel("Абсолютная отметка (TVDSS), м", fontsize=11)
    axes[0].set_xticks([])
    axes[0].grid(axis="y", alpha=0.3)

    # --- Панель 1: SP ----------------------------------------------
    ax = axes[1]
    if "SP" in df.columns:
        ax.plot(df["SP"].to_numpy(), depth_tvdss,
                color="tab:blue", lw=0.8)
    ax.set_xlabel("SP", color="tab:blue")
    ax.tick_params(axis="x", labelcolor="tab:blue")
    ax.grid(alpha=0.3)
    ax.set_title("SP (ПС)")

    ax2 = ax.twiny()
    if "SP_norm" in df.columns:
        ax2.plot(df["SP_norm"].to_numpy(), depth_tvdss,
                 color="tab:red", lw=1.0, alpha=0.9)
    ax2.set_xlim(0, 1)
    # Порог коллектора — вертикальная штриховая линия
    ax2.axvline(x=SP_RESERVOIR_THRESHOLD, color="tab:red",
                lw=0.7, ls="--", alpha=0.5)
    ax2.set_xlabel("SP_norm (0=песок, 1=глина)",
                   color="tab:red", fontsize=8)
    ax2.tick_params(axis="x", labelcolor="tab:red", labelsize=8)

    # --- Панель 2: GK ----------------------------------------------
    ax = axes[2]
    if "GK" in df.columns:
        ax.plot(df["GK"].to_numpy(), depth_tvdss,
                color="tab:green", lw=0.8)
    ax.set_xlabel("GK", color="tab:green")
    ax.tick_params(axis="x", labelcolor="tab:green")
    ax.grid(alpha=0.3)
    ax.set_title("GK (ГК)")

    ax2 = ax.twiny()
    if "GK_norm" in df.columns:
        ax2.plot(df["GK_norm"].to_numpy(), depth_tvdss,
                 color="tab:red", lw=1.0, alpha=0.9)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("GK_norm (0=песок, 1=глина)",
                   color="tab:red", fontsize=8)
    ax2.tick_params(axis="x", labelcolor="tab:red", labelsize=8)

    # --- Панель 3: интервалы ---------------------------------------
    ax = axes[3]
    ax.set_xticks([])
    ax.set_title("Интервалы", fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    if intervals_df is not None and not intervals_df.empty:
        for _, iv in intervals_df.iterrows():
            top = iv["top_tvdss"]
            bottom = iv["bottom_tvdss"]

            # Пропускаем интервалы вне отображаемого диапазона
            if tvdss_range is not None:
                if top < tvdss_range[1] or bottom > tvdss_range[0]:
                    continue

            kind = iv["kind"]
            color = (RESERVOIR_COLOR if kind == "reservoir"
                     else NON_RESERVOIR_COLOR)

            ax.axhspan(top, bottom, xmin=0, xmax=1,
                       facecolor=color, edgecolor="black",
                       alpha=0.7, linewidth=0.3)

            # Подпись с номером интервала в центре
            mid = (top + bottom) / 2
            ax.text(0.5, mid, str(int(iv["interval_index"])),
                    ha="center", va="center", fontsize=7)

    # --- Кровли пластов --------------------------------------------
    for lay in layers:
        top = lay.get("top_tvdss")
        if top is None:
            continue
        if top > depth_tvdss.max() or top < depth_tvdss.min():
            continue

        # Горизонтальная линия кровли на всех панелях
        for ax in axes:
            ax.axhline(y=top, color="red", lw=0.8, alpha=0.6)

        # Подпись имени пласта справа от панели интервалов
        axes[3].text(
            axes[3].get_xlim()[1], top,
            f"  {lay['name']}",
            va="center", ha="left", fontsize=8, color="darkred",
        )

    # --- Оформление -------------------------------------------------
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.96, 0.98))

    # --- Сохранение -------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"  График сохранён: {output_path.resolve()}")