"""P5/final RQ1 orchestration using one frozen common-support reference."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    from . import analyze_rq1, rq1_standardization as standard
    from .analyze_eda import plot_style, save_figure
except ImportError:
    import analyze_rq1
    import rq1_standardization as standard
    from analyze_eda import plot_style, save_figure

ROOT = Path(__file__).resolve().parents[1]


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_scoped_connections(path, development=True):
    """Read accepted rows; filter UTC scope before any target analysis."""
    path = Path(path)
    cache = path if path.suffix == ".pkl" else path.with_name("flight_connections.pkl")
    source = pd.read_pickle(cache)[analyze_rq1.RQ1_COLUMNS] if cache.is_file() else pd.read_csv(path, usecols=analyze_rq1.RQ1_COLUMNS, low_memory=False)
    prediction = pd.to_datetime(source.upstream_actual_arr_utc, utc=True, errors="coerce")
    end = pd.Timestamp("2026-08-01" if development else "2026-09-01", tz="UTC")
    return source.loc[prediction.ge(pd.Timestamp("2026-04-01", tz="UTC")) & prediction.lt(end)].copy()


def monthly_archives(source, frozen, directory, development=True, frozen_path=None):
    """Same reference every month; each bootstrap calendar is that month only."""
    rows, summaries = [], []
    for month in range(4, 8 if development else 9):
        start = pd.Timestamp(year=2026, month=month, day=1, tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        output = Path(directory) / start.strftime("%Y-%m")
        summaries.append(standard.run(source, output, frozen=frozen, development=development, period=(start, end)))
        if frozen_path is not None:
            shutil.copyfile(frozen_path, output / "frozen_reference.json")
        table = pd.read_csv(output / "rq1_airports.csv")
        table.insert(0, "month_utc", start.strftime("%Y-%m"))
        rows.append(table)
    return pd.concat(rows, ignore_index=True), summaries


def _plot_network(nodes, edges, destination):
    """Geographic coordinate layout without adding a map-service dependency."""
    visible = nodes.loc[nodes.latitude.notna() & nodes.longitude.notna() & nodes.valid_connection_count.gt(0)].copy()
    top = edges.sort_values(["flight_count", "origin_airport_id", "destination_airport_id"], ascending=[False, True, True]).head(100)
    largest = max(float(visible.valid_connection_count.max()) if len(visible) else 0, 1)
    values = visible.standardized_mean.dropna()
    extent = max(float(values.abs().max()) if len(values) else 0, 1)
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent)
    cmap = plt.get_cmap("RdBu_r")
    contiguous = visible.longitude.between(-125, -66) & visible.latitude.between(24, 50)
    alaska = visible.latitude.ge(50)
    hawaii = visible.longitude.between(-162, -153) & visible.latitude.between(18, 24)
    masks = [("Contiguous US", contiguous, (-126, -65, 23, 51)),
             ("Alaska / northern airports", alaska, (-190, -125, 50, 73)),
             ("Hawaii", hawaii, (-162, -153, 18, 24)),
             ("Territories / other airports", ~(contiguous | alaska | hawaii), (-180, 180, -20, 50))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    maximum_edge = max(float(top.flight_count.max()) if len(top) else 0, 1)
    for ax, (title, mask, limits) in zip(axes.flat, masks):
        panel = visible.loc[mask].copy()
        panel["plot_lon"] = panel.longitude
        if title.startswith("Alaska"):
            panel.loc[panel.longitude.gt(0), "plot_lon"] -= 360
        panel_coordinates = panel.set_index("airport_id")
        ids = set(panel.airport_id)
        for edge in top.itertuples(index=False):
            if edge.origin_airport_id not in ids or edge.destination_airport_id not in ids:
                continue
            origin, dest = panel_coordinates.loc[edge.origin_airport_id], panel_coordinates.loc[edge.destination_airport_id]
            if abs(origin.plot_lon - dest.plot_lon) > 180:
                continue  # Do not draw a spurious line across the dateline.
            ax.annotate("", xy=(dest.plot_lon, dest.latitude), xytext=(origin.plot_lon, origin.latitude),
                        arrowprops={"arrowstyle": "->", "color": "#555555", "alpha": .20,
                                    "lw": .4 + 2 * edge.flight_count / maximum_edge, "shrinkA": 2, "shrinkB": 2}, zorder=1)
        sizes = 12 + 150 * np.sqrt(panel.valid_connection_count / largest)
        colors = [cmap(norm(value)) if pd.notna(value) else "#aaaaaa" for value in panel.standardized_mean]
        ax.scatter(panel.plot_lon, panel.latitude, s=sizes, c=colors, edgecolors="white", linewidths=.4, zorder=3)
        for row in panel.nlargest(8, "valid_connection_count").itertuples(index=False):
            ax.annotate(str(row.airport_code), (row.plot_lon, row.latitude), xytext=(3, 3), textcoords="offset points", fontsize=7, zorder=4)
        xlabel = "Longitude (degrees, wrapped across dateline)" if title.startswith("Alaska") else "Longitude (degrees)"
        ax.set(title=f"{title} | n={len(panel)} airports", xlim=limits[:2], ylim=limits[2:], xlabel=xlabel, ylabel="Latitude (degrees)")
        ax.grid(alpha=.18)
    for count in np.unique(np.maximum(1, np.round(np.array([.1, .5, 1]) * largest).astype(int))):
        axes[0, 0].scatter([], [], s=12 + 150 * np.sqrt(count / largest), color="#aaaaaa", label=f"{count:,} connections")
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat), shrink=.75,
                 label="Standardized downstream minus upstream delay (min / connection), >=15")
    fig.suptitle("Airport connections and scheduled flight flow\nNode size: valid connection count; gray: unsupported. Top 100 directed routes by BTS flight count; width scales with count.\nOnly within-panel routes shown; full node/edge tables include all routes. Coordinate layout, not causal attribution.", fontsize=10)
    save_figure(fig, destination / "rq1_network_geographic.png")


def _plot_airports(airports, nodes, destination, min_valid_replicates=950):
    main = airports.loc[airports.threshold.eq("ge15") & airports.supported].copy()
    main = main.nlargest(30, "retained_n").merge(nodes[["airport_id", "airport_code"]], on="airport_id", how="left")
    main = main.sort_values("airport_id")
    fig, ax = plt.subplots(figsize=(10, max(4, len(main) * .24 + 1.5)))
    if len(main):
        positions = np.arange(len(main))
        ci = main.ci_low.notna() & main.ci_high.notna()
        ax.hlines(positions[ci], main.loc[ci, "ci_low"], main.loc[ci, "ci_high"], color="#53718a", linewidth=1.3)
        ax.scatter(main.loc[ci, "standardized_mean"], positions[ci], color="#24536c", s=24,
                   label="Point estimate with 95% interval")
        ax.scatter(main.loc[~ci, "standardized_mean"], positions[~ci], facecolors="none",
                   edgecolors="#24536c", linewidths=1.3, s=32,
                   label=f"Point only: valid bootstrap < {min_valid_replicates} or interval unavailable")
        labels = main.airport_code.fillna(main.airport_id.astype(str)) + " (n=" + main.retained_n.astype(int).astype(str) + ")"
        ax.set(yticks=positions, yticklabels=labels)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), frameon=False, fontsize=8)
    else:
        ax.text(.5, .5, "No airport meets frozen common-support criteria", ha="center", transform=ax.transAxes)
    ax.axvline(0, color="#777777", linewidth=.8)
    ax.set(xlabel="Standardized mean net delay change (min / connection)",
           title=f"Common-support estimates | intervals require >= {min_valid_replicates} valid date-block draws\nUp to 30 airports chosen by retained support, ordered by AirportID; not a causal ranking")
    save_figure(fig, destination / "rq1_standardized_airports.png")


def _plot_stability(stability, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, column, title in zip(axes, ["direction_agreement", "spearman"], ["Direction agreement with >=15", "Spearman rank correlation with >=15"]):
        for position, row in enumerate(stability.itertuples(index=False)):
            value = getattr(row, column)
            if pd.notna(value):
                ax.bar(position, value, color="#53718a", width=.6)
            else:
                ax.text(position, 0, "NA", ha="center")
        labels = [f"{row.threshold}\n{row.common_airports}/{row.frozen_airports} airports" for row in stability.itertuples(index=False)]
        ax.set(xticks=np.arange(len(stability)), xticklabels=labels, title=title, ylim=(-1.05 if column == "spearman" else 0, 1.05))
        ax.axhline(0, color="#777777", linewidth=.7)
    save_figure(fig, destination / "rq1_threshold_stability.png")


def run(config, run_dir):
    """Execute raw + standardized RQ1; final mode must reuse a frozen file."""
    run_dir = Path(run_dir)
    development = config.get("mode", "development") == "development"
    if config.get("mode", "development") not in ["development", "final"]:
        raise ValueError("RQ1 mode must be development or final")
    frozen_path = config.get("rq1_frozen_reference_path")
    if not development and not frozen_path:
        raise ValueError("Final RQ1 requires rq1_frozen_reference_path from development")
    output = run_dir / "rq1_standardized"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite RQ1 archive: {output}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8-sig")) != config:
            raise ValueError("Existing run config differs from supplied actual config")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    source = read_scoped_connections(_resolve(config.get("connection_path", "processed/flight_connections.csv.gz")), development)
    frozen = json.loads(_resolve(frozen_path).read_text(encoding="utf-8-sig")) if frozen_path else None
    protocol = standard.load_protocol(_resolve(config.get("rq1_protocol_path", "configs/rq1_protocol.json"))) if frozen is None else None
    # Preserve the established raw analysis code and distinct sample-total units.
    raw_summary = analyze_rq1.run(run_dir, development=development)
    standardized_summary = standard.run(source, output, frozen=frozen, development=development, protocol=protocol)
    reference_path = output / "frozen_reference.json"
    if frozen_path:
        shutil.copyfile(_resolve(frozen_path), reference_path)  # exact frozen evidence bytes
    frozen = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    attempts = []
    for number, attempt in enumerate(frozen["selection_attempts"], 1):
        attempts.append({"attempt": number, "airport_limit": attempt["limit"], "airports": len(attempt["airport_ids"]),
                         "passed": attempt["passed"], "common_master_cells": json.dumps(attempt["common_master_cells"]),
                         "airport_ids": json.dumps(attempt["airport_ids"]), "threshold_support": json.dumps(attempt["threshold_support"])})
    pd.DataFrame(attempts).to_csv(output / "selection_attempts.csv", index=False)
    monthly, monthly_summaries = monthly_archives(source, frozen, output / "monthly", development, reference_path)
    monthly.to_csv(output / "rq1_airports_monthly.csv", index=False)
    flights = pd.read_csv(_resolve(config.get("flights_path", "processed/flights_clean.csv.gz")),
                          usecols=["OriginAirportID", "DestAirportID", "scheduled_dep_utc"], low_memory=False)
    airports = pd.read_csv(output / "rq1_airports.csv")
    nodes, edges = standard.network_tables(flights, source, airports, development,
                    reference_path=_resolve(config.get("airport_reference_path", "references/airports.csv")))
    nodes.to_csv(output / "network_nodes.csv", index=False)
    edges.to_csv(output / "network_edges.csv", index=False)
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_style()
    _plot_network(nodes, edges, figures)
    _plot_airports(airports, nodes, figures,
                   min_valid_replicates=frozen["protocol"]["bootstrap"]["min_valid_replicates"])
    stability = pd.read_csv(output / "rq1_threshold_stability.csv")
    _plot_stability(stability, figures)
    summary = {"development": development, "raw": raw_summary, "standardized": standardized_summary,
               "monthly": monthly_summaries, "frozen_reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
               "network": {"nodes": len(nodes), "edges": len(edges), "bts_flights": int(edges.flight_count.sum()),
                           "valid_connections": int(nodes.valid_connection_count.sum()), "display_top_routes": 100,
                           "edge_window": "scheduled departure UTC", "node_window": "upstream actual arrival UTC",
                           "unsupported_color": "gray", "full_coordinate_tables": "rq1_standardized/network_nodes.csv; network_edges.csv"}}
    (run_dir / "rq1_complete_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
