"""Formal exploratory analysis for the audited METR-LA speed matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_pipeline import QualityMasks, TrafficMatrix


WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def finite_positive_speeds(matrix: TrafficMatrix) -> np.ndarray:
    """Return the main EDA view: positive speeds kept, zero/invalid values as NaN."""

    return np.where(
        np.isfinite(matrix.speeds) & (matrix.speeds > 0),
        matrix.speeds,
        np.nan,
    )


def descriptive_free_flow(valid_speeds: np.ndarray, quantile: float = 0.85) -> np.ndarray:
    """Estimate an EDA-only per-sensor free-flow reference.

    This descriptive statistic is not persisted as a model feature and must not be
    reused for model training, where training-period-only estimates are required.
    """

    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    return np.nanquantile(valid_speeds, quantile, axis=0)


def congestion_state(
    valid_speeds: np.ndarray,
    free_flow: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return an EDA-only point congestion state using a relative speed threshold."""

    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    ratio = valid_speeds / free_flow[np.newaxis, :]
    return np.isfinite(ratio) & (ratio < threshold)


def _true_runs(column: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([False], column.astype(bool), [False]))
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    return boundaries[0::2], boundaries[1::2]


def _event_metrics(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_count = np.zeros(state.shape[1], dtype=int)
    average_duration = np.zeros(state.shape[1], dtype=float)
    max_duration = np.zeros(state.shape[1], dtype=int)
    for sensor_index in range(state.shape[1]):
        starts, ends = _true_runs(state[:, sensor_index])
        durations = ends - starts
        event_count[sensor_index] = len(durations)
        if len(durations):
            average_duration[sensor_index] = float(durations.mean() * 5)
            max_duration[sensor_index] = int(durations.max() * 5)
    return event_count, average_duration, max_duration


def hourly_patterns(matrix: TrafficMatrix, valid: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    hours = matrix.timestamps.hour.to_numpy()
    for hour in range(24):
        selected = hours == hour
        values = valid[selected].ravel()
        raw_values = matrix.speeds[selected].ravel()
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "hour": hour,
                "valid_count": int(len(finite)),
                "total_count": int(values.size),
                "missing_candidate_rate": float(1 - len(finite) / values.size),
                "mean_speed_valid": float(np.mean(finite)),
                "median_speed_valid": float(np.median(finite)),
                "p10_speed_valid": float(np.quantile(finite, 0.10)),
                "p90_speed_valid": float(np.quantile(finite, 0.90)),
                "mean_speed_keep_zero": float(np.mean(raw_values)),
                "median_speed_keep_zero": float(np.median(raw_values)),
            }
        )
    return pd.DataFrame(rows)


def weekday_hour_patterns(matrix: TrafficMatrix, valid: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weekdays = matrix.timestamps.dayofweek.to_numpy()
    hours = matrix.timestamps.hour.to_numpy()
    for weekday in range(7):
        for hour in range(24):
            selected = (weekdays == weekday) & (hours == hour)
            values = valid[selected].ravel()
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "day_of_week": weekday,
                    "weekday_name": WEEKDAY_NAMES[weekday],
                    "hour": hour,
                    "valid_count": int(len(finite)),
                    "total_count": int(values.size),
                    "missing_candidate_rate": float(1 - len(finite) / values.size),
                    "mean_speed_valid": float(np.mean(finite)),
                    "median_speed_valid": float(np.median(finite)),
                }
            )
    return pd.DataFrame(rows)


def day_type_hour_patterns(matrix: TrafficMatrix, valid: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weekend = np.asarray(matrix.timestamps.dayofweek >= 5, dtype=bool)
    hours = matrix.timestamps.hour.to_numpy()
    for is_weekend, label in [(False, "Weekday"), (True, "Weekend")]:
        for hour in range(24):
            selected = (weekend == is_weekend) & (hours == hour)
            values = valid[selected].ravel()
            finite = values[np.isfinite(values)]
            if values.size:
                missing_rate = float(1 - len(finite) / values.size)
                mean_speed = float(np.mean(finite)) if len(finite) else float("nan")
                median_speed = float(np.median(finite)) if len(finite) else float("nan")
            else:
                missing_rate = float("nan")
                mean_speed = float("nan")
                median_speed = float("nan")
            rows.append(
                {
                    "day_type": label,
                    "hour": hour,
                    "valid_count": int(len(finite)),
                    "total_count": int(values.size),
                    "missing_candidate_rate": missing_rate,
                    "mean_speed_valid": mean_speed,
                    "median_speed_valid": median_speed,
                }
            )
    return pd.DataFrame(rows)


def sensor_risk_table(
    matrix: TrafficMatrix,
    valid: np.ndarray,
    free_flow: np.ndarray,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    valid_count = np.isfinite(valid).sum(axis=0)
    total_count = valid.shape[0]
    table = pd.DataFrame(
        {
            "sensor_id": matrix.sensor_ids,
            "valid_count": valid_count,
            "total_count": total_count,
            "coverage_rate": valid_count / total_count,
            "free_flow_q85_eda": free_flow,
            "mean_speed_valid": np.nanmean(valid, axis=0),
            "median_speed_valid": np.nanmedian(valid, axis=0),
            "p10_speed_valid": np.nanquantile(valid, 0.10, axis=0),
            "std_speed_valid": np.nanstd(valid, axis=0),
        }
    )

    states: dict[float, np.ndarray] = {}
    for threshold in (0.50, 0.60, 0.70):
        state = congestion_state(valid, free_flow, threshold)
        states[threshold] = state
        table[f"congestion_rate_{int(threshold * 100)}"] = state.sum(axis=0) / valid_count

    event_count, average_duration, max_duration = _event_metrics(states[0.60])
    table["congestion_event_count_60"] = event_count
    table["average_event_duration_minutes_60"] = average_duration
    table["max_event_duration_minutes_60"] = max_duration

    location_columns = locations[["sensor_id", "latitude", "longitude"]].copy()
    location_columns["sensor_id"] = location_columns["sensor_id"].astype(str)
    return table.merge(location_columns, how="left", on="sensor_id", validate="one_to_one")


def daily_trend(matrix: TrafficMatrix, valid: np.ndarray) -> pd.DataFrame:
    dates = matrix.timestamps.normalize()
    rows: list[dict[str, Any]] = []
    for date in dates.unique():
        selected = dates == date
        values = valid[selected].ravel()
        raw_values = matrix.speeds[selected].ravel()
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "date": pd.Timestamp(date),
                "valid_count": int(len(finite)),
                "total_count": int(values.size),
                "missing_candidate_rate": float(1 - len(finite) / values.size),
                "mean_speed_valid": float(np.mean(finite)),
                "median_speed_valid": float(np.median(finite)),
                "p10_speed_valid": float(np.quantile(finite, 0.10)),
                "p90_speed_valid": float(np.quantile(finite, 0.90)),
                "mean_speed_keep_zero": float(np.mean(raw_values)),
            }
        )
    return pd.DataFrame(rows)


def pre_congestion_profile(
    matrix: TrafficMatrix,
    valid: np.ndarray,
    free_flow: np.ndarray,
    *,
    threshold: float = 0.60,
    minimum_event_steps: int = 3,
    history_steps: int = 12,
    future_steps: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match sustained event starts to clear windows at the same sensor/day type/time slot."""

    ratio = valid / free_flow[np.newaxis, :]
    state = np.isfinite(ratio) & (ratio < threshold)
    weekend = np.asarray(matrix.timestamps.dayofweek >= 5, dtype=bool)
    slot = matrix.timestamps.hour.to_numpy() * 12 + matrix.timestamps.minute.to_numpy() // 5
    event_windows: list[np.ndarray] = []
    control_windows: list[np.ndarray] = []
    unmatched_events = 0
    window_size = history_steps + future_steps + 1

    for sensor_index in range(state.shape[1]):
        sensor_state = state[:, sensor_index]
        sensor_ratio = ratio[:, sensor_index]
        starts, ends = _true_runs(sensor_state)
        event_starts = [
            int(start)
            for start, end in zip(starts, ends, strict=True)
            if end - start >= minimum_event_steps
            and start >= history_steps
            and start + future_steps < len(sensor_state)
            and np.isfinite(
                sensor_ratio[start - history_steps : start + future_steps + 1]
            ).all()
        ]

        finite = np.isfinite(sensor_ratio).astype(np.int16)
        clear = (~sensor_state).astype(np.int16)
        finite_window_count = np.convolve(finite, np.ones(window_size, dtype=np.int16), mode="same")
        clear_window_count = np.convolve(clear, np.ones(window_size, dtype=np.int16), mode="same")
        valid_centers = (
            (finite_window_count == window_size)
            & (clear_window_count == window_size)
        )
        valid_centers[:history_steps] = False
        valid_centers[len(sensor_state) - future_steps :] = False

        candidate_groups: dict[tuple[bool, int], np.ndarray] = {}
        for key_weekend in (False, True):
            matching_weekend = weekend == key_weekend
            for key_slot in np.unique(slot[matching_weekend]):
                candidates = np.flatnonzero(
                    valid_centers & matching_weekend & (slot == key_slot)
                )
                if len(candidates):
                    candidate_groups[(key_weekend, int(key_slot))] = candidates

        used_controls: set[int] = set()
        for event_start in event_starts:
            candidates = candidate_groups.get((bool(weekend[event_start]), int(slot[event_start])))
            if candidates is None or not len(candidates):
                unmatched_events += 1
                continue
            ordered = candidates[np.argsort(np.abs(candidates - event_start))]
            control_index = next(
                (int(value) for value in ordered if int(value) not in used_controls),
                int(ordered[0]),
            )
            used_controls.add(control_index)
            event_windows.append(
                sensor_ratio[event_start - history_steps : event_start + future_steps + 1]
            )
            control_windows.append(
                sensor_ratio[control_index - history_steps : control_index + future_steps + 1]
            )

    if not event_windows:
        raise RuntimeError("No valid sustained congestion events were available for profile analysis")

    event_array = np.vstack(event_windows)
    control_array = np.vstack(control_windows)
    relative_minutes = np.arange(-history_steps, future_steps + 1) * 5
    profile = pd.DataFrame(
        {
            "relative_minutes": relative_minutes,
            "event_median_speed_ratio": np.median(event_array, axis=0),
            "event_p25_speed_ratio": np.quantile(event_array, 0.25, axis=0),
            "event_p75_speed_ratio": np.quantile(event_array, 0.75, axis=0),
            "control_median_speed_ratio": np.median(control_array, axis=0),
            "control_p25_speed_ratio": np.quantile(control_array, 0.25, axis=0),
            "control_p75_speed_ratio": np.quantile(control_array, 0.75, axis=0),
            "matched_event_count": len(event_windows),
        }
    )

    def value_at(minutes: int, column: str) -> float:
        return float(profile.loc[profile["relative_minutes"] == minutes, column].iloc[0])

    summary = {
        "matched_event_count": int(len(event_windows)),
        "unmatched_event_count": int(unmatched_events),
        "minimum_event_duration_minutes": int(minimum_event_steps * 5),
        "event_ratio_minus_60": value_at(-60, "event_median_speed_ratio"),
        "event_ratio_minus_30": value_at(-30, "event_median_speed_ratio"),
        "event_ratio_minus_15": value_at(-15, "event_median_speed_ratio"),
        "event_ratio_at_start": value_at(0, "event_median_speed_ratio"),
        "control_ratio_minus_15": value_at(-15, "control_median_speed_ratio"),
        "control_ratio_at_event_time": value_at(0, "control_median_speed_ratio"),
    }
    return profile, summary


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    hourly: pd.DataFrame,
    weekday_hour: pd.DataFrame,
    day_type_hour: pd.DataFrame,
    risk: pd.DataFrame,
    daily: pd.DataFrame,
    profile: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(hourly["hour"], hourly["median_speed_valid"], marker="o", label="Zero as missing")
    ax.plot(hourly["hour"], hourly["median_speed_keep_zero"], marker="o", label="Keep zero")
    ax.set(title="Hourly median speed sensitivity", xlabel="Hour", ylabel="Speed (mph)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    path = figures_dir / "hourly_speed_pattern.png"
    _save_figure(fig, path)
    outputs.append(str(path))

    pivot = weekday_hour.pivot(index="day_of_week", columns="hour", values="median_speed_valid")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    ax.set(title="Median valid speed by weekday and hour", xlabel="Hour", ylabel="Day")
    ax.set_yticks(range(7), WEEKDAY_NAMES)
    ax.set_xticks(range(0, 24, 2))
    fig.colorbar(image, ax=ax, label="Speed (mph)")
    path = figures_dir / "weekday_hour_heatmap.png"
    _save_figure(fig, path)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for day_type, group in day_type_hour.groupby("day_type"):
        ax.plot(group["hour"], group["median_speed_valid"], marker="o", label=day_type)
    ax.set(title="Weekday and weekend hourly speed", xlabel="Hour", ylabel="Median speed (mph)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    path = figures_dir / "weekday_weekend_pattern.png"
    _save_figure(fig, path)
    outputs.append(str(path))

    eligible = risk.loc[risk["coverage_rate"] >= 0.75].nlargest(15, "congestion_rate_60")
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = "S" + eligible["sensor_id"].astype(str)
    ax.barh(labels, eligible["congestion_rate_60"] * 100)
    ax.invert_yaxis()
    ax.set(title="Top sensors by descriptive congestion rate", xlabel="Congestion rate (%)", ylabel="Sensor")
    path = figures_dir / "sensor_congestion_ranking.png"
    _save_figure(fig, path)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(daily["date"], daily["mean_speed_valid"], label="Mean, zero as missing")
    ax.plot(daily["date"], daily["mean_speed_keep_zero"], alpha=0.7, label="Mean keeping zero")
    ax.set(title="Daily mean speed and zero-value sensitivity", xlabel="Date", ylabel="Speed (mph)")
    ax.grid(alpha=0.25)
    ax.legend()
    path = figures_dir / "daily_speed_trend.png"
    _save_figure(fig, path)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(profile["relative_minutes"], profile["event_median_speed_ratio"], label="Event")
    ax.fill_between(
        profile["relative_minutes"],
        profile["event_p25_speed_ratio"],
        profile["event_p75_speed_ratio"],
        alpha=0.2,
    )
    ax.plot(profile["relative_minutes"], profile["control_median_speed_ratio"], label="Matched clear control")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.60, color="red", linestyle=":", linewidth=1, label="0.60 threshold")
    ax.set(
        title="Speed-ratio profile around sustained congestion starts",
        xlabel="Minutes relative to event start",
        ylabel="Speed / descriptive free-flow speed",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    path = figures_dir / "pre_congestion_profile.png"
    _save_figure(fig, path)
    outputs.append(str(path))

    return outputs


def run_eda(
    matrix: TrafficMatrix,
    quality: QualityMasks,
    locations: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Run all required Phase 3 EDA tables and figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = finite_positive_speeds(matrix)
    free_flow = descriptive_free_flow(valid)

    hourly = hourly_patterns(matrix, valid)
    weekday_hour = weekday_hour_patterns(matrix, valid)
    day_type_hour = day_type_hour_patterns(matrix, valid)
    risk = sensor_risk_table(matrix, valid, free_flow, locations)
    daily = daily_trend(matrix, valid)
    profile, event_summary = pre_congestion_profile(matrix, valid, free_flow)

    tables = {
        "hourly_speed_pattern.csv": hourly,
        "weekday_hour_speed.csv": weekday_hour,
        "day_type_hour_speed.csv": day_type_hour,
        "sensor_risk_ranking.csv": risk.sort_values("congestion_rate_60", ascending=False),
        "daily_speed_trend.csv": daily,
        "pre_congestion_profile.csv": profile,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    figure_paths = create_figures(
        hourly,
        weekday_hour,
        day_type_hour,
        risk,
        daily,
        profile,
        figures_dir,
    )

    eligible = risk.loc[risk["coverage_rate"] >= 0.75]
    top_sensor = eligible.nlargest(1, "congestion_rate_60").iloc[0]
    lowest_hour = hourly.nsmallest(1, "median_speed_valid").iloc[0]
    lowest_weekday_hour = weekday_hour.nsmallest(1, "median_speed_valid").iloc[0]
    weekday = day_type_hour.loc[day_type_hour["day_type"] == "Weekday"]
    weekend = day_type_hour.loc[day_type_hour["day_type"] == "Weekend"]
    weekday_morning = weekday.loc[weekday["hour"].between(7, 9), "median_speed_valid"].mean()
    weekend_morning = weekend.loc[weekend["hour"].between(7, 9), "median_speed_valid"].mean()

    summary = {
        "analysis_scope": "descriptive EDA only; no supervised future label or model training",
        "main_zero_rule": "zero treated as missing candidate",
        "valid_observations": int(np.isfinite(valid).sum()),
        "missing_candidate_observations": int(np.isnan(valid).sum()),
        "systemwide_zero_timestamps": int(quality.is_systemwide_zero.sum()),
        "lowest_network_median_hour": int(lowest_hour["hour"]),
        "lowest_network_hour_median_speed": float(lowest_hour["median_speed_valid"]),
        "lowest_weekday_hour_day": str(lowest_weekday_hour["weekday_name"]),
        "lowest_weekday_hour": int(lowest_weekday_hour["hour"]),
        "lowest_weekday_hour_median_speed": float(lowest_weekday_hour["median_speed_valid"]),
        "weekday_7_9_mean_of_hourly_medians": float(weekday_morning),
        "weekend_7_9_mean_of_hourly_medians": float(weekend_morning),
        "weekday_vs_weekend_7_9_speed_difference": float(weekday_morning - weekend_morning),
        "top_risk_sensor": str(top_sensor["sensor_id"]),
        "top_risk_sensor_congestion_rate_60": float(top_sensor["congestion_rate_60"]),
        "top_risk_sensor_coverage": float(top_sensor["coverage_rate"]),
        "risk_rank_spearman_50_vs_60": float(
            eligible["congestion_rate_50"].corr(eligible["congestion_rate_60"], method="spearman")
        ),
        "risk_rank_spearman_60_vs_70": float(
            eligible["congestion_rate_60"].corr(eligible["congestion_rate_70"], method="spearman")
        ),
        "zero_sensitivity": {
            "average_hourly_median_gap_mph": float(
                (hourly["median_speed_valid"] - hourly["median_speed_keep_zero"]).mean()
            ),
            "maximum_hourly_median_gap_mph": float(
                (hourly["median_speed_valid"] - hourly["median_speed_keep_zero"]).max()
            ),
            "average_daily_mean_gap_mph": float(
                (daily["mean_speed_valid"] - daily["mean_speed_keep_zero"]).mean()
            ),
            "maximum_daily_mean_gap_mph": float(
                (daily["mean_speed_valid"] - daily["mean_speed_keep_zero"]).max()
            ),
        },
        "event_profile": event_summary,
        "tables": [str(output_dir / filename) for filename in tables],
        "figures": figure_paths,
    }
    return summary
