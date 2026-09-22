"""Battery sizing from a directly-specified power rating and capacity.

The paper uses a 300 kW / 1,200 kWh battery (4 hours of storage). Here the
power rating and capacity are user-configurable, defaulting to those values.
"""
from __future__ import annotations

from dataclasses import dataclass

PAPER_P_RATE_KW = 300.0
PAPER_CAPACITY_KWH = 1200.0


@dataclass
class Battery:
    p_rate_kw: float          # delta: charge/discharge power per step (kW)
    soc_max: float            # kW-equivalent capacity (= steps_full * delta)
    soc_min: float = 0.0
    steps_full: float = 4.0
    capacity_kwh: float = PAPER_CAPACITY_KWH
    battery_hours: float = 4.0
    battery_ratio: float = 0.0   # P_rate / mean load, for reference only


def size_battery(p_rate_kw: float, capacity_kwh: float, mean_load_kw: float, dt_hours: float,
                  units: str = "energy") -> Battery:
    """Size the battery directly from its power rating and capacity.

    ``battery_hours = capacity_kwh / p_rate_kw`` (4 h for the paper's
    defaults). ``units='energy'`` (default) uses the physically consistent
    formula: delta in kWh = P_rate * dt, so ``steps_full = battery_hours / dt``
    (16 discharges when full at the default sizing on 15-minute data).
    ``units='paper'`` instead reproduces the paper's Figure 5b, where SOC
    changes by exactly delta each step regardless of dt (4 discharges when
    full at the default sizing).
    """
    battery_hours = capacity_kwh / p_rate_kw
    steps_full = battery_hours / (dt_hours if units == "energy" else 1.0)
    soc_max = steps_full * p_rate_kw
    ratio = p_rate_kw / mean_load_kw if mean_load_kw else 0.0
    return Battery(p_rate_kw=p_rate_kw, soc_max=soc_max, soc_min=0.0, steps_full=steps_full,
                    capacity_kwh=capacity_kwh, battery_hours=battery_hours, battery_ratio=ratio)
