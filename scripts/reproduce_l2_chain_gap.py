"""Demonstrate incompatible graph boundaries in realized returns and TD bootstrapping.

This does not prescribe treating missing observations as a real terminal state.
"""
import numpy as np
import pandas as pd

from datamatch.l2_value import realized_return, td_delta


def reproduce():
    nodes = pd.DataFrame({"node": [0, 1], "team_id": ["H", "H"], "home_team_id": ["H", "H"],
                          "segment": [1, 1], "next_node": [-1, -1], "sigma_next": [0, 0],
                          "clock": [10., 20.], "chain_gap_before": [False, True],
                          "terminal": [False, True], "valid_for_transition": [False, True]})
    reward = np.array([0., .2])
    observed_return = realized_return(nodes, reward)
    residual = td_delta(nodes, reward, observed_return)
    print("Observed return:", observed_return.tolist())
    print("TD residual with that return:", residual.tolist())
    if not (np.isnan(observed_return[0]) and np.isnan(residual[0]) and
            np.isclose(observed_return[1], .2) and np.isclose(residual[1], 0.)):
        print("FAILED: censored target and TD must stay unknown; observed suffix remains valid.")
        return 1
    print("PASS: unknown continuation is censored consistently, without invented absorption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(reproduce())
