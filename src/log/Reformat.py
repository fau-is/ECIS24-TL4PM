import pandas as pd
import numpy as np


def roll_sequence(data, time_column="EndTime", case_column="CaseID"):
    trace = None
    for column in data.columns:
        if column != case_column:
            data_col = data.groupby(case_column)[column].apply(np.array)
            if trace is None:
                trace = data_col
            else:
                trace = pd.merge(trace, data_col, on=case_column, how='inner')
    trace["Start Time"] = trace[time_column].apply(lambda x: x[0])
    trace = trace.sort_values("Start Time", ascending=True)

    return trace
