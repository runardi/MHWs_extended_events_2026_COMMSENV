# MHWs_extended_events_2026_COMMSENV
Code supporting the publication: Nardi, R. U. et al., Persistent warm water anomalies before and after marine heatwaves amplify heat exposure and associated risks. Nature Communications Earth &amp; Environment

Python module for detecting pre- and post-marine heatwaves periods, extending the standard Hobday et al. (2016) framework.

**Dependencies**

marineHeatWaves — https://github.com/ecjoliver/marineHeatWaves

numpy, pandas, scipy

**Usage**
```python
import pandas as pd
from mhw_extended_events import detect_mhw_extended

df = pd.read_csv("your_sst_data.csv")
t   = pd.to_datetime(df["time"]).map(pd.Timestamp.toordinal).to_numpy()
sst = df["sst"].to_numpy(dtype=float)

_, df_events = detect_mhw_extended(t, sst)
df_events.to_csv("results.csv", index=False)
```
**Input data format**

CSV with columns time (dates) and sst (daily sea surface temperature in °C).

**Test files**

A sample input (`test_sst_data.csv`) and expected output (`test_output.csv`) are provided to verify your setup:

python mhw_extended_events.py
