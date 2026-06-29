# Ablation Matrix Results

| Run                      |   Features |     MAE |   Shift Accuracy |
|:-------------------------|-----------:|--------:|-----------------:|
| A0: Baseline (Last Week) |          0 | 7.70126 |         0.610178 |
| A1: Temporal             |         34 | 6.19084 |         0.704295 |
| A2: +Weather             |         39 | 6.20748 |         0.720739 |
| A3: +Events              |         56 | 5.81218 |         0.751382 |
| A4: +Footfall            |         59 | 6.0038  |         0.739562 |
| A5: +Airport/Cruise      |         65 | 6.02949 |         0.739617 |
| A6: +Failte/Proxies      |         70 | 5.95567 |         0.746594 |