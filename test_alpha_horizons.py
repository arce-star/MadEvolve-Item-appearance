import sys; sys.path.insert(0,'code')
import numpy as np, pandas as pd
from quant_simulator import AlphaForecaster

fc = AlphaForecaster(); fc.load('data/forecaster.pkl')

for label, path in [('val 2024', 'data/btcusdt_1m_val.parquet'), ('test 2025', 'data/btcusdt_1m_test.parquet')]:
    df = pd.read_parquet(path)
    preds = fc.model.predict(fc.compute_features(df).to_numpy())
    print(f'\n=== {label} ===')
    for i, name in enumerate(['1m','10m','100m','1000m','sum']):
        a = preds[:, i] if i < 4 else preds.sum(axis=1)
        over = (np.abs(a) > 0.00005).sum()
        print(f'  {name}: std={a.std():.2e}, >fee_threshold={over}/{len(a)} ({over/len(a)*100:.2f}%)')
