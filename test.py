import requests
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100

df_test = pd.DataFrame({
    'A': np.random.uniform(0, 10, n).round(4),
    'B': np.random.randint(10, 40, n)
})

results = []
for _, row in df_test.iterrows():

    res_predict = requests.post(
        'http://127.0.0.1:8000/predict',
        json={'A': row['A'], 'B': int(row['B'])}
    )
    y_pred = res_predict.json()['Y_pred']

    res_classe = requests.post(
        'http://127.0.0.1:8000/classe',
        json={'A': row['A'], 'B': int(row['B'])}
    )
    classe = res_classe.json()['classe']

    res_classe_aby = requests.post(
        'http://127.0.0.1:8000/classe_aby',
        json={'A': row['A'], 'B': int(row['B']), 'Y': y_pred}
    )
    classe_aby = res_classe_aby.json()['classe']

    results.append({
        'A':          row['A'],
        'B':          row['B'],
        'Y_pred':     y_pred,
        'classe_ab':  classe,
        'classe_aby': classe_aby
    })

df_results = pd.DataFrame(results)
df_results.to_csv('resultats.csv', index=False)

df_results.head()