import papermill as pm

N = 4
for i in range(N):
  print('executing papermill...', i)
  pm.execute_notebook(
     'otto_lgb_test.ipynb',
     f'output_{i}.ipynb',
     parameters=dict(OUTPUT_SPLIT_NUM = i)
  )

