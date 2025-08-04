psid_path = "/psid_controls.dta"
nsw_path = "/nsw_dw.dta"

psid = pd.read_stata(psid_path)
nsw = pd.read_stata(nsw_path)

#  Columns check ['data_id', 'treat', 'age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75', 're78']

# Correct canonical list matching the actual columns
canonical = ['treat', 'age', 'education', 'black', 'hispanic',
             'married', 'nodegree', 're74', 're75', 're78']

# Standardize both datasets
psid['treat'] = 0
nsw['treat'] = nsw['treat']

nsw = nsw[canonical]
psid = psid[canonical]

# Concatenate
df = pd.concat([nsw, psid], ignore_index=True)

# Employment indicator
df['emp78'] = (df['re78'] > 0).astype(int)

# Optional renaming for final DataFrame
df = df.rename(columns={
    'education': 'educ',
    'hispanic': 'hispan',
    'treat': 'treatment',
    'emp78': 'employment78'
})

print(df.shape)
print(df.columns.tolist())
df
