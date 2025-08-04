df = pd.read_csv('RA-data.csv')

# rename race columns
df = df.rename(columns={
    'Race_Asian, Hawaii/Pac. Islander': 'Asian',
    'Race_Black or African American':   'Black',
    'Race_Hispanic':                    'Hispanic',
    'Race_More than one race':          'MR',
    'Race_White':                       'White'
})
racial_cols = ['Asian','Black','Hispanic','MR','White'][:base_arms]

# keep only rows with at least one race indicator
df = df[df[racial_cols].sum(axis=1) > 0]

# binary resource column
df['resource_binary'] = (df['resource'] > 0).astype(int)

# drop unwanted columns
df = df.drop(columns=[
    'Unnamed: 0',
    'gender_Male',
    'Race_Amer. Indian/Alaska Native',
    'F3_Highest level of education',
    'resource'
]).dropna()

print(f"Cleaned dataset shape: {df.shape}")