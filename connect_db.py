import pandas as pd
import sqlite3

DB_PATH = r'C:\Users\bey77\OneDrive\Desktop\Projects\FinanceProj\Credit Card Approval Model\warehouse.db'

df = pd.read_csv(r'C:\Users\bey77\OneDrive\Desktop\Projects\FinanceProj\Credit Card Approval Model\Credit_card_merged_data.csv')

df.columns = [c.lower().replace(' ', '_') for c in df.columns]

conn = sqlite3.connect(DB_PATH)
df.to_sql('raw_credit_data', conn, if_exists='replace', index=False)

print(pd.read_sql("SELECT COUNT(*) FROM raw_credit_data", conn))
conn.close()

    
