import snowflake.connector
import pandas as pd

# connect to Snowflake
conn = snowflake.connector.connect(
    user='YOUR_USER',
    password='YOUR_PASSWORD',
    account='YOUR_ACCOUNT',
    warehouse='YOUR_WAREHOUSE',
    database='YOUR_DATABASE',
    schema='YOUR_SCHEMA'
)

# run query
query = "select * from your_table limit 10"
df = pd.read_sql(query, conn)

print(df.head())

conn.close()
