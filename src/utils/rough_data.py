import mysql.connector
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="lu_iz",
    database="crowd_detection",
)

mycursor = db.cursor()

query = "SELECT * FROM crowd_records"

df = pd.read_sql(query, db)
print(df)