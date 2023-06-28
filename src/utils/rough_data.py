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

# mycursor.execute("CREATE DATABASE test")
# Q1 = "CREATE TABLE Crowd (id int PRIMARY KEY AUTO_INCREMENT, crowd_id int UNSIGNED, size int UNSIGNED, )"
query = "SELECT * FROM test"

df = pd.read_sql(query, db)
print(df)