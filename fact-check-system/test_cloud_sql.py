import psycopg2

conn = psycopg2.connect(
    host="34.80.177.243",
    port=5432,
    user="postgres",
    password="@Aa123456",
    dbname="postgres"
)

cur = conn.cursor()
cur.execute("SELECT * FROM session;")
print(cur.fetchall())
