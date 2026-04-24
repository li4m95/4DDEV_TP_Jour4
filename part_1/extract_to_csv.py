import os
import pandas as pd
import sqlite3

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def extract_data_from_sqlite(db_path, query):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def extract_list_of_tables(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    conn.close()
    return tables['name'].tolist()

if __name__ == "__main__":
    db_path = "Chinook_Sqlite.sqlite"
    list_of_tables = extract_list_of_tables(db_path)
    print("List of tables in the database:")
    print(list_of_tables)
    for table in list_of_tables:
        query = f"SELECT * FROM {table}"
        df = extract_data_from_sqlite(db_path, query)
        print(f"Data from {table}:")
        print(df.head())
        df.to_csv(f"DataSet_csv/{table}.csv", index=False)
