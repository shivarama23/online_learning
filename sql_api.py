from pyexpat import model
import sqlite3
from sqlite3 import Error
from flask import Flask, request
import time


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def insert_row(conn, sql_insert_row, model_details):
    """
    Create a new model row into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    cur = conn.cursor()
    cur.execute(sql_insert_row, model_details)
    conn.commit()
    return cur.lastrowid





def main(database):
    # database = r"testDB.db"
    sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS projects (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        model_id integer,
                                        model_name text NOT NULL,
                                        begin_time text,
                                        end_time text,
                                        model_status text
                                    ); """
    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_projects_table)
    else:
        print("Error! cannot create the database connection.")

def row_insert(database, model_id, model_name, status):
    
    sql_insert_row = ''' INSERT INTO projects(model_id,model_name,begin_time,end_time, model_status)
              VALUES(?,?,?,?, ?) '''
    model_details = (model_id,model_name,"120101", "120101", status)
    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        # create row in projects table
        insert_row(conn, sql_insert_row, model_details)
    else:
        print("Error! cannot create the database connection.")

def get_status(database):
    conn = create_connection(database)
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects ORDER BY id DESC LIMIT 1")
    result = cur.fetchone()
    print(result)
    return result

def update_training_status(database):
    conn = create_connection(database)
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects ORDER BY id DESC LIMIT 1")
    last_id = cur.fetchone()[0]
    sql_update_row = '''UPDATE projects SET model_status = 'completed' WHERE id = {};'''.format(last_id)
    print(sql_update_row)
    cur = conn.cursor()
    cur.execute(sql_update_row)
    conn.commit()
    conn.close()
    return 'Done updating status'

app = Flask(__name__)

@app.route("/training", methods=['GET', 'POST'])
def train():
    input_dict = request.json
    model_name = input_dict['model_name']
    model_id = input_dict['model_id']
    database = r"model_status.db"
    status = "training"
    row_insert(database, model_id, model_name, status)
    time.sleep(30)
    print(update_training_status(database))
    print("The model name, id is:", model_name, model_id, status)
    return 'Done'

@app.route("/prediction", methods=['GET', 'POST'])
def predict():
    database = r"model_status.db"
    result = get_status(database)
    model_status = result[-1]
    print("The model status is:", model_status)
    return 'Done'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)

# if __name__ == '__main__':
#     main_insert()