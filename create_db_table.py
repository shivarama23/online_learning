from pyexpat import model
import sqlite3
from sqlite3 import Error
from flask import Flask, request


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
    database = r"testDB.db"
    # sql = ''' INSERT INTO projects(model_id,model_name,begin_time,end_time)
    #           VALUES(1234,"text_classifier","120101", "120101") '''
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

def row_insert(model_id, model_name):
    database = r"testDB.db"
    sql_insert_row = ''' INSERT INTO projects(model_id,model_name,begin_time,end_time)
              VALUES(?,?,?,?) '''
    model_details = (model_id,model_name,"120101", "120101")
    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        # create row in projects table
        insert_row(conn, sql_insert_row, model_details)
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    database = r"model_status.db"
    main(database)