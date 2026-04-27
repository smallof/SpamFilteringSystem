import pymysql

conn = pymysql.connect(
    host='localhost',
    user='root',
    password='root',
    cursorclass=pymysql.cursors.DictCursor,
    database='comments'
)

try:
    with conn.cursor() as cursor:
        # sql = "CREATE DATABASE IF NOT EXISTS comments"
        # cursor.execute(sql)

        table_comments = "CREATE TABLE IF NOT EXISTS comments (id INT PRIMARY KEY AUTO_INCREMENT, text TEXT NOT NULL, classification TEXT NOT NULL)"
        table_classification = "CREATE TABLE IF NOT EXISTS classification_result (result_id INT PRIMARY KEY AUTO_INCREMENT, comment_id INT NOT NULL, spam_predict FLOAT NOT NULL, ham_predict FLOAT NOT NULL, predict_label TEXT NOT NULL)"
        cursor.execute(table_comments)
        cursor.execute(table_classification)
    conn.commit()
finally:
    conn.close()

def init_table():
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INT PRIMARY KEY AUTO_INCREMENT,
                text TEXT NOT NULL,
                classification TEXT NOT NULL
            )
        """)
    conn.commit()

def save_comment(text: str, classification: str):
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='comments',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO comments (text, classification) VALUES (%s, %s)"
            cursor.execute(sql, (text, classification))
        conn.commit()

        return cursor.lastrowid
    finally:
        conn.close()

def save_classification(comment_id, spam_predict, ham_predict, predict_label):
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='comments',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO classification_result (comment_id, spam_predict, ham_predict, predict_label) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (comment_id, spam_predict, ham_predict, predict_label))
        conn.commit()
    finally:
        conn.close()