import sqlite3 as sq

db = sq.connect("company.db")
cur = db.cursor()

async def db_start():
    cur.execute("CREATE TABLE IF NOT EXISTS org_data("
                "id INTEGER PRIMARY KEY AUTOINCREMENT , "
                "Organization_name TEXT, "
                "Address TEXT, "
                "Worktime TEXT, "
                "Telephone TEXT, "
                "Code TEXT)")
    db.commit()



