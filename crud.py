from database import cur


def receving_inf(label_number) -> list:
    #return cur.execute(f"SELECT Organization_name, Address, Worktime, Telephone FROM org_data WHERE Code = {code}").fetchall()
    data = []
    for code in label_number:
        data.append(cur.execute(f"SELECT Code, Organization_name, Address, Worktime, Telephone FROM org_data WHERE Code = {code}").fetchone())
    return data
