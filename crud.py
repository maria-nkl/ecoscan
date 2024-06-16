from database import cur


def receving_inf(label_number) -> list:
    data = []
    for code in label_number:
        data.append(cur.execute(f"SELECT Code, Organization_name, Address, Worktime, Telephone FROM org_data WHERE Code = {code}").fetchone())
    return data
