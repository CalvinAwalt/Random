import sqlite3

# Gold Database (useful equations)
conn_gold = sqlite3.connect('gold_eqs.db')
c_gold = conn_gold.cursor()
c_gold.execute('CREATE TABLE IF NOT EXISTS eqs (id TEXT PRIMARY KEY, eq TEXT)')

# Garbage Archive (failed but potentially useful later)
conn_garbage = sqlite3.connect('garbage_eqs.db')
c_garbage = conn_garbage.execute('''CREATE TABLE IF NOT EXISTS eqs 
                                    (id TEXT, eq TEXT, tags TEXT)''')

def save_equation(eq, is_useful):
    eq_id = hash(srepr(eq))
    if is_useful:
        c_gold.execute('INSERT OR IGNORE INTO eqs VALUES (?, ?)', (eq_id, str(eq)))
    else:
        # Tag garbage equations by properties
        tags = f"depth:{str(eq).count('(')} complexity:{len(str(eq))}"
        c_garbage.execute('INSERT INTO eqs VALUES (?, ?, ?)', (eq_id, str(eq), tags))