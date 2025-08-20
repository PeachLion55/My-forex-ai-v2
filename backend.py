from fastapi import FastAPI, HTTPException
import requests
import sqlite3
import json

app = FastAPI()

# SQLite for storing tokens
conn = sqlite3.connect("tokens.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS myfxbook_tokens (
    username TEXT PRIMARY KEY,
    token TEXT
)
""")
conn.commit()

MYFXBOOK_API = "https://www.myfxbook.com/api"

@app.post("/connect-myfxbook")
def connect_myfxbook(username: str, email: str, password: str):
    url = f"{MYFXBOOK_API}/login.json?email={email}&password={password}"
    r = requests.get(url)
    data = r.json()

    if not data.get("error"):
        token = data["session"]
        c.execute("REPLACE INTO myfxbook_tokens (username, token) VALUES (?, ?)", (username, token))
        conn.commit()
        return {"success": True, "token": token}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/myfxbook/accounts")
def get_accounts(username: str):
    c.execute("SELECT token FROM myfxbook_tokens WHERE username=?", (username,))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=403, detail="User not connected")
    token = row[0]

    url = f"{MYFXBOOK_API}/get-my-accounts.json?session={token}"
    r = requests.get(url)
    return r.json()

@app.get("/myfxbook/history/{account_id}")
def get_history(account_id: str, username: str):
    c.execute("SELECT token FROM myfxbook_tokens WHERE username=?", (username,))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=403, detail="User not connected")
    token = row[0]

    url = f"{MYFXBOOK_API}/get-history.json?session={token}&id={account_id}"
    r = requests.get(url)
    return r.json()
