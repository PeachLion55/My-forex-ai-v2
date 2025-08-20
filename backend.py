from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import requests
import datetime
import schedule
import time
import threading

app = FastAPI()

# DB Setup
ENGINE = create_engine('sqlite:///myfxbook.db')
Base = declarative_base()

class Token(Base):
    __tablename__ = "tokens"
    user_id = Column(Integer, primary_key=True)
    token = Column(String)
    expiration = Column(String)

Base.metadata.create_all(ENGINE)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)

class LoginCreds(BaseModel):
    email: str
    password: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/connect-myfxbook")
def connect(creds: LoginCreds, db: Session = Depends(get_db)):
    resp = requests.get(f"https://www.myfxbook.com/api/login.json?email={creds.email}&password={creds.password}")
    data = resp.json()
    if data.get("error"):
        raise HTTPException(400, data["message"])
    token = data["session"]
    # Store token, assume user_id from auth
    user_id = 1  # Replace with real user
    db_token = Token(user_id=user_id, token=token, expiration=(datetime.datetime.now() + datetime.timedelta(hours=24)).isoformat())
    db.add(db_token)
    db.commit()
    return {"session": token}

@app.get("/myfxbook/accounts")
def get_accounts(db: Session = Depends(get_db)):
    # Get token for user
    token = db.query(Token).filter(Token.user_id == 1).first().token  # Replace user_id
    resp = requests.get(f"https://www.myfxbook.com/api/get-my-accounts.json?session={token}")
    return resp.json()

@app.get("/myfxbook/history/{account_id}")
def get_history(account_id: int, db: Session = Depends(get_db)):
    token = db.query(Token).filter(Token.user_id == 1).first().token
    resp = requests.get(f"https://www.myfxbook.com/api/get-history.json?session={token}&id={account_id}")
    data = resp.json()
    # Cache in DB
    # ...
    return data

# Refresh schedule
def refresh_data():
    # Query tokens, fetch data for each, cache
    pass

schedule.every(1).hour.do(refresh_data)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

threading.Thread(target=run_scheduler).start()
