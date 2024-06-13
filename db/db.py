from flask import Flask
import requests as request

app = Flask(__name__)

@app.route("/")
def callback():
    request.get("http://localhost:7000")
    return "calling database"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500)
