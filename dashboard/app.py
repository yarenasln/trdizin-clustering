from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():

    return """
    <h1>TR Dizin Clustering</h1>

    <p>Ortak K-Means + HDBSCAN sistemi çalışıyor.</p>

    <ul>
        <li>K-Means: Yaren</li>
        <li>HDBSCAN: Nisa</li>
    </ul>
    """


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )