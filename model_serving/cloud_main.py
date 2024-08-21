from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/",methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


