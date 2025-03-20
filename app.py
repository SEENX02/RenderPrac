import yfinance
from flask import *
import service
import secrets

mySessionKey = secrets.token_hex(16)

#This is my Flask App
app = Flask(__name__)

@app.route("/")
def interface():
    #This is to route to the main page
    return render_template("interface.html")

@app.route("/details", methods=["POST", "GET"])
def getPriceHistory():

    # Getting the company Name, Start to End date from the User
    companyName = request.form["company"].upper()
    startDate = request.form["startDate"]
    endDate = request.form["endDate"]

    # Not having .NS attracts bugs
    if companyName.find(".NS") == -1:
        companyName += ".NS"

    # Entering the service payer
    companyData = service.getCompanyDetail(companyName, startDate, endDate)

    # Empty return from the service layer attracts bugs
    if companyData.empty:
        return render_template("interface.html")

    # Displaying the chart on the webpage
    htmlTable = companyData.to_html(classes='table table-striped')
    return render_template("displayTable.html", table=htmlTable)

@app.route("/candle", methods=["POST", "GET"])
def getCandleChart():
    companyName = request.form["company"].upper()
    startDate = request.form["startDate"]
    endDate = request.form["endDate"]
    theme = request.form["theme"]

    if companyName.find(".NS") == -1:
        companyName += ".NS"

    result = service.getCandle(companyName, startDate, endDate, theme)

    if result is None:  # If getCandle() returned None, show an error message on homepage
        return redirect(url_for("interface", error="Invalid company name. Please try again."))

    return redirect("/")  # Redirect to homepage after generating the chart

app.secret_key = mySessionKey  # Needed for session to work

@app.route("/predict", methods=["POST", "GET"])
def predictor():
    if request.method == "POST":
        session["company"] = request.form["company"]
        session["startDate"] = request.form["startDate"]
        session["endDate"] = request.form["endDate"]
        session["theme"] = request.form["theme"]

    return render_template("warning.html")

@app.route("/processPredict", methods=["POST", "GET"])
def processPredict():
    companyName = session.get("company", "").upper()
    startDate = session.get("startDate", "")
    endDate = session.get("endDate", "")
    theme = session.get("theme", "plotly_dark")

    if companyName.find(".NS") == -1:
        companyName += ".NS"

    try:
        print(f"Predicting for: {companyName}, {startDate} to {endDate}, Theme: {theme}")

        result = service.predict(companyName, startDate, endDate, theme)

        if result is None:  # If prediction fails due to invalid company
            return redirect(url_for("interface", error="Invalid company name. Please try again."))

    except Exception as e:
        print("Exception occurred:", e)
        return render_template("interface.html", error="An error occurred during prediction. Please try again.")

    return render_template("interface.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

