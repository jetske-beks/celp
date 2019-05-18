import data
import recommender

from tempfile import mkdtemp
from flask import Flask, render_template, redirect, request, session, flash
from flask_session import Session

app = Flask(__name__)

app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/")
def index():
    """Landing page, shows 10 recommendations."""
    # Get current user if logged in
    user = session.get("user")
    user_id = user["user_id"] if user else None

    # Get 10 recommendations
    recommendations = recommender.recommend(user=user, n=10)

    # Render
    return render_template("index.html", recommendations=recommendations, user=session.get("user"))


@app.route("/login", methods=["POST"])
def login():
    """Login route, redirects to root."""
    # Grab username
    username = request.form.get("username")

    # Try to "log in" (note to self: should implement a password check one day)
    try:
        session["user"] = data.get_user(username=username)
    except IndexError:
        flash("Could not log you in: unknown username")

    # Goto root
    return redirect("/")


@app.route("/logout", methods=["GET"])
def logout():
    """Logout route, redirects to root."""
    # Log out
    session.pop("user")

    # Goto root
    return redirect("/")


@app.route("/business/<city>/<id>")
def business(city, id):
    """Business page, shows the business, reviews and 10 recommendations."""
    # Get current user if logged in
    user = session.get("user")
    user_id = user["user_id"] if user else None

    # Get business by city and business_id
    business = data.get_business(city.lower(), id)

    # Grab reviews
    reviews = data.get_reviews(city=business["city"].lower(), business_id=business["business_id"])

    # Get 10 recommendations
    recommendations = recommender.recommend(user=user, business_id=id, city=business["city"].lower(), n=10)

    # Render
    return render_template("business.html", business=business, recommendations=recommendations, reviews=reviews, user=user)


@app.route("/static/<path:path>")
def send_static(path):
    """Route to serve anything from the static dir."""
    return send_from_directory("static", path)


if __name__ == "__main__":
    data.initialisation(state='OH', n=10)
    app.run()
