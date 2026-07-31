from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("library_book_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get data from the web form
    student_id = int(request.form["Student_ID"])
    age = float(request.form["Age"])
    semester = int(request.form["Semester"])
    cgpa = float(request.form["CGPA"])
    library_visits = int(request.form["Library_Visits"])
    study_hours = float(request.form["Study_Hours"])
    previous_borrowing = int(request.form["Previous_Borrowing"])
    ebooks_accessed = int(request.form["Ebooks_Accessed"])
    membership_years = float(request.form["Membership_Years"])

    gender = request.form["Gender"]
    department = request.form["Department"]

    # Convert Gender
    gender_male = True if gender == "Male" else False

    # Convert Department
    department_cs = department == "Computer Science"
    department_education = department == "Education"
    department_engineering = department == "Engineering"
    department_im = department == "Information Management"

    # Create DataFrame
    input_data = pd.DataFrame([[
        student_id,
        age,
        semester,
        cgpa,
        library_visits,
        study_hours,
        previous_borrowing,
        ebooks_accessed,
        membership_years,
        gender_male,
        department_cs,
        department_education,
        department_engineering,
        department_im
    ]], columns=[
        "Student_ID",
        "Age",
        "Semester",
        "CGPA",
        "Library_Visits",
        "Study_Hours",
        "Previous_Borrowing",
        "Ebooks_Accessed",
        "Membership_Years",
        "Gender_Male",
        "Department_Computer Science",
        "Department_Education",
        "Department_Engineering",
        "Department_Information Management"
    ])

    # Make prediction
    prediction = model.predict(input_data)

    prediction = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)