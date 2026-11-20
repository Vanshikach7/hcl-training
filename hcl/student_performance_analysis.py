import pandas as pd
import numpy as np


# PROJECT: STUDENT ACADEMIC PERFORMANCE ANALYSIS

# 1. Load and inspect the dataset
df = pd.read_csv("student_performance_dataset_1000.csv")

print("========== DATASET ==========")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())


# 2. Report missing values and duplicate Student_ID values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE STUDENT IDs ==========")
print(df[df["Student_ID"].duplicated(keep=False)])


# 3. Handle missing Attendance values
# Fill missing Attendance with the average attendance
attendance_mean = df["Attendance"].mean()

df["Attendance"] = df["Attendance"].fillna(attendance_mean)

print("\n========== AFTER HANDLING MISSING ATTENDANCE ==========")
print("Missing Attendance values:")
print(df["Attendance"].isnull().sum())


# 4. Check score columns for values outside 0-100
score_columns = ["Python_Marks", "Maths_Marks", "Physics_Marks"]

print("\n========== INVALID MARKS ==========")

for column in score_columns:
    invalid = df[(df[column] < 0) | (df[column] > 100)]
    
    print("\n", column)
    print(invalid[[ "Student_ID", "Name", column ]])


# 5. Calculate total and average marks for each student
df["Total_Marks"] = (
    df["Python_Marks"] +
    df["Maths_Marks"] +
    df["Physics_Marks"]
)

df["Average_Marks"] = df["Total_Marks"] / 3

print("\n========== TOTAL AND AVERAGE MARKS ==========")
print(df[[
    "Student_ID",
    "Name",
    "Total_Marks",
    "Average_Marks"
]].head(10))


# 6. Create Grade column
def calculate_grade(marks):

    if marks >= 90:
        return "A+"
    elif marks >= 80:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 60:
        return "C"
    elif marks >= 50:
        return "D"
    else:
        return "F"


df["Grade"] = df["Average_Marks"].apply(calculate_grade)

print("\n========== GRADES ==========")
print(df[[
    "Student_ID",
    "Name",
    "Average_Marks",
    "Grade"
]].head(10))


# 7. Calculate department-wise average marks
department_average = df.groupby("Department")[
    ["Python_Marks", "Maths_Marks", "Physics_Marks"]
].mean()

print("\n========== DEPARTMENT-WISE AVERAGE MARKS ==========")
print(department_average)


# 8. Find Top 10 students by average marks
top_10 = df.sort_values(
    by="Average_Marks",
    ascending=False
).head(10)

print("\n========== TOP 10 STUDENTS ==========")
print(top_10[[
    "Student_ID",
    "Name",
    "Department",
    "Average_Marks",
    "Grade"
]])


# 9. Identify students below 40% average marks
below_40 = df[df["Average_Marks"] < 40]

print("\n========== STUDENTS BELOW 40% ==========")

if len(below_40) == 0:
    print("No student has an average mark below 40%.")
else:
    print(below_40[[
        "Student_ID",
        "Name",
        "Department",
        "Average_Marks"
    ]])


# 10. Compare average marks for different attendance bands
df["Attendance_Band"] = pd.cut(
    df["Attendance"],
    bins=[0, 60, 75, 90, 100],
    labels=[
        "Below 60",
        "60-75",
        "76-90",
        "Above 90"
    ]
)

attendance_comparison = df.groupby(
    "Attendance_Band",
    observed=True
)["Average_Marks"].mean()

print("\n========== ATTENDANCE BAND VS AVERAGE MARKS ==========")
print(attendance_comparison)


# 11. Analyze relationship between study hours and average marks
study_hours_relation = df[[
    "Study_Hours",
    "Average_Marks"
]].corr()

print("\n========== STUDY HOURS AND AVERAGE MARKS ==========")
print(study_hours_relation)

correlation = df["Study_Hours"].corr(df["Average_Marks"])

print("\nCorrelation between Study Hours and Average Marks:")
print(round(correlation, 2))


# 12. Compare results for students with and without internet access
internet_comparison = df.groupby(
    "Internet_Access"
)["Average_Marks"].mean()

print("\n========== INTERNET ACCESS VS AVERAGE MARKS ==========")
print(internet_comparison)


# 13. Calculate mean, standard deviation and quartiles of Python marks
python_mean = df["Python_Marks"].mean()
python_std = df["Python_Marks"].std()
python_quartiles = df["Python_Marks"].quantile(
    [0.25, 0.50, 0.75]
)

print("\n========== PYTHON MARKS STATISTICS ==========")

print("Mean:", round(python_mean, 2))
print("Standard Deviation:", round(python_std, 2))

print("\nQuartiles:")
print(python_quartiles)


# 14. Identify potential score outliers using IQR
Q1 = df["Python_Marks"].quantile(0.25)
Q3 = df["Python_Marks"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["Python_Marks"] < lower_limit) |
    (df["Python_Marks"] > upper_limit)
]

print("\n========== PYTHON MARKS OUTLIERS ==========")

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

print("\nLower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

print("\nPotential Outliers:")
print(outliers[[
    "Student_ID",
    "Name",
    "Python_Marks"
]])


# 15. Write recommendations for academic support
print("\n========== ACADEMIC SUPPORT RECOMMENDATIONS ==========")

print("1. Students with low average marks should receive additional academic support.")

print("2. Students with low attendance should be encouraged to attend classes regularly.")

print("3. Students with low study hours should be encouraged to increase their study time.")

print("4. Students with low subject marks should receive extra practice and guidance.")

print("5. Students without internet access may be provided with suitable learning resources.")


# Final dataset preview
print("\n========== FINAL DATASET ==========")
print(df.head(10))
# 16. File Handling

with open("student_analysis.txt", "w") as file:
    file.write("STUDENT ACADEMIC PERFORMANCE ANALYSIS\n")
    file.write(df.head(10).to_string())

with open("student_analysis.txt", "r") as file:
    print("\n========== FILE HANDLING ==========")
    print(file.read())

    # Final dataset preview
print("\n========== FINAL DATASET ==========")
print(df.head(10))
