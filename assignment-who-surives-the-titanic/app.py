from flask import Flask, render_template, request, send_file

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
import csv


app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():

    return render_template("new.html")


@app.route("/new", methods=["GET"])
def new():

    return render_template("new.html")


@app.route("/result", methods=["GET"])
def result():

    return render_template("result.html")


######Model FIT Function####
def alg_scr(algo, X_train, y_train, cv):

    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)

    # Cross Validation
    train_pred = model_selection.cross_val_predict(
        algo, X_train, y_train, cv=cv, n_jobs=-1
    )
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    return train_pred, acc, acc_cv


###############################################################


@app.route("/", methods=["POST"])
def predict():

    testfile = request.files["userfile"]
    if testfile.filename != "":
        filepath = "./userfiles/" + testfile.filename
        testfile.save(filepath)

    # Importing training/test set into panda
    train = pd.read_csv("./data/titanic-training-dataset.csv")
    train = train[
        [
            "PassengerId",
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ]
    ]

    userf = pd.read_csv(filepath)
    test = userf[
        ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",]
    ]

    # Fillin the missing vals for training set
    train["Fare"] = train["Fare"].fillna(train["Fare"].dropna().median())
    train["Age"] = train["Age"].fillna(train["Age"].dropna().median())
    train["Embarked"] = train["Embarked"].fillna("S")  # 'S' is de MOD.

    # Fillin the missing vals for test data.
    test["Fare"] = test["Fare"].fillna(test["Fare"].dropna().median())
    test["Age"] = test["Age"].fillna(test["Age"].dropna().median())
    test["Embarked"] = test["Embarked"].fillna("S")  # 'S' is de MOD.

    # Encode columns train and test
    train_enc = train.apply(LabelEncoder().fit_transform)
    test_enc = test.apply(LabelEncoder().fit_transform)

    ## Modeling
    x_train = train_enc
    x_train = x_train.drop("Survived", axis=1)
    y_train = train_enc.Survived

    # pred, accuracy_gbc, accuracy_crvl_gbc = alg_scr(
    #     GradientBoostingClassifier(), x_train, y_train, 10
    # )

    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    acc = model.score(x_train, y_train)
    train_pred = model_selection.cross_val_predict(
        GradientBoostingClassifier(), x_train, y_train, cv=10, n_jobs=-1
    )
    acc_cv = metrics.accuracy_score(y_train, train_pred)

    # Apllying the model on test and creating dataframe with Surv, PassID
    prediction = model.predict(test_enc)
    sub = pd.DataFrame()
    sub["PassengerId"] = test["PassengerId"]
    sub["Pclass"] = test["Pclass"]
    sub["Sex"] = test["Sex"]
    sub["Age"] = test["Age"]
    sub["SibSp"] = test["SibSp"]
    sub["Parch"] = test["Parch"]
    sub["Fare"] = test["Fare"]
    sub["Embarked"] = test["Embarked"]
    sub["Survived"] = prediction

    df = sub

    sub.to_csv("./userfiles/result.csv", index=False)

    return render_template(
        "result.html",
        column_names=df.columns.values,
        row_data=list(df.values.tolist()),
        link_column="PassengerId",
        zip=zip,
    )

    ## Score table of training predictions
    # score_table = pd.DataFrame(
    #     {"CV Accuracy %": acc_cv, "Accuracy %": acc, "Model": "GBC",}, index=[0]
    # )

    # table.sort_values(by = 'CV Accuracy %', ascending=False)
    # return render_template("index.html", tabless=[accuracy_gbc, accuracy_crvl_gbc])
    # return render_template(
    #     "new.html", tables=[sub.to_html(classes="data")], titles=sub.columns.values,
    # )

    # return render_template("new.html", table=sub)


@app.route("/download")
def download_file():

    p = "./userfiles/result.csv"
    return send_file(p, as_attachment=True)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
