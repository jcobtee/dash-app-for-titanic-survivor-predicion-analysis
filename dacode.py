import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import cross_val_predict
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from sklearn.base import clone


graphviz_dot_path = '/opt/homebrew/bin' 
os.environ["PATH"] += os.pathsep + graphviz_dot_path
import graphviz
import base64 
from io import StringIO 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

###Daten einlesen und generell vorbereiten
df = pd.read_csv("titanic.csv", sep=',', quotechar='\"', on_bad_lines='warn', low_memory=False, parse_dates=False)

df = df.dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

###Daten ändern, damit knn Sinn macht

categorical_cols = ["Sex", "Embarked"]
numerical_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline für kategoriale Features: Imputation + OneHot-Encoding
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Gesamt-Pipeline für kNN
knn_preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


models = {
    "Logistische Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=3)": Pipeline([
        ("preprocessing", knn_preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=3))
    ])
}


def extract_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

# --- 10-fache Cross-Validation ---
results_cv = {}
cm_cv_plots = {}

for name, model in models.items():
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X, y, cv=skf)

    results_cv[name] = extract_metrics(y, y_pred_cv)

    cm_cv = confusion_matrix(y, y_pred_cv)
    fig_cv = go.Figure(data=go.Heatmap(
                       z=cm_cv,
                       x=['Nicht Überlebt', 'Überlebt'],
                       y=['Nicht Überlebt', 'Überlebt'],
                       colorscale='Blues',
                       colorbar=dict(title='Anzahl')),
                       layout=go.Layout(title=f'Confusion Matrix - {name} (10-fache CV)',
                                        xaxis_title='Vorhergesagte Klasse',
                                        yaxis_title='Wahre Klasse'))
    cm_cv_plots[name] = fig_cv

# --- Bootstrapping 0.632 ---
def bootstrap_632_metrics(model_instance, X_data, y_data, n_iterations=100):
    X_data = X_data.reset_index(drop=True)
    y_data = y_data.reset_index(drop=True)

    accs, precs, recalls, f1s = [], [], [], []

    for i in range(n_iterations):
        X_boot, y_boot = resample(X_data, y_data, replace=True, n_samples=len(X_data), random_state=i)
        original_indices = np.arange(len(X_data))
        bootstrap_indices = X_boot.index.to_numpy()
        oob_indices = np.array(list(set(original_indices) - set(bootstrap_indices)))
        if len(oob_indices) == 0:
            continue

        current_model = clone(model_instance)
        if hasattr(model_instance, 'max_iter'):
            current_model.max_iter = model_instance.max_iter
        if hasattr(model_instance, 'n_neighbors'):
            current_model.n_neighbors = model_instance.n_neighbors
        if hasattr(model_instance, 'random_state'):
            current_model.random_state = model_instance.random_state

        current_model.fit(X_boot, y_boot)

        X_oob = X_data.iloc[oob_indices]
        y_oob = y_data.iloc[oob_indices]

        y_pred_oob = current_model.predict(X_oob)
        y_pred_train = current_model.predict(X_boot)

        acc_oob = accuracy_score(y_oob, y_pred_oob)
        prec_oob = precision_score(y_oob, y_pred_oob, zero_division=0)
        rec_oob = recall_score(y_oob, y_pred_oob, zero_division=0)
        f1_oob = f1_score(y_oob, y_pred_oob, zero_division=0)

        acc_train = accuracy_score(y_boot, y_pred_train)
        prec_train = precision_score(y_boot, y_pred_train, zero_division=0)
        rec_train = recall_score(y_boot, y_pred_train, zero_division=0)
        f1_train = f1_score(y_boot, y_pred_train, zero_division=0)

        acc_632 = 0.368 * acc_train + 0.632 * acc_oob
        prec_632 = 0.368 * prec_train + 0.632 * prec_oob
        rec_632 = 0.368 * rec_train + 0.632 * rec_oob
        f1_632 = 0.368 * f1_train + 0.632 * f1_oob

        accs.append(acc_632)
        precs.append(prec_632)
        recalls.append(rec_632)
        f1s.append(f1_632)

    if not accs:
        return {
            "accuracy": (np.nan, np.nan), "precision": (np.nan, np.nan),
            "recall": (np.nan, np.nan), "f1": (np.nan, np.nan),
        }

    return {
        "accuracy": (np.mean(accs), np.std(accs)),
        "precision": (np.mean(precs), np.std(precs)),
        "recall": (np.mean(recalls), np.std(recalls)),
        "f1": (np.mean(f1s), np.std(f1s)),
    }

results_bootstrap = {}
cm_bootstrap_plots = {}

for name, model in models.items():
    results = bootstrap_632_metrics(model, X, y)
    results_bootstrap[name] = results

    model_trained_full = clone(model)
    if hasattr(model, 'max_iter'):
        model_trained_full.max_iter = model.max_iter
    if hasattr(model, 'n_neighbors'):
        model_trained_full.n_neighbors = model.n_neighbors
    if hasattr(model, 'random_state'):
        model_trained_full.random_state = model.random_state

    model_trained_full.fit(X, y)
    y_pred_full = model_trained_full.predict(X)

    cm_full = confusion_matrix(y, y_pred_full)
    fig_full = go.Figure(data=go.Heatmap(
                       z=cm_full,
                       x=['Nicht Überlebt', 'Überlebt'],
                       y=['Nicht Überlebt', 'Überlebt'],
                       colorscale='Blues',
                       colorbar=dict(title='Anzahl')),
                       layout=go.Layout(title=f'Confusion Matrix - {name} (Full Data Trained Model - Bootstrapping Context)',
                                        xaxis_title='Vorhergesagte Klasse',
                                        yaxis_title='Wahre Klasse'))
    cm_bootstrap_plots[name] = fig_full

metrics = ["accuracy", "precision", "recall", "f1"]
method_names = list(results_cv.keys())

metric_graphs = []
for i in range(0, len(metrics), 2):
    row_children = []
    for j in range(2):
        if i + j < len(metrics):
            metric = metrics[i + j]
            row_children.append(html.Div([
                html.H3(f"{metric.capitalize()}", style={"textAlign": "center"}),
                dcc.Graph(
                    figure={
                        'data': [
                            go.Scatter(
                                x=method_names,
                                y=[results_cv[model][metric] for model in method_names],
                                mode='lines+markers',
                                name='Cross-Validation'
                            ),
                            go.Scatter(
                                x=method_names,
                                y=[results_bootstrap[model][metric][0] for model in method_names],
                                mode='lines+markers',
                                name='Bootstrapping 0.632'
                            )
                        ],
                        'layout': go.Layout(
                            title=f"{metric.capitalize()} Vergleich",
                            yaxis=dict(title='Wert'),
                            xaxis=dict(title='Modell'),
                            legend=dict(x=0, y=1.2, orientation='h')
                        )
                    }
                )
            ], style={"width": "100%", "display": "inline-block", "verticalAlign": "top", "margin": "1%"}))
    metric_graphs.append(html.Div(row_children, style={"display": "inline-block", "justifyContent": "center"}))

# --- Entscheidungsbaum Visualisierung ---
dt_model_for_viz = DecisionTreeClassifier(random_state=42)
dt_model_for_viz.fit(X, y)

dot_data = StringIO()
export_graphviz(dt_model_for_viz, out_file=dot_data,
                feature_names=X.columns,
                class_names=['Nicht Überlebt', 'Überlebt'],
                filled=True, rounded=True,
                special_characters=True)

graph = graphviz.Source(dot_data.getvalue(), format="svg")
encoded_image = base64.b64encode(graph.pipe()).decode('utf-8')

# --- Dash App ---
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Modellbewertung Titanic Datensatz"),

    html.H2("Vergleich der Klassifikationsmetriken (CV vs. Bootstrapping)", style={"textAlign": "center"}),
    *metric_graphs,

    html.H2("10-fache Kreuzvalidierung Metriken"),
    html.Div([
        html.Div(f"Modell: {name} - Accuracy: {res['accuracy']:.4f}, Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}, F1: {res['f1']:.4f}")
        for name, res in results_cv.items()
    ]),
    html.H3("Confusion Matrices (10-fache Kreuzvalidierung)"),
    dcc.Dropdown(
        id='cv-model-dropdown',
        options=[{'label': name, 'value': name} for name in models.keys()],
        value=list(models.keys())[0]
    ),
    dcc.Graph(id='cv-confusion-matrix-graph'),

    html.Hr(),

    html.H2("0.632 Bootstrapping Metriken"),
    html.Div([
        html.Div(f"Modell: {name} - Accuracy: {res['accuracy'][0]:.4f} ± {res['accuracy'][1]:.4f}, "
                 f"Precision: {res['precision'][0]:.4f} ± {res['precision'][1]:.4f}, "
                 f"Recall: {res['recall'][0]:.4f} ± {res['recall'][1]:.4f}, "
                 f"F1: {res['f1'][0]:.4f} ± {res['f1'][1]:.4f}")
        for name, res in results_bootstrap.items()
    ]),
    html.H3("Confusion Matrices (Modell auf vollem Datensatz trainiert, im Bootstrapping Kontext)"),
    dcc.Dropdown(
        id='bootstrap-model-dropdown',
        options=[{'label': name, 'value': name} for name in models.keys()],
        value=list(models.keys())[0]
    ),
    dcc.Graph(id='bootstrap-confusion-matrix-graph'),

    html.Hr(),

    html.H2("Visualisierung des Decision Tree Classifiers"),
    html.Div([
        html.Img(src='data:image/svg+xml;base64,{}'.format(encoded_image), 
                 style={'width': '100%', 'height': 'auto', 'max-width': '1000px', 'display': 'block', 'margin': 'auto'}),
        html.P("Hinweis: Der Entscheidungsbaum wurde einmal auf dem gesamten Datensatz trainiert, um ihn zu visualisieren.", 
               style={'text-align': 'center', 'font-style': 'italic', 'margin-top': '10px'})
    ])
])

@app.callback(
    Output('cv-confusion-matrix-graph', 'figure'),
    Input('cv-model-dropdown', 'value')
)
def update_cv_graph(selected_model_name):
    return cm_cv_plots[selected_model_name]

@app.callback(
    Output('bootstrap-confusion-matrix-graph', 'figure'),
    Input('bootstrap-model-dropdown', 'value')
)
def update_bootstrap_graph(selected_model_name):
    return cm_bootstrap_plots[selected_model_name]

if __name__ == '__main__':
    app.run(debug=True)
