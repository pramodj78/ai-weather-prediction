import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import pickle

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(

    "weather_dataset.csv"
)

# ==========================================
# REDUCE DATASET SIZE
# ==========================================

df = df.head(500)

# ==========================================
# FEATURES
# ==========================================

X = df[[

    "Temp_C",

    "Rel Hum_%",

    "Wind Speed_km/h",

    "Visibility_km",

    "Press_kPa"
]]

# ==========================================
# TARGET
# ==========================================

y = df["Weather"]

# ==========================================
# SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)

# ==========================================
# CREATE SMALLER MODEL
# ==========================================

model = RandomForestClassifier(

    n_estimators=20,

    max_depth=5,

    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(

    X_train,

    y_train
)

# ==========================================
# SAVE MODEL
# ==========================================

pickle.dump(

    model,

    open(
        "model.pkl",
        "wb"
    )
)

print(

    "Model Trained Successfully"
)
