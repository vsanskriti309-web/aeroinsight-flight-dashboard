from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="AeroInsight | Flight Fare Prediction", page_icon="✈️", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#f3f6fb}
[data-testid="stSidebar"] {background-color:#10243a}
[data-testid="stSidebar"] * {color:white}
.hero {padding:28px;border-radius:16px;color:white;background:linear-gradient(120deg,#10243a,#176b87);margin-bottom:20px}
.hero h1 {color:white;margin-bottom:5px}
div[data-testid="stMetric"] {background:#fff!important;color:#10243a!important;padding:15px;border-radius:12px;border:1px solid #dbe4ee}
div[data-testid="stMetric"] label,div[data-testid="stMetric"] [data-testid="stMetricLabel"],div[data-testid="stMetric"] [data-testid="stMetricValue"],div[data-testid="stMetric"] [data-testid="stMetricDelta"] {color:#10243a!important;opacity:1!important}
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    rng = np.random.default_rng(42)
    airlines = ["IndiGo", "Air India", "Vistara", "SpiceJet", "AirAsia"]
    cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"]
    rows = []
    for _ in range(250):
        airline = rng.choice(airlines)
        source, destination = rng.choice(cities, 2, replace=False)
        stops = int(rng.choice([0, 1, 2], p=[0.55, 0.38, 0.07]))
        month, day = int(rng.integers(1, 13)), int(rng.integers(1, 29))
        dep_hour = int(rng.integers(0, 24))
        duration = int(rng.integers(70, 350) + stops * 50)
        airline_cost = {"IndiGo": 0, "Air India": 1600, "Vistara": 2400, "SpiceJet": -400, "AirAsia": -600}[airline]
        price = 2500 + duration * 12 + stops * 1200 + airline_cost + (1200 if month in [5, 6, 11, 12] else 0) + rng.normal(0, 700)
        rows.append([airline, source, destination, stops, day, month, dep_hour, duration, max(1500, round(price / 10) * 10)])
    return pd.DataFrame(rows, columns=["Airline", "Source", "Destination", "Total_Stops", "Journey_Day", "Journey_Month", "Dep_Hour", "Duration_Minutes", "Price"])

def find_fare_column(frame):
    for col in frame.columns:
        if col.strip().lower() in ["price", "fare", "flight_price", "ticket_price"]:
            return col
    return None

def make_statistics(frame):
    numeric = frame.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Count": numeric.count(), "Mean": numeric.mean(), "Median": numeric.median(),
        "Mode": [numeric[c].mode().iloc[0] if not numeric[c].mode().empty else np.nan for c in numeric.columns],
        "Standard Deviation": numeric.std(), "Minimum": numeric.min(), "Maximum": numeric.max(),
        "Missing Values": numeric.isna().sum()
    }).round(2)

def show_plot(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.sidebar.title("✈ AeroInsight")
st.sidebar.caption("Flight Fare Analytics Studio")
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Upload your flight dataset", type=["csv"])
st.sidebar.info("Upload a CSV with a numeric Price or Fare column to enable fare prediction.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read this CSV: {error}")
        st.stop()
else:
    sample_path = Path(__file__).parent / "data" / "sample_flights.csv"
    df = pd.read_csv(sample_path) if sample_path.exists() else create_sample_data()

df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
df = df.dropna(how="all")
if df.empty:
    st.error("The dataset is empty. Please upload a CSV containing data.")
    st.stop()
fare_column = find_fare_column(df)

st.markdown("""<div class="hero"><h1>✈ AeroInsight</h1><p>Flight Fare Prediction & Statistical Analytics Dashboard</p></div>""", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Total Features", f"{len(df.columns):,}")
c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
if fare_column and pd.api.types.is_numeric_dtype(df[fare_column]):
    c4.metric("Average Fare", f"{df[fare_column].mean():,.2f}")
else:
    c4.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))

overview, statistics, charts, prediction, dataset = st.tabs(["🏠 Overview", "📊 Statistics", "📈 Visual Analytics", "🤖 Fare Prediction", "📁 Dataset"])

with overview:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    with right:
        st.subheader("Data Quality Report")
        quality = pd.DataFrame({"Column": df.columns, "Data Type": [str(df[c].dtype) for c in df.columns],
            "Missing Values": [int(df[c].isna().sum()) for c in df.columns], "Unique Values": [int(df[c].nunique()) for c in df.columns]})
        st.dataframe(quality, use_container_width=True, hide_index=True)
    if fare_column and pd.api.types.is_numeric_dtype(df[fare_column]):
        st.subheader("Fare at a Glance")
        fares = df[fare_column].dropna()
        if not fares.empty:
            a, b, c, d = st.columns(4)
            a.metric("Mean", f"{fares.mean():,.2f}")
            b.metric("Median", f"{fares.median():,.2f}")
            modes = fares.mode()
            c.metric("Mode", f"{modes.iloc[0]:,.2f}" if not modes.empty else "N/A")
            d.metric("Fare Range", f"{fares.min():,.0f} – {fares.max():,.0f}")

with statistics:
    st.subheader("Descriptive Statistics")
    st.write("Statistical measures for all numeric columns.")
    summary = make_statistics(df)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True)
        st.download_button("Download Statistics CSV", summary.to_csv().encode("utf-8"), "flight_statistics.csv", "text/csv")
    else:
        st.warning("No numeric columns were detected.")
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()
    if categorical:
        st.subheader("Categorical Frequency Table")
        category = st.selectbox("Select a category", categorical, key="frequency_category")
        freq = df[category].fillna("(Missing)").astype(str).value_counts().rename_axis(category).reset_index(name="Frequency")
        st.dataframe(freq, use_container_width=True, hide_index=True)

with charts:
    st.subheader("Visual Analytics")
    st.caption("Explore distributions, relationships, and flight fare patterns.")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    selected_numeric = fare_column if fare_column in numeric_columns else (numeric_columns[0] if numeric_columns else None)
    left, right = st.columns(2)
    with left:
        st.markdown("### 1. Histogram")
        if selected_numeric:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df[selected_numeric].dropna(), bins=25, edgecolor="white")
            ax.set_title(f"Distribution of {selected_numeric}"); ax.set_xlabel(selected_numeric); ax.set_ylabel("Frequency")
            show_plot(fig)
        else: st.info("A numeric column is required.")
    with right:
        st.markdown("### 2. Box Plot")
        if selected_numeric:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.boxplot(df[selected_numeric].dropna())
            ax.set_title(f"Box Plot of {selected_numeric}"); ax.set_ylabel(selected_numeric)
            show_plot(fig)
        else: st.info("A numeric column is required.")
    left, right = st.columns(2)
    with left:
        st.markdown("### 3. Average Fare by Category")
        if fare_column and fare_column in numeric_columns and categorical_columns:
            category = st.selectbox("Group fares by", categorical_columns, key="average_category")
            grouped = df.groupby(category, dropna=False)[fare_column].mean().sort_values(ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(7, 4)); grouped.plot(kind="bar", ax=ax)
            ax.set_title(f"Average Fare by {category}"); ax.set_ylabel("Average Fare"); ax.tick_params(axis="x", rotation=35)
            show_plot(fig)
        else: st.info("Requires a numeric fare column and a category.")
    with right:
        st.markdown("### 4. Correlation Heatmap")
        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(7, 4)); im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=70, ha="right")
            ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index); ax.set_title("Numeric Feature Correlations")
            fig.colorbar(im, ax=ax); show_plot(fig)
        else: st.info("Requires at least two numeric columns.")
    left, right = st.columns(2)
    with left:
        st.markdown("### 5. Category Counts")
        if categorical_columns:
            category = st.selectbox("Select category to count", categorical_columns, key="count_category")
            counts = df[category].fillna("(Missing)").astype(str).value_counts().head(12).sort_values()
            fig, ax = plt.subplots(figsize=(7, 4)); counts.plot(kind="barh", ax=ax)
            ax.set_title(f"Most Common {category} Values"); ax.set_xlabel("Number of Records"); show_plot(fig)
        else: st.info("Requires a categorical column.")
    with right:
        st.markdown("### 6. Scatter Plot")
        if len(numeric_columns) >= 2:
            x_col = st.selectbox("X-axis", numeric_columns, key="scatter_x")
            y_col = st.selectbox("Y-axis", [c for c in numeric_columns if c != x_col], key="scatter_y")
            points = df[[x_col, y_col]].dropna()
            if len(points) > 2500: points = points.sample(2500, random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(points[x_col], points[y_col], alpha=0.6)
            ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"{y_col} vs {x_col}"); show_plot(fig)
        else: st.info("Requires at least two numeric columns.")

with prediction:
    st.subheader("Flight Fare Prediction")
    st.write("Train a Random Forest model, predict fares for a route, and compare estimates across locations.")
    if not fare_column:
        st.warning("Name the numeric target column Price or Fare to enable prediction.")
    elif not pd.api.types.is_numeric_dtype(df[fare_column]):
        st.warning("Your fare column must contain numeric values.")
    elif len(df.dropna(subset=[fare_column])) < 10:
        st.warning("Please provide at least 10 rows with known fare values.")
    else:
        model_data = df.dropna(subset=[fare_column]).copy()
        y = pd.to_numeric(model_data[fare_column], errors="coerce")
        model_data = model_data.loc[y.notna()].copy()
        y = pd.to_numeric(model_data[fare_column], errors="coerce")
        features = [col for col in model_data.columns if col != fare_column and model_data[col].nunique(dropna=True) > 1]
        if not features:
            st.warning("No useful input features were found.")
        else:
            X = model_data[features].copy()
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = [col for col in features if col not in numeric_features]
            for col in numeric_features: X[col] = X[col].fillna(X[col].median())
            for col in categorical_features: X[col] = X[col].fillna("Unknown").astype(str)
            transformers = []
            if numeric_features: transformers.append(("numbers", "passthrough", numeric_features))
            if categorical_features: transformers.append(("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features))
            pipeline = Pipeline([("preprocessing", ColumnTransformer(transformers)),
                ("model", RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2, n_jobs=-1))])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if st.button("Train Prediction Model", type="primary"):
                with st.spinner("Training model..."):
                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)
                    st.session_state["fare_model"] = pipeline
                    st.session_state["fare_features"] = features
                    st.session_state["fare_target"] = fare_column
                    a, b, c = st.columns(3)
                    a.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, preds):,.2f}")
                    b.metric("Root Mean Squared Error", f"{np.sqrt(mean_squared_error(y_test, preds)):,.2f}")
                    c.metric("R² Score", f"{r2_score(y_test, preds):.3f}")
                    st.success("Model trained successfully!")

            if st.session_state.get("fare_model") is not None and st.session_state.get("fare_target") == fare_column and st.session_state.get("fare_features") == features:
                model = st.session_state["fare_model"]
                route_tab, compare_tab = st.tabs(["🛫 Predict by Route", "📍 Compare Locations"])
                cities = sorted(set(df["Source"].dropna().astype(str)) | set(df["Destination"].dropna().astype(str))) if {"Source", "Destination"}.issubset(df.columns) else []

                def build_input(source=None, destination=None, overrides=None):
                    values = {}
                    for col in features:
                        if col in numeric_features:
                            vals = pd.to_numeric(df[col], errors="coerce").dropna()
                            values[col] = float(vals.median()) if not vals.empty else 0.0
                        else:
                            opts = sorted(df[col].dropna().astype(str).unique().tolist()) or ["Unknown"]
                            values[col] = opts[0]
                    if "Source" in features and source is not None: values["Source"] = source
                    if "Destination" in features and destination is not None: values["Destination"] = destination
                    if overrides:
                        for col, val in overrides.items():
                            if col in features and val is not None: values[col] = val
                    return pd.DataFrame([{col: values[col] for col in features}], columns=features)

                def estimate(source, destination, overrides=None):
                    return float(model.predict(build_input(source, destination, overrides))[0])

                with route_tab:
                    st.markdown("### Predict fare for a specific route")
                    if not {"Source", "Destination"}.issubset(features) or len(cities) < 2:
                        st.warning("Both Source and Destination need to be usable model features.")
                    else:
                        c1, c2 = st.columns(2)
                        source = c1.selectbox("Departure city", cities, key="route_source")
                        destination = c2.selectbox("Destination city", [city for city in cities if city != source], key="route_destination")
                        overrides = {}
                        c1, c2, c3 = st.columns(3)
                        if "Airline" in features:
                            airlines = sorted(df["Airline"].dropna().astype(str).unique())
                            overrides["Airline"] = c1.selectbox("Airline", airlines, key="route_airline")
                        if "Total_Stops" in features:
                            stops = sorted(pd.to_numeric(df["Total_Stops"], errors="coerce").dropna().astype(int).unique())
                            overrides["Total_Stops"] = c2.selectbox("Number of stops", stops, key="route_stops")
                        if "Journey_Month" in features:
                            overrides["Journey_Month"] = c3.selectbox("Journey month", list(range(1, 13)), key="route_month")
                        c1, c2, c3 = st.columns(3)
                        if "Journey_Day" in features:
                            overrides["Journey_Day"] = c1.selectbox("Journey day", list(range(1, 29)), key="route_day")
                        if "Dep_Hour" in features:
                            overrides["Dep_Hour"] = c2.slider("Departure hour", 0, 23, 10, key="route_hour")
                        if "Duration_Minutes" in features:
                            vals = pd.to_numeric(df["Duration_Minutes"], errors="coerce").dropna()
                            overrides["Duration_Minutes"] = c3.number_input("Duration (minutes)", min_value=30, max_value=1500, value=int(vals.median()) if not vals.empty else 180, step=10, key="route_duration")
                        if st.button("Predict Fare for This Route", type="primary", key="predict_route"):
                            fare = estimate(source, destination, overrides)
                            st.success(f"Estimated fare for {source} → {destination}: {fare:,.0f} (dataset currency units)")
                            st.caption("This is a model estimate, not a live airline quote.")

                with compare_tab:
                    st.markdown("### Compare estimated fares by location")
                    if not {"Source", "Destination"}.issubset(features) or len(cities) < 2:
                        st.warning("Both Source and Destination need to be usable model features.")
                    else:
                        mode = st.radio("Comparison type", ["One departure city → many destinations", "Many departure cities → one destination"], horizontal=True)
                        if mode.startswith("One departure"):
                            fixed = st.selectbox("Departure city", cities, key="compare_source")
                            options = [city for city in cities if city != fixed]
                            selected = st.multiselect("Destinations to compare", options, default=options[:min(4, len(options))], key="compare_destinations")
                            if st.button("Compare Destination Fares", type="primary", key="compare_dest_button"):
                                rows = [{"Departure": fixed, "Destination": dest, "Estimated Fare": round(estimate(fixed, dest), 2)} for dest in selected]
                                if rows:
                                    result = pd.DataFrame(rows).sort_values("Estimated Fare")
                                    st.dataframe(result, use_container_width=True, hide_index=True)
                                    st.bar_chart(result.set_index("Destination")["Estimated Fare"])
                                    st.caption("Other flight details use typical dataset values. These are estimates, not live prices.")
                                else: st.info("Choose at least one destination.")
                        else:
                            fixed = st.selectbox("Destination city", cities, key="compare_destination")
                            options = [city for city in cities if city != fixed]
                            selected = st.multiselect("Departure cities to compare", options, default=options[:min(4, len(options))], key="compare_sources")
                            if st.button("Compare Departure Fares", type="primary", key="compare_source_button"):
                                rows = [{"Departure": src, "Destination": fixed, "Estimated Fare": round(estimate(src, fixed), 2)} for src in selected]
                                if rows:
                                    result = pd.DataFrame(rows).sort_values("Estimated Fare")
                                    st.dataframe(result, use_container_width=True, hide_index=True)
                                    st.bar_chart(result.set_index("Departure")["Estimated Fare"])
                                    st.caption("Other flight details use typical dataset values. These are estimates, not live prices.")
                                else: st.info("Choose at least one departure city.")
            else:
                st.info("Click **Train Prediction Model** to enable route prediction and location comparisons.")

with dataset:
    st.subheader("Complete Dataset")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download Dataset as CSV", df.to_csv(index=False).encode("utf-8"), "flight_data.csv", "text/csv")

st.sidebar.success(f"Loaded {len(df):,} records")
st.sidebar.caption("AeroInsight · Data Science Dashboard")
