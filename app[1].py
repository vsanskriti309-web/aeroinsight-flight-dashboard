import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AeroInsight | Flight Analytics", page_icon="✈️", layout="wide")
NAVY, BLUE, TEAL, PALE, MUTED = "#102A43", "#246BCE", "#0F9D91", "#F2F6FC", "#627D98"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{font-family:Inter,sans-serif}}
.stApp {{background:{PALE};color:{NAVY}}}
[data-testid="stHeader"] {{background:rgba(242,246,252,.92)}}
.block-container {{padding-top:1.5rem;max-width:1500px}}
[data-testid="stSidebar"] {{background:linear-gradient(180deg,#102A43,#0B1F33)}}
[data-testid="stSidebar"] * {{color:#F4F8FC}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{background:#173B5D;border:1px dashed #76A9E8}}
.hero {{background:linear-gradient(110deg,#102A43,#174C78 60%,#246BCE);color:white;padding:1.8rem 2rem;border-radius:18px;margin-bottom:1.2rem}}
.hero h1 {{color:white;margin:0;font-size:2.3rem;font-weight:800}}
.hero p {{color:#D9EAFB;margin:.5rem 0 0}}
.eyebrow {{text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;color:#A9D7FF;font-weight:700}}
.metric-card {{background:white;border:1px solid #D9E4F2;border-radius:14px;padding:1rem 1.15rem;min-height:108px;box-shadow:0 3px 12px #102a430a}}
.metric-label {{color:{MUTED};font-size:.8rem;font-weight:600}}
.metric-value {{color:{NAVY};font-size:1.8rem;font-weight:800;margin-top:.3rem}}
.metric-note {{color:{TEAL};font-size:.76rem;margin-top:.15rem}}
div[data-testid="stPlotlyChart"], div[data-testid="stImage"] {{background:white;border:1px solid #D9E4F2;border-radius:14px;padding:.4rem}}
.stButton button,.stDownloadButton button {{border-radius:9px;font-weight:700}}
</style>
""", unsafe_allow_html=True)

def find_col(df, names):
    cols={str(c).lower().strip():c for c in df.columns}
    for n in names:
        if n.lower() in cols: return cols[n.lower()]
    for c in df.columns:
        norm=str(c).lower().replace("_"," ").strip()
        if any(n.lower() in norm for n in names): return c
    return None

def metric(label,value,note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

@st.cache_data
def sample_data():
    path=os.path.join(os.path.dirname(__file__),"data","sample_flights.csv")
    if os.path.exists(path): return pd.read_csv(path)
    rng=np.random.default_rng(12); n=250
    airline=rng.choice(["IndiGo","Air India","Vistara","SpiceJet","AirAsia"],n)
    duration=rng.integers(60,600,n); stops=rng.choice([0,1,2],n)
    fare=np.maximum(1800,3000+duration*7+stops*1400+rng.normal(0,1200,n))
    return pd.DataFrame({"Airline":airline,"Source":rng.choice(["Delhi","Mumbai","Chennai","Bengaluru"],n),"Destination":rng.choice(["Goa","Delhi","Kolkata","Mumbai"],n),"Duration_Minutes":duration,"Total_Stops":stops,"Price":fare.round(0)})

st.sidebar.markdown("# ✈ AeroInsight")
st.sidebar.caption("FLIGHT FARE PREDICTION")
st.sidebar.markdown("---")
page=st.sidebar.radio("NAVIGATION",["Overview","Explore Data","Fare Prediction","About"])
uploaded=st.sidebar.file_uploader("Upload flight dataset (CSV)",type=["csv"])
if uploaded:
    try: df=pd.read_csv(uploaded); st.sidebar.success(f"Loaded {len(df):,} rows")
    except Exception as e: st.sidebar.error(f"Unable to read CSV: {e}"); df=sample_data()
else:
    df=sample_data(); st.sidebar.info("Using sample dataset")
df.columns=[str(c).strip() for c in df.columns]
fare_col=find_col(df,["price","fare","ticket price","flight price"])
if fare_col is None:
    st.error("Please upload a CSV with a numeric column named Price or Fare.")
    st.write("Columns found:",list(df.columns)); st.stop()
df[fare_col]=pd.to_numeric(df[fare_col].astype(str).str.replace(r"[₹$,]","",regex=True),errors="coerce")
df=df.dropna(subset=[fare_col]).copy()
if df.empty: st.error("No numeric fare values found in that CSV."); st.stop()
airline=find_col(df,["airline","carrier"])
source=find_col(df,["source","origin","from"])
dest=find_col(df,["destination","dest","to"])
duration=find_col(df,["duration minutes","duration_min","duration","journey time"])
stops=find_col(df,["total stops","stops","stop"])
st.markdown('<div class="hero"><div class="eyebrow">FLIGHT INTELLIGENCE • DATA-DRIVEN DECISIONS</div><h1>✈ AeroInsight</h1><p>Understand flight fares, discover patterns, and estimate prices with machine learning.</p></div>',unsafe_allow_html=True)
st.caption(f"Dataset: {len(df):,} records · Fare column: {fare_col} · INR assumed for display")
if page=="Overview":
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("TOTAL RECORDS",f"{len(df):,}","Rows in current dataset")
    with c2: metric("TOTAL FEATURES",str(len(df.columns)),"Columns available")
    with c3: metric("MISSING CELLS",f"{df.isna().sum().sum():,}","Across all columns")
    with c4: metric("AVERAGE FARE",f"₹{df[fare_col].mean():,.0f}",f"Median ₹{df[fare_col].median():,.0f}")
    st.markdown("### Fare & route analytics")
    st.caption("Six views of your dataset. Charts adapt to the columns available in your CSV.")
    a,b,c=st.columns(3)
    with a:
        st.markdown("**1 · Fare distribution**")
        fig,ax=plt.subplots(figsize=(5,3)); ax.hist(df[fare_col],bins=20,color=BLUE,edgecolor="white"); ax.set_xlabel("Fare"); ax.set_ylabel("Flights"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with b:
        st.markdown("**2 · Average fare by airline**")
        if airline:
            x=df.groupby(airline)[fare_col].mean().sort_values()
            fig,ax=plt.subplots(figsize=(5,3)); x.plot.barh(ax=ax,color=TEAL); ax.set_xlabel("Average fare"); ax.set_ylabel(""); ax.grid(axis="x",alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else: st.info("Add an Airline column to show this chart.")
    with c:
        st.markdown("**3 · Average fare by source/destination**")
        group=source or dest
        if group:
            x=df.groupby(group)[fare_col].mean().sort_values(ascending=False).head(10)
            fig,ax=plt.subplots(figsize=(5,3)); x.plot.bar(ax=ax,color="#F59E0B"); ax.set_ylabel("Average fare"); ax.set_xlabel(""); ax.tick_params(axis="x",rotation=35); ax.grid(axis="y",alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else: st.info("Add Source or Destination to show this chart.")
    d,e,f=st.columns(3)
    with d:
        st.markdown("**4 · Duration vs fare**")
        if duration:
            dur=pd.to_numeric(df[duration].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],errors="coerce")
            fig,ax=plt.subplots(figsize=(5,3)); ax.scatter(dur,df[fare_col],alpha=.55,s=18,color=BLUE); ax.set_xlabel(duration); ax.set_ylabel("Fare"); ax.grid(alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else: st.info("Add a Duration column to show this chart.")
    with e:
        st.markdown("**5 · Flights by number of stops**")
        if stops:
            x=df[stops].fillna("Unknown").astype(str).value_counts()
            fig,ax=plt.subplots(figsize=(5,3)); ax.pie(x.values,labels=x.index,autopct="%1.0f%%",startangle=90,colors=[BLUE,TEAL,"#F59E0B","#8B5CF6","#94A3B8"]); ax.axis("equal"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else: st.info("Add Total_Stops or Stops to show this chart.")
    with f:
        st.markdown("**6 · Fare spread by airline**")
        if airline:
            top=df[airline].value_counts().head(8).index
            vals=[df.loc[df[airline]==v,fare_col].dropna().values for v in top]
            fig,ax=plt.subplots(figsize=(5,3)); ax.boxplot(vals,tick_labels=[str(v) for v in top]); ax.set_ylabel("Fare"); ax.tick_params(axis="x",rotation=35); ax.grid(axis="y",alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            fig,ax=plt.subplots(figsize=(5,3)); ax.boxplot(df[fare_col]); ax.set_ylabel("Fare"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown("### Data preview")
    st.dataframe(df.head(10),use_container_width=True,hide_index=True)
    st.download_button("Download cleaned dataset",df.to_csv(index=False).encode(), "aeroinsight_cleaned_data.csv","text/csv")
elif page=="Explore Data":
    st.markdown("## Explore your dataset")
    left,right=st.columns(2)
    with left:
        st.markdown("### Descriptive statistics")
        st.dataframe(df.describe(include="all").T,use_container_width=True)
    with right:
        st.markdown("### Missing values by column")
        x=df.isna().sum().sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(6,3.5)); x.plot.bar(ax=ax,color=BLUE); ax.set_ylabel("Missing cells"); ax.tick_params(axis="x",rotation=40); ax.grid(axis="y",alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.markdown("### Numeric correlation heatmap")
        numeric=df.select_dtypes(include=np.number)
        if len(numeric.columns)>=2:
            corr=numeric.corr(); fig,ax=plt.subplots(figsize=(6,4)); im=ax.imshow(corr,cmap="Blues",vmin=-1,vmax=1); ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns,rotation=45,ha="right"); ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns); fig.colorbar(im,ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else: st.info("Need at least two numeric columns.")
    st.markdown("### Filter records")
    col=st.selectbox("Filter by column",["None"]+list(df.columns))
    filtered=df
    if col!="None":
        vals=df[col].dropna().unique().tolist()
        if len(vals)<=100:
            chosen=st.multiselect("Values to keep",vals,default=vals); filtered=df[df[col].isin(chosen)]
        else: st.caption("Many unique values; showing all rows.")
    st.dataframe(filtered,use_container_width=True,hide_index=True)
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records.")
elif page=="Fare Prediction":
    st.markdown("## Flight fare prediction")
    st.caption("A Random Forest model learns patterns in your dataset. Predictions are estimates, not guaranteed ticket prices.")
    work=df.dropna(subset=[fare_col]).copy()
    features=[c for c in work.columns if c!=fare_col and work[c].nunique(dropna=True)>1]
    if len(work)<12 or not features:
        st.warning("Upload at least 12 rows and one useful feature column besides Price/Fare.")
    else:
        X=work[features].copy(); y=work[fare_col].astype(float)
        cat=list(X.select_dtypes(include=["object","category"]).columns)
        num=[c for c in X.columns if c not in cat]
        for c in cat: X[c]=X[c].fillna("Unknown").astype(str)
        for c in num: X[c]=pd.to_numeric(X[c],errors="coerce")
        transformers=[]
        if num: transformers.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median"))]),num))
        if cat: transformers.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat))
        model=Pipeline([("prep",ColumnTransformer(transformers)),("rf",RandomForestRegressor(n_estimators=150,random_state=42))])
        test_n=max(2,int(round(len(work)*.2)))
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_n,random_state=42)
        model.fit(Xtr,ytr); pred=model.predict(Xte)
        mae=mean_absolute_error(yte,pred); rmse=float(np.sqrt(mean_squared_error(yte,pred))); r2=r2_score(yte,pred)
        c1,c2,c3=st.columns(3)
        with c1: metric("MAE",f"₹{mae:,.0f}","Lower is generally better")
        with c2: metric("RMSE",f"₹{rmse:,.0f}","Penalizes larger errors")
        with c3: metric("R² SCORE",f"{r2:.3f}","Closer to 1 is generally better")
        st.markdown("### Enter flight details")
        with st.form("predict_form"):
            values={}; cols=st.columns(2)
            for i,col in enumerate(features):
                s=work[col]
                with cols[i%2]:
                    if col in cat:
                        opts=s.dropna().astype(str).value_counts().index.tolist() or ["Unknown"]
                        values[col]=st.selectbox(str(col),opts,key="pred_"+str(col))
                    else:
                        ns=pd.to_numeric(s,errors="coerce").dropna()
                        lo=float(ns.min()) if len(ns) else 0.0; hi=float(ns.max()) if len(ns) else 100.0; default=float(ns.median()) if len(ns) else 0.0
                        if lo==hi: hi=lo+1
                        values[col]=st.number_input(str(col),min_value=lo,max_value=hi,value=min(max(default,lo),hi),key="pred_"+str(col))
            submitted=st.form_submit_button("Predict estimated fare",type="primary",use_container_width=True)
        if submitted:
            row=pd.DataFrame([values],columns=features)
            for col in cat: row[col]=row[col].astype(str)
            for col in num: row[col]=pd.to_numeric(row[col],errors="coerce")
            result=float(model.predict(row)[0])
            st.success(f"Estimated fare: ₹{result:,.0f}")
            st.caption("Real fares may vary with date, demand, baggage, and availability.")
        st.markdown("### Actual vs predicted fares")
        fig,ax=plt.subplots(figsize=(8,4)); ax.scatter(yte,pred,alpha=.7,color=BLUE); lim=[min(float(yte.min()),float(np.min(pred))),max(float(yte.max()),float(np.max(pred)))]; ax.plot(lim,lim,"--",color=TEAL); ax.set_xlabel("Actual fare"); ax.set_ylabel("Predicted fare"); ax.grid(alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
else:
    st.markdown("## About AeroInsight")
    st.write("AeroInsight is a flight fare exploration and prediction dashboard built with Streamlit, pandas, Matplotlib, and scikit-learn.")
    st.markdown("- Upload your own CSV dataset\n- View summary KPI cards and six charts\n- Explore descriptive statistics, missing values, and correlations\n- Train a Random Forest regression model\n- Predict estimated fares and evaluate model performance")
    st.info("For best results, use a numeric Price or Fare column plus useful features such as Airline, Source, Destination, Duration, and Stops.")
