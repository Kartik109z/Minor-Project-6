# QUICKCART STOCKOUT RISK - COMPLETE PROJECT
# Requires: pandas, numpy, scikit-learn, openpyxl, reportlab
# Input files: fact_inventory_daily.csv, dim_suppliers.csv, dim_stores.csv, dim_skus.csv, dim_events.csv

import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix

BASE="."
NA_KW=dict(na_values=["N/A","missing","--","NA","null"], keep_default_na=True)

fact=pd.read_csv(f"{BASE}/fact_inventory_daily.csv",**NA_KW)
sup=pd.read_csv(f"{BASE}/dim_suppliers.csv",**NA_KW)
stores=pd.read_csv(f"{BASE}/dim_stores.csv",**NA_KW)
skus=pd.read_csv(f"{BASE}/dim_skus.csv",**NA_KW)
events=pd.read_csv(f"{BASE}/dim_events.csv",**NA_KW)
fact["date"]=pd.to_datetime(fact["date"]); events["date"]=pd.to_datetime(events["date"])

# Sanity checks
assert len(fact)==21600
assert len(stores)==12 and len(skus)==60 and len(sup)==15 and len(events)==30
assert fact["stockout_risk"].value_counts().to_dict()=={"Safe":14131,"At-Risk":5186,"Imminent":2283}
assert fact["date"].nunique()==30

# Joins
stores["city_display_clean"]=stores["city_display"].str.title()
df=(fact.merge(stores,on="store_id",validate="many_to_one")
       .merge(skus,on=["sku_id","supplier_id"],validate="many_to_one")
       .merge(sup,on="supplier_id",validate="many_to_one",suffixes=("","_sup"))
       .merge(events,on="date",validate="many_to_one"))
df=df.sort_values(["store_id","sku_id","date"]).reset_index(drop=True)

# Feature engineering
df["reorder_gap"]=df["reorder_point"]-df["closing_stock"]
df["days_of_cover_ratio"]=df["days_of_cover"]/df["lead_time_days_expected"].replace(0,np.nan)
df["supplier_reliability_clean"]=df["reliability_score"].fillna(sup["reliability_score"].median())
df["is_recent_reorder_3d"]=(df.groupby(["store_id","sku_id"])["reorder_placed"]
    .transform(lambda x:x.eq("Y").shift(1).rolling(3,min_periods=1).max()).fillna(0).astype(int))
df["day_of_month"]=df.date.dt.day
fest_start=pd.Timestamp("2026-10-22")
df["days_since_festival_start"]=np.where(df.date>=fest_start,(df.date-fest_start).dt.days,-1)
df["is_perishable_flag"]=(df["is_perishable"]=="Y").astype(int)
df["festive_relevant_flag"]=(df["festive_relevant"]=="Y").astype(int)
df["reorder_placed_flag"]=(df["reorder_placed"]=="Y").astype(int)
df["festival_week_flag"]=(df["is_festival_week"]=="Y").astype(int)

features=["opening_stock","units_demanded","units_sold","closing_stock","reorder_point",
"lead_time_days_expected","sales_velocity_7d","days_of_cover","reorder_gap","days_of_cover_ratio",
"supplier_reliability_clean","is_recent_reorder_3d","day_of_month","days_since_festival_start",
"is_perishable_flag","festive_relevant_flag","reorder_placed_flag","festival_week_flag",
"store_size","category","popularity_tier","city","event_type"]

X=df[features]; y=df["stockout_risk"]
train=df.date<=pd.Timestamp("2026-10-23"); test=~train
Xtr,Xte=X[train],X[test]; ytr,yte=y[train],y[test]
cat=[c for c in X.columns if X[c].dtype=="object"]; num=[c for c in X.columns if c not in cat]
pre=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])

models={
"Logistic Regression":LogisticRegression(max_iter=2000,class_weight="balanced"),
"Random Forest":RandomForestClassifier(n_estimators=350,max_depth=14,min_samples_leaf=2,
    class_weight="balanced_subsample",random_state=42,n_jobs=-1),
"Gradient Boosting":GradientBoostingClassifier(random_state=42,n_estimators=200,max_depth=3,learning_rate=0.06)
}

for name,model in models.items():
    pipe=Pipeline([("pre",pre),("model",model)])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    pr,re,fu,_=precision_recall_fscore_support(yte,pred,labels=["Safe","At-Risk","Imminent"],zero_division=0)
    print(name)
    print("accuracy:",round(accuracy_score(yte,pred),4))
    print("balanced_accuracy:",round(balanced_accuracy_score(yte,pred),4))
    print("Imminent precision/recall/F1:",round(pr[2],4),round(re[2],4),round(fu[2],4))
    print("macro F1:",round(f1_score(yte,pred,average="macro"),4))
    print()

# Final model: cost-sensitive Imminent decision
final_model=Pipeline([("pre",pre),("model",models["Gradient Boosting"])])
final_model.fit(Xtr,ytr)
proba=final_model.predict_proba(Xte)
classes=final_model.named_steps["model"].classes_
proba[:,list(classes).index("Imminent")]*=1.7
final_pred=classes[proba.argmax(axis=1)]

pr,re,fu,_=precision_recall_fscore_support(yte,final_pred,labels=["Safe","At-Risk","Imminent"],zero_division=0)
print("FINAL MODEL")
print("accuracy:",round(accuracy_score(yte,final_pred),4))
print("balanced_accuracy:",round(balanced_accuracy_score(yte,final_pred),4))
print("Imminent precision/recall/F1:",round(pr[2],4),round(re[2],4),round(fu[2],4))
print("macro F1:",round(f1_score(yte,final_pred,average="macro"),4))
print(confusion_matrix(yte,final_pred,labels=["Safe","At-Risk","Imminent"]))

# Save test predictions
out=df.loc[test,["date","store_id","sku_id","supplier_id","category","city",
    "is_festival_week","days_of_cover","lead_time_days_expected",
    "supplier_reliability_clean","stockout_risk"]].copy()
out["predicted_risk"]=final_pred
out["correct"]=out.stockout_risk==out.predicted_risk
out.to_csv("QuickCart_test_predictions.csv",index=False)
