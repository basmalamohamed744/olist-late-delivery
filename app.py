import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta

st.set_page_config(page_title="Late Delivery Risk", page_icon="📦", layout="wide")

ART = ""


@st.cache_resource
def load_assets():
    model = joblib.load(ART + "late_delivery_pipeline.joblib")
    cfg = json.load(open(ART + "config.json"))
    cust = pd.read_csv(ART + "customer_state_centroids.csv", index_col=0)
    sell = pd.read_csv(ART + "seller_state_centroids.csv", index_col=0)
    sellers = pd.read_csv(ART + "seller_stats.csv")
    return model, cfg, cust, sell, sellers


model, cfg, cust_c, sell_c, sellers = load_assets()


def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def risk_level(p):
    if p >= cfg["q95"]:
        return "Very high", "🔴", "Top 5% riskiest orders: contact the carrier and warn the customer now."
    if p >= cfg["q90"]:
        return "High", "🟠", "Top 10% riskiest orders: prioritize tracking and consider proactive notice."
    if p >= cfg["q80"]:
        return "Medium", "🟡", "Top 20% riskiest orders: keep on the watch list."
    return "Low", "🟢", "No special action needed."


st.title("📦 Late Delivery Risk Predictor")
st.caption("Brazilian e-commerce marketplace (Olist). Estimates, at purchase time, how likely an order is to arrive after its estimated delivery date.")

tab1, tab2, tab3 = st.tabs(["Order risk", "Seller late rates", "About"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Order")
        category = st.selectbox("Product category", cfg["categories"])
        price = st.number_input("Price (BRL)", min_value=1.0, max_value=20000.0, value=120.0, step=10.0)
        freight = st.number_input("Freight (BRL)", min_value=0.0, max_value=2000.0, value=20.0, step=1.0)
        n_items = st.number_input("Number of items", min_value=1, max_value=30, value=1)
        n_sellers = st.number_input("Number of sellers in the order", min_value=1, max_value=10, value=1)
    with c2:
        st.subheader("Product")
        weight = st.number_input("Weight (g)", min_value=1.0, max_value=40000.0,
                                 value=float(cfg["median_weight_g"]), step=50.0)
        length = st.number_input("Length (cm)", min_value=1.0, max_value=200.0, value=25.0)
        height = st.number_input("Height (cm)", min_value=1.0, max_value=200.0, value=15.0)
        width = st.number_input("Width (cm)", min_value=1.0, max_value=200.0, value=20.0)
    with c3:
        st.subheader("Locations and dates")
        seller_state = st.selectbox("Seller state", cfg["seller_states"],
                                    index=cfg["seller_states"].index("SP") if "SP" in cfg["seller_states"] else 0)
        customer_state = st.selectbox("Customer state", cfg["customer_states"],
                                      index=cfg["customer_states"].index("RJ") if "RJ" in cfg["customer_states"] else 0)
        purchase_date = st.date_input("Purchase date", value=date.today())
        purchase_hour = st.slider("Purchase hour", 0, 23, 12)
        est_date = st.date_input("Estimated delivery date shown to customer",
                                 value=date.today() + timedelta(days=20))

    if st.button("Predict risk", type="primary"):
        est_days = (est_date - purchase_date).days
        if est_days <= 0:
            st.error("The estimated delivery date must be after the purchase date.")
        elif est_days > 150:
            st.error("Estimated delivery is more than 150 days away, which is outside the range the model saw.")
        else:
            try:
                same_state = int(seller_state == customer_state)
                if same_state:
                    dist = cfg["same_state_median_km"]
                else:
                    dist = haversine(sell_c.loc[seller_state, "sell_lat"], sell_c.loc[seller_state, "sell_lng"],
                                     cust_c.loc[customer_state, "cust_lat"], cust_c.loc[customer_state, "cust_lng"])

                row = pd.DataFrame([{
                    "purchase_month": purchase_date.month,
                    "purchase_dow": purchase_date.weekday(),
                    "purchase_hour": purchase_hour,
                    "estimated_days": float(est_days),
                    "n_items": int(n_items),
                    "price": float(price),
                    "freight": float(freight),
                    "n_sellers": int(n_sellers),
                    "category": category,
                    "product_weight_g": float(weight),
                    "volume_cm3": float(length * height * width),
                    "customer_state": customer_state,
                    "seller_state": seller_state,
                    "distance_km": float(dist),
                    "same_state": same_state,
                    "freight_ratio": float(freight / (price + 1)),
                }])

                p = float(model.predict_proba(row)[0, 1])
                level, icon, advice = risk_level(p)

                m1, m2, m3 = st.columns(3)
                m1.metric("Late probability", f"{p:.1%}")
                m2.metric("Risk level", f"{icon} {level}")
                m3.metric("Estimated distance", f"{dist:,.0f} km")
                st.info(advice)
                st.caption("Average late rate in the training data is about 8%. "
                           "Distance is approximated from state centers.")
            except Exception as e:
                st.error(f"Could not compute a prediction for these inputs: {e}")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Seller late rates")
    st.caption("Sellers with at least 30 delivered orders in the dataset.")
    f1, f2 = st.columns(2)
    states = ["All"] + sorted(sellers["seller_state"].unique().tolist())
    sel_state = f1.selectbox("Seller state", states)
    min_orders = f2.slider("Minimum orders", 30, int(sellers["n_orders"].max()), 30)

    view = sellers[sellers["n_orders"] >= min_orders]
    if sel_state != "All":
        view = view[view["seller_state"] == sel_state]
    view = view.sort_values("late_rate", ascending=False).reset_index(drop=True)
    view["late_rate"] = (view["late_rate"] * 100).round(1)
    view = view.rename(columns={"late_rate": "late_rate_%"})

    st.write(f"{len(view)} sellers. Overall average late rate: {sellers['late_rate'].mean():.1%} (seller average).")
    st.dataframe(view, use_container_width=True, height=420)

    pick = st.text_input("Look up a seller ID")
    if pick:
        hit = view[view["seller_id"] == pick.strip()]
        st.dataframe(hit) if len(hit) else st.warning("Seller not found in the current filter.")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("""
**What this is.** A gradient boosting model trained on about 96K delivered orders from Olist (2016 to 2018).

**How to read the result.** The score ranks orders by risk. It is not a guarantee. On held-out data, the top 10% riskiest orders contained about 40% of all late orders.

**Limitations**
- Data comes from one marketplace and from 2016 to 2018, so patterns may have changed.
- The purchase month is an important feature, but late spikes in the data happened in specific periods, so it may not repeat. Performance on a time-based split was weaker than on a random split.
- The model does not know about seller or carrier problems that appear after purchase.
- Distance is estimated from state centers, not exact addresses.

**Data source:** Brazilian E-Commerce Public Dataset by Olist (Kaggle).
""")
