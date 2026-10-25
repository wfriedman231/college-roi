import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="College ROI Predictor", layout="wide")
st.title("College ROI Predictor")
st.caption("Upload the college major ROI file, explore a column, and see what a straight line predicts.")

# Part 2 of the assignment: one sentence on what the model is for and what happens when it is wrong.
st.markdown(
    "**What this model is for:** it gives a student or adviser a rough estimate of a graduate's "
    "10-year payoff from a few known facts (like GPA and school selectivity), and when it is wrong "
    "a student could take on debt for a degree that pays back far less than predicted, so treat it "
    "as a starting point, not a guarantee."
)

# Same shape as the first app: the user uploads a file,
# and everything else sits inside an if.
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------- 1. Preview ----------
    st.subheader("1. Preview")
    st.write(f"{len(df):,} rows and {df.shape[1]} columns.")
    st.dataframe(df.head(10), width="stretch")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("This file needs at least three numeric columns: two inputs and one target.")
        st.stop()

    # ---------- 2. Chart ----------
    # The user picks a column and a chart type.
    # A scatter needs a second column, so we only ask for one when scatter is chosen.
    st.subheader("2. Explore a column")
    c1, c2 = st.columns(2)
    chart_col = c1.selectbox("Column", numeric_cols, index=numeric_cols.index("net_roi_usd") if "net_roi_usd" in numeric_cols else 0)
    chart_type = c2.radio("Chart type", ["Histogram", "Scatter", "Line"], horizontal=True)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=chart_col, nbins=50, title=f"Distribution of {chart_col}")
    elif chart_type == "Scatter":
        other_cols = [c for c in numeric_cols if c != chart_col]
        x_scatter = st.selectbox("Plot it against", other_cols)
        # 30,000 dots is a lot; a sample keeps the chart fast and readable.
        sample = df.sample(min(len(df), 3000), random_state=1)
        fig = px.scatter(sample, x=x_scatter, y=chart_col, opacity=0.4,
                         title=f"{chart_col} versus {x_scatter} (sample of {len(sample):,} rows)")
    else:
        fig = px.line(df.reset_index(), x="index", y=chart_col, title=f"{chart_col} by row")
    st.plotly_chart(fig, width="stretch")

    # ---------- 3. Model ----------
    # This time we use two inputs instead of one.
    st.subheader("3. Predict")
    default_x1 = numeric_cols.index("gpa") if "gpa" in numeric_cols else 0
    default_x2 = numeric_cols.index("institution_selectivity_pctile") if "institution_selectivity_pctile" in numeric_cols else 1

    col1, col2, col3 = st.columns(3)
    x1 = col1.selectbox("Input 1: what you know", numeric_cols, index=default_x1)
    x2_options = [c for c in numeric_cols if c != x1]
    x2 = col2.selectbox("Input 2: what else you know", x2_options,
                        index=x2_options.index(numeric_cols[default_x2]) if numeric_cols[default_x2] in x2_options else 0)
    y_options = [c for c in numeric_cols if c not in (x1, x2)]
    y_col = col3.selectbox("Target: what you want to predict", y_options,
                           index=y_options.index("net_roi_usd") if "net_roi_usd" in y_options else 0)

    # The order of this list is the order the model learns.
    features = [x1, x2]
    data = df[features + [y_col]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in all three columns.")
        st.stop()

    x = data[features]
    y = data[y_col]
    model = LinearRegression()
    model.fit(x, y)

    # One number per input, plus R squared.
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Each extra 1 of {x1} adds", f"{model.coef_[0]:,.2f}")
    m2.metric(f"Each extra 1 of {x2} adds", f"{model.coef_[1]:,.2f}")
    m3.metric("R2 on this data", f"{model.score(x, y):.2f}")

    # The user types values, and the app reads the answer from the model.
    # The table we give to predict has the SAME column names in the SAME order
    # as the training data: [x1, x2].
    st.markdown("**Try it**")

    # Pick the right kind of input for each column.
    # A yes/no column (only 0 and 1) gets a choice of 0 or 1, not a free number:
    # its mean (e.g. 0.52) is not a value any real graduate can have.
    # A whole-number column steps by 1; anything else is a free decimal.
    def value_input(container, col):
        values = data[col]
        if set(values.unique()) <= {0, 1}:
            return container.selectbox(f"Choose a value for {col}", [0, 1],
                                       format_func=lambda v: f"{v} ({'yes' if v else 'no'})")
        if pd.api.types.is_integer_dtype(values):
            return container.number_input(f"Enter a value for {col}",
                                          value=int(round(values.median())), step=1)
        return container.number_input(f"Enter a value for {col}", value=float(values.mean()))

    v1, v2 = st.columns(2)
    value1 = value_input(v1, x1)
    value2 = value_input(v2, x2)

    new_row = pd.DataFrame({x1: [value1], x2: [value2]})[features]
    prediction = model.predict(new_row)[0]
    st.metric(f"Predicted {y_col}", f"{prediction:,.0f}")

    # The warning: a line does not know where its data stopped.
    for col, val in [(x1, value1), (x2, value2)]:
        low, high = data[col].min(), data[col].max()
        if val < low or val > high:
            st.warning(f"{col} = {val:,.2f} is outside the range the model learned from "
                       f"({low:,.2f} to {high:,.2f}). Treat the number with caution.")

    if model.score(x, y) < 0.2:
        st.info("R2 is low: these two inputs explain only a small part of the target. "
                "Individual graduates will often land far from this prediction.")
else:
    st.info("Upload college_major_roi.csv (or any CSV with at least three numeric columns) to get started.")
