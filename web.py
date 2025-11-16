import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Load data
df = pd.read_csv("Affordable_Housing_by_Town_2011-2022 2.csv")

# Clean up column name with leading space
df = df.rename(columns={
    " Single Family CHFA/ USDA Mortgages": "Single Family CHFA/USDA Mortgages"
})

st.title("Affordable Housing in Connecticut Towns (2011–2022)")

st.write(
    "Explore affordable housing statistics by town and year. "
    "Use the controls in each section to interact with the dataset."
)

# Make a list of years and towns for widgets
years = sorted(df["Year"].unique())
towns = sorted(df["Town"].unique())

# ============================================
# 1. Feature viz: Percent Affordable by town for a chosen year
# ============================================
st.header("1. Percent Affordable by Town")

year_selected = st.selectbox("Select a year", years, key="year_bar")

df_year = df[df["Year"] == year_selected].copy()

# Optional: allow user to limit to top N towns by Percent Affordable
top_n = st.slider("Show top N towns by Percent Affordable", 5, 50, 20)

df_year_sorted = df_year.sort_values("Percent Affordable", ascending=False).head(top_n)

bar_chart = (
    alt.Chart(df_year_sorted)
    .mark_bar()
    .encode(
        x=alt.X("Percent Affordable:Q", title="Percent Affordable"),
        y=alt.Y("Town:N", sort="-x", title="Town"),
        tooltip=[
            "Town",
            "Percent Affordable",
            "Total Assisted Units",
            "2010 Census Units",
        ],
    )
)

st.altair_chart(bar_chart, use_container_width=True)

# ============================================
# 2. Feature viz: Scatter of Census Units vs Percent Affordable
# ============================================
st.header("2. Census Units vs Percent Affordable")

# Widgets for filtering by year range
min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
year_range = st.slider(
    "Select year range",
    min_year,
    max_year,
    (min_year, max_year),
    key="year_range_scatter",
)

df_range = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])].copy()

# Filter by minimum census units
min_units = float(df_range["2010 Census Units"].min())
max_units = float(df_range["2010 Census Units"].max())
units_low, units_high = st.slider(
    "Filter towns by 2010 Census Units",
    min_units,
    max_units,
    (min_units, max_units),
    key="units_filter",
)

df_range = df_range[
    (df_range["2010 Census Units"] >= units_low)
    & (df_range["2010 Census Units"] <= units_high)
].copy()

scatter_chart = (
    alt.Chart(df_range)
    .mark_circle(size=60)
    .encode(
        x=alt.X("2010 Census Units:Q", title="2010 Census Units"),
        y=alt.Y("Percent Affordable:Q", title="Percent Affordable"),
        color=alt.Color("Year:O", title="Year"),
        tooltip=[
            "Town",
            "Year",
            "2010 Census Units",
            "Total Assisted Units",
            "Percent Affordable",
        ],
    )
    .interactive()
)

st.altair_chart(scatter_chart, use_container_width=True)

st.write("Number of points:", len(df_range))

# ============================================
# 3. Feature viz: Time series of Percent Affordable for one town
# ============================================
st.header("3. Time Series for a Selected Town")

town_selected = st.selectbox("Select a town", towns, key="town_timeseries")

df_town = df[df["Town"] == town_selected].copy()
df_town = df_town.sort_values("Year")

line_chart = (
    alt.Chart(df_town)
    .mark_line(point=True)
    .encode(
        x=alt.X("Year:O", title="Year"),
        y=alt.Y("Percent Affordable:Q", title="Percent Affordable"),
        tooltip=["Year", "Percent Affordable", "Total Assisted Units"],
    )
    .interactive()
)

st.altair_chart(line_chart, use_container_width=True)

st.write(
    f"Showing {len(df_town)} data points for {town_selected} "
    "over the available years."
)


# 4. Linear regression: Total Assisted Units vs Percent Affordable

st.header("4. Linear Regression")

st.write(
    "We will fit a simple linear model: "
    "`Percent Affordable = a + b * Total Assisted Units` "
    "on a chosen year."
)

reg_year = st.selectbox("Select year for regression", years, key="reg_year")

df_reg = df[df["Year"] == reg_year].copy()

# Drop rows with missing values to avoid errors
df_reg = df_reg.dropna(subset=["Total Assisted Units", "Percent Affordable"])

if len(df_reg) < 2:
    st.warning("Not enough data points for regression in this year.")
else:
    x = df_reg["Total Assisted Units"].values
    y = df_reg["Percent Affordable"].values

    # Fit linear regression using numpy
    # y = a + b x
    b, a = np.polyfit(x, y, 1)

    st.write(f"Fitted model for {reg_year}:")
    st.latex(r"\text{Percent Affordable} = %.3f + %.5f \times \text{Total Assisted Units}" % (a, b))

    # Create a dataframe for the regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a + b * x_line
    df_line = pd.DataFrame({
        "Total Assisted Units": x_line,
        "Percent Affordable (fitted)": y_line,
    })

    scatter_reg = (
        alt.Chart(df_reg)
        .mark_circle(size=60)
        .encode(
            x=alt.X("Total Assisted Units:Q", title="Total Assisted Units"),
            y=alt.Y("Percent Affordable:Q", title="Percent Affordable"),
            tooltip=["Town", "Total Assisted Units", "Percent Affordable"],
        )
    )

    line_reg = (
        alt.Chart(df_line)
        .mark_line()
        .encode(
            x="Total Assisted Units:Q",
            y="Percent Affordable (fitted):Q",
        )
    )

    st.altair_chart(scatter_reg + line_reg, use_container_width=True)

    
    y_pred = a + b * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    st.write(f"R² for this regression: {r2:.3f}")
