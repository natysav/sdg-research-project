import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import utils as u
from datetime import datetime
st.title("Company Contribution to a Better World")
st.markdown("""
    We believe that every individual and every organisation changes the world. 
    We want your opinion how the organisations are transforming our world in terms of 
    poverty, hunger, health, education, work conditions, energy, gender equality, 
    clean water, innovations, economy, communities, climate, 
    life of earth and under water, justice, collaboration. 
    More information on sustainable development goals can be found on United Nations website: 
    https://www.un.org/en/exhibits/page/sdgs-17-goals-transform-world """)
@st.cache_data
def get_sp100_companies():
    companies = [" ",
    "3M Company (MMM)",
    "Abbott Laboratories (ABT)",
    "AbbVie Inc. (ABBV)",
    "Accenture plc (ACN)",
    "Activision Blizzard, Inc. (ATVI)",
    "Adobe Inc. (ADBE)",
    "Advanced Micro Devices, Inc. (AMD)",
    "Aflac Incorporated (AFL)",
    "Air Products and Chemicals, Inc. (APD)",
    "Amgen Inc. (AMGN)",
    "Analog Devices, Inc. (ADI)",
    "Apple Inc. (AAPL)",
    "Applied Materials, Inc. (AMAT)",
    "AT&T Inc. (T)",
    "Bank of America Corp. (BAC)",
    "Berkshire Hathaway Inc. (BRK.B)",
    "BlackRock, Inc. (BLK)",
    "Boeing Company (BA)",
    "Bristol-Myers Squibb Company (BMY)",
    "Broadcom Inc. (AVGO)",
    "Caterpillar Inc. (CAT)",
    "Chevron Corporation (CVX)",
    "Cisco Systems, Inc. (CSCO)",
    "Citigroup Inc. (C)",
    "The Coca-Cola Company (KO)",
    "Colgate-Palmolive Company (CL)",
    "ConocoPhillips (COP)",
    "Costco Wholesale Corporation (COST)",
    "CVS Health Corporation (CVS)",
    "Danaher Corporation (DHR)",
    "Deere & Company (DE)",
    "The Home Depot, Inc. (HD)",
    "Honeywell International Inc. (HON)",
    "International Business Machines Corporation (IBM)",
    "Intel Corporation (INTC)",
    "Johnson & Johnson (JNJ)",
    "JPMorgan Chase & Co. (JPM)",
    "Lockheed Martin Corporation (LMT)",
    "Lowe's Companies, Inc. (LOW)",
    "Linde plc (LIN)",
    "Mastercard Incorporated (MA)",
    "Medtronic plc (MDT)",
    "Merck & Co., Inc. (MRK)",
    "Microsoft Corporation (MSFT)",
    "Moderna, Inc. (MRNA)",
    "Morgan Stanley (MS)",
    "Netflix, Inc. (NFLX)",
    "Nike, Inc. (NKE)",
    "NVIDIA Corporation (NVDA)",
    "Oracle Corporation (ORCL)",
    "PepsiCo, Inc. (PEP)",
    "Pfizer Inc. (PFE)",
    "Procter & Gamble Company (PG)",
    "QUALCOMM Incorporated (QCOM)",
    "Raytheon Technologies Corporation (RTX)",
    "Salesforce, Inc. (CRM)",
    "Schlumberger Limited (SLB)",
    "Simon Property Group, Inc. (SPG)",
    "Starbucks Corporation (SBUX)",
    "S&P Global Inc. (SPGI)",
    "Target Corporation (TGT)",
    "Texas Instruments Incorporated (TXN)",
    "The Travelers Companies, Inc. (TRV)",
    "Thermo Fisher Scientific Inc. (TMO)",
    "Union Pacific Corporation (UNP)",
    "UnitedHealth Group Incorporated (UNH)",
    "Verizon Communications Inc. (VZ)",
    "Visa Inc. (V)",
    "Walmart Inc. (WMT)",
    "Wells Fargo & Company (WFC)",
    "Xcel Energy Inc. (XEL)",
    "Amphenol Corporation (APH)",
    "Automatic Data Processing, Inc. (ADP)",
    "CME Group Inc. (CME)",
    "Crown Castle International Corp. (CCI)",
    "Ecolab Inc. (ECL)",
    "General Dynamics Corporation (GD)",
    "Gilead Sciences, Inc. (GILD)",
    "NextEra Energy, Inc. (NEE)",
    "Northrop Grumman Corporation (NOC)",
    "Norfolk Southern Corporation (NSC)",
    "Parker-Hannifin Corporation (PH)",
    "Public Storage (PSA)",
    "Regeneron Pharmaceuticals, Inc. (REGN)",
    "Ross Stores, Inc. (ROST)",
    "Southern Company (SO)",
    "The TJX Companies, Inc. (TJX)",
    "U.S. Bancorp (USB)",
    "Ventas, Inc. (VTR)",
    "Waste Management, Inc. (WM)",
    "Weyerhaeuser Company (WY)",
    "Zimmer Biomet Holdings, Inc. (ZBH)",
    "Meta Platforms, Inc. (META)",
    "American Express Company (AXP)",
    "Stryker Corporation (SYK)",
    "General Electric Company (GE)",
    "Eli Lilly and Company (LLY)",
    "Chubb Limited (CB)",
    "Aon plc (AON)",
    "Rockwell Automation, Inc. (ROK)"
    ]
    return companies

sp100_companies = get_sp100_companies()
company = st.selectbox("Select a Company", options=sp100_companies, index=0)
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.image("pages/images/E-WEB-Goal-01.png")
    sdg1 = st.slider('No Poverty', min_value=0, max_value=100, step=1)
    st.image("pages/images/E-WEB-Goal-07.png")
    sdg7 = st.slider('Clean Energy', min_value=0, max_value=100, step=1,)
    st.image("pages/images/E-WEB-Goal-13.png")
    sdg13 = st.slider('Climat Actions', min_value=0, max_value=100, step=1)
with c2:
    st.image("pages/images/E-WEB-Goal-02.png")
    sdg2 = st.slider('Zero Hunger', min_value=0, max_value=100, step=1)
    st.image("pages/images/E-WEB-Goal-08.png")
    sdg8 = st.slider('Decent Work', min_value=0, max_value=100, step=1,)
    st.image("pages/images/E-WEB-Goal-14.png")
    sdg14 = st.slider('Life below water', min_value=0, max_value=100, step=1)

with c3:
    st.image("pages/images/E-WEB-Goal-03.png")
    sdg3 = st.slider('Health', min_value=0, max_value=100, step=1)
    st.image("pages/images/E-WEB-Goal-09.png")
    sdg9 = st.slider('Innovations', min_value=0, max_value=100, step=1,)
    st.image("pages/images/E-WEB-Goal-15.png")
    sdg15 = st.slider('Life on land', min_value=0, max_value=100, step=1)

with c4:
    st.image("pages/images/E-WEB-Goal-04.png")
    sdg4 = st.slider('Education', min_value=0, max_value=100, step=1)
    st.image("pages/images/E-WEB-Goal-10.png")
    sdg10 = st.slider('Equality', min_value=0, max_value=100, step=1,)
    st.image("pages/images/E-WEB-Goal-16.png")
    sdg16 = st.slider('Peace and Justice', min_value=0, max_value=100, step=1)

with c5:
    st.image("pages/images/E-WEB-Goal-05.png")
    sdg5 = st.slider('Gender Equality', min_value=0, max_value=100, step=1)
    st.image("pages/images/E-WEB-Goal-11.png")
    sdg11 = st.slider('Communities', min_value=0, max_value=100, step=1,)
    st.image("pages/images/E-WEB-Goal-17.png")
    sdg17 = st.slider('Cooperation', min_value=0, max_value=100, step=1)
with c6:
    st.image("pages/images/E-WEB-Goal-06.png")
    sdg6 = st.slider('Clean Water', min_value=0, max_value=100, step=1)
    st.image("pages/images/E-WEB-Goal-12.png")
    sdg12 = st.slider('No Waste', min_value=0, max_value=100, step=1)
df = pd.DataFrame([{
    "Company": company,
    "SDG1": sdg1,
    "SDG2": sdg2,
    "SDG3": sdg3,
    "SDG4": sdg4,
    "SDG5": sdg5,
    "SDG6": sdg6,
    "SDG7": sdg7,
    "SDG8": sdg8,
    "SDG9": sdg9,
    "SDG10": sdg10,
    "SDG11": sdg11,
    "SDG12": sdg12,
    "SDG13": sdg13,
    "SDG14": sdg14,
    "SDG15": sdg15,
    "SDG16": sdg16,
    "SDG17": sdg17
}])
#st.dataframe(df)
# Calculate the total score as the average of the 17 SDG ratings
sdg_columns = [f"SDG{i}" for i in range(1, 18)]
df["Total Score"] = df[sdg_columns].mean(axis=1)

# Prepare data for the radar chart.
# Prepare data for polar area chart
values = df.loc[0, sdg_columns].tolist()
n = len(sdg_columns)
# Each wedge's angular width (in degrees)
width = 360 / n

# Generate angles so each wedge is centered appropriately
# Here theta gives the starting angle of each bar
angles = [i * width for i in range(n)]
center_angles = [angle + width/2 for angle in angles]
# List of 17 distinct colors (feel free to adjust as needed)
colors = [
    "#E5243B",  # SDG 1: No Poverty
    "#DDA63A",  # SDG 2: Zero Hunger
    "#4C9F38",  # SDG 3: Good Health & Well-being
    "#C5192D",  # SDG 4: Quality Education
    "#FF3A21",  # SDG 5: Gender Equality
    "#26BDE2",  # SDG 6: Clean Water & Sanitation
    "#FCC30B",  # SDG 7: Affordable & Clean Energy
    "#A21942",  # SDG 8: Decent Work & Economic Growth
    "#FD6925",  # SDG 9: Industry, Innovation & Infrastructure
    "#DD1367",  # SDG 10: Reduced Inequalities
    "#FD9D24",  # SDG 11: Sustainable Cities & Communities
    "#BF8B2E",  # SDG 12: Responsible Consumption & Production
    "#3F7E44",  # SDG 13: Climate Action
    "#0A97D9",  # SDG 14: Life Below Water
    "#56C02B",  # SDG 15: Life on Land
    "#00689D",  # SDG 16: Peace, Justice & Strong Institutions
    "#19486A"  # SDG 17: Partnerships for the Goals
]

# Create the polar area chart using Barpolar
fig = go.Figure()

fig.add_trace(go.Barpolar(
    r=values,
    theta=angles,
    width=[width]*n,
    marker_color=colors,
    marker_line_color="black",
    marker_line_width=1,
    opacity=0.8,
    text=sdg_columns,              # SDG names as text labels
    #textposition='inside',
    # Show which SDG and its score on hover
    hovertemplate = "<b>%{customdata}</b><br>Score: %{r}<extra></extra>",
    customdata = sdg_columns
))

fig.update_layout(
    title=f"{df.loc[0, 'Company']} SDG Scores (Avg: {df.loc[0, 'Total Score']:.2f})",
    polar=dict(
        angularaxis=dict(
            tickmode="array",
            tickvals=center_angles,
            ticktext=sdg_columns,
            rotation=90,  # rotate so the first label is at the top
            direction="clockwise"
        ),
        radialaxis=dict(
            visible=True,
            range=[0, 100]  # assuming the ratings are between 0 and 10
        )
    ),
    showlegend=False
)

st.plotly_chart(fig)



