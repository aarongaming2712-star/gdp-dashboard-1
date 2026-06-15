import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import time

# ==========================================
# 1. GLOBAL CONFIGURATION & CUSTOM DESIGN PALETTE
# ==========================================
st.set_page_config(
    page_title="Logistics Enterprise Data Quality Hub", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS Override for Premium Canvas Styling
st.markdown("""
    <style>
    /* Main Canvas Background Styling */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f1f5f9 !important;
    }
    
    /* Sidebar Background Overrides */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    
    /* Ensure Sidebar text and navigation elements match the new look */
    [data-testid="stSidebar"] *, [data-testid="stSidebarNav"] * {
        color: #f8fafc !important;
    }
    
    /* Top Header Navbar transparency match */
    [data-testid="stHeader"] {
        background-color: rgba(241, 245, 249, 0.85) !important;
        backdrop-filter: blur(8px);
    }

    /* Executive Hero Card Design */
    .hero-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 25px;
    }
    .hero-value { font-size: 64px; font-weight: 800; color: #ccff33; line-height: 1; }
    .hero-label { font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.9; }
    
    /* Modern Grid Metric Blocks */
    .metric-card {
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04); 
        border: 1px solid #e2e8f0; 
        margin-bottom: 5px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover { 
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
    }
    .metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .metric-title { font-size: 15px; font-weight: 600; color: #334155; }
    .badge { padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .badge-healthy { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .badge-warning { background-color: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
    .badge-critical { background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .metric-value { font-size: 38px; font-weight: 700; color: #0f172a; margin-bottom: 8px; line-height: 1; }
    .progress-bg { background-color: #e2e8f0; height: 6px; border-radius: 3px; width: 100%; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. DATA INGESTION ENGINE & VALIDATION REGISTRY
# ==========================================
@st.cache_data
def load_data():
    try:
        # Load production dataset
        return pd.read_csv("Train.csv")
    except FileNotFoundError:
        # Generate perfect baseline if file doesn't exist
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            'ID': np.arange(1, n + 1),
            'Warehouse_block': np.random.choice(['A', 'B', 'C', 'D', 'F'], n),
            'Mode_of_Shipment': np.random.choice(['Flight', 'Ship', 'Road'], n),
            'Customer_care_calls': np.random.randint(2, 8, n),
            'Customer_rating': np.random.randint(1, 6, n),
            'Cost_of_the_Product': np.random.randint(50, 350, n),
            'Prior_purchases': np.random.randint(2, 11, n),
            'Product_importance': np.random.choice(['low', 'medium', 'high'], n),
            'Gender': np.random.choice(['F', 'M'], n),
            'Discount_offered': np.random.randint(0, 65, n),
            'Weight_in_gms': np.random.randint(500, 6000, n),
            'Reached.on.Time_Y.N': np.random.choice([0, 1], n)
        })

df_raw = load_data()

REQUIREMENTS_TEXT = {
    'ID': "Unique positive integer sequence greater than 0",
    'Warehouse_block': "Single uppercase character matching ranges [A-F]",
    'Mode_of_Shipment': "String matches exactly: 'Flight', 'Ship', or 'Road'",
    'Customer_care_calls': "Numerical integer bounds between 2 and 7",
    'Customer_rating': "Numerical evaluation score strictly between 1 and 5",
    'Cost_of_the_Product': "Standard financial valuation greater than or equal to 0",
    'Prior_purchases': "Historical custom order profile range between 2 and 10",
    'Product_importance': "String value matches: 'low', 'medium', or 'high'",
    'Gender': "Customer gender identity string matching 'F' or 'M'",
    'Discount_offered': "Promotional deduction value greater than or equal to 0",
    'Weight_in_gms': "Physical package mass scale in grams greater than 0",
    'Reached.on.Time_Y.N': "Target performance criteria indicator mapping strictly to binaries [0, 1]"
}

MEASUREMENT_REGISTRY = {
    'ID': lambda series: (pd.to_numeric(series, errors='coerce') > 0).fillna(False),
    'Warehouse_block': lambda series: series.astype(str).str.match(r'^[A-F]$').fillna(False),
    'Mode_of_Shipment': lambda series: series.astype(str).isin(['Flight', 'Ship', 'Road']).fillna(False),
    'Customer_care_calls': lambda series: pd.to_numeric(series, errors='coerce').between(2, 7).fillna(False),
    'Customer_rating': lambda series: pd.to_numeric(series, errors='coerce').between(1, 5).fillna(False),
    'Cost_of_the_Product': lambda series: (pd.to_numeric(series, errors='coerce') >= 0).fillna(False),
    'Prior_purchases': lambda series: pd.to_numeric(series, errors='coerce').between(2, 10).fillna(False),
    'Product_importance': lambda series: series.astype(str).isin(['low', 'medium', 'high']).fillna(False),
    'Gender': lambda series: series.astype(str).isin(['F', 'M']).fillna(False),
    'Discount_offered': lambda series: (pd.to_numeric(series, errors='coerce') >= 0).fillna(False),
    'Weight_in_gms':   lambda series: (pd.to_numeric(series, errors='coerce') > 0).fillna(False),
    'Reached.on.Time_Y.N': lambda series: series.isin([0, 1]).fillna(False)
}

# ==========================================
# SIDEBAR CONTROLS & CONTROLLER VARIABLES
# ==========================================
with st.sidebar:
    st.logo("https://img.icons8.com/fluency/96/database.png", icon_image="https://img.icons8.com/fluency/48/database.png")
    st.markdown("### 🎛️ Governance Controls")
    
    target_threshold = st.slider("🎯 Target Accuracy Threshold (%)", min_value=80.0, max_value=100.0, value=95.0, step=0.5)
    
    st.markdown("---")
    st.markdown("### 🔍 Global Block Filter")
    selected_blocks = st.multiselect("Select Warehouse Blocks:", options=sorted(df_raw['Warehouse_block'].unique().tolist()), default=None)
    
    st.markdown("---")
    st.info("💡 **Multi-Dimensional Mode Active:** System rules engines are evaluating dataset integrity parameters.")

# Filter dataset based on sidebar block choices
filtered_df = df_raw.copy()
if selected_blocks:
    filtered_df = filtered_df[filtered_df['Warehouse_block'].isin(selected_blocks)]


# Initialize dynamic session states to store injected error configurations across script updates
if "inject_text_typos" not in st.session_state:
    st.session_state.inject_text_typos = 0
if "inject_math_outliers" not in st.session_state:
    st.session_state.inject_math_outliers = 0

# Apply the error injections directly to our operational runtime dataset copy
if st.session_state.inject_text_typos > 0:
    sample_size = int(len(filtered_df) * (st.session_state.inject_text_typos / 100))
    if sample_size > 0:
        corrupt_indices = filtered_df.sample(n=sample_size, random_state=42).index
        filtered_df.loc[corrupt_indices, 'Warehouse_block'] = 'X' # Breaks range rule [A-F]
        filtered_df.loc[corrupt_indices, 'Mode_of_Shipment'] = 'Drone' # Breaks category constraint

if st.session_state.inject_math_outliers > 0:
    sample_size = int(len(filtered_df) * (st.session_state.inject_math_outliers / 100))
    if sample_size > 0:
        corrupt_indices = filtered_df.sample(n=sample_size, random_state=24).index
        filtered_df.loc[corrupt_indices, 'Cost_of_the_Product'] = -999 # Breaks positive value rule
        filtered_df.loc[corrupt_indices, 'Customer_rating'] = 9 # Breaks max rating limit of 5
        filtered_df.loc[corrupt_indices, 'Weight_in_gms'] = -50 # Breaks logical package mass scale


# Run rules engines checks against computed/simulated states
registry_masks = {}
variable_data = {}
total_elements_audited = 0
total_elements_passed = 0

for column_name, validation_rule in MEASUREMENT_REGISTRY.items():
    if column_name in filtered_df.columns:
        validity_mask = validation_rule(filtered_df[column_name])
        registry_masks[column_name] = validity_mask
        passed = int(validity_mask.sum())
        total = int(len(filtered_df))
        quality_percentage = (passed / total) * 100 if total > 0 else 0
        
        if quality_percentage >= target_threshold:
            badge_class, fill_color = "badge-healthy", "#16a34a"
        elif quality_percentage >= (target_threshold - 10.0):
            badge_class, fill_color = "badge-warning", "#ea580c"
        else:
            badge_class, fill_color = "badge-critical", "#dc2626"
            
        variable_data[column_name] = {
            "score": quality_percentage, "badge": badge_class, "color": fill_color,
            "passed": passed, "total": total
        }
        total_elements_audited += total
        total_elements_passed += passed

global_dataset_score = (total_elements_passed / total_elements_audited) * 100 if total_elements_audited > 0 else 0

mask_df = pd.DataFrame(registry_masks)
failure_counts = (~mask_df).sum(axis=1)

priority_conditions = [
    (failure_counts >= 3),
    (failure_counts.between(1, 2)),
    (failure_counts == 0)
]
priority_choices = ['🔴 High Priority (3+ Inaccuracies)', '🟡 Medium Priority (1-2 Inaccuracies)', '🟢 Low Priority (Fully Compliant)']
filtered_df['Cleaning_Priority'] = np.select(priority_conditions, priority_choices, default='🟢 Low Priority (Fully Compliant)')


# ==========================================
# 3. PAGE 1: OVERVIEW SCOREBOARD
# ==========================================
def show_overview_page():
    st.title("✅ Scoreboard Overview")
    st.markdown("---")
    
    st.subheader("📊 Main KPI: Corporate Accuracy Scorecard")
    st.markdown(f"""
        <div class="hero-card">
            <div>
                <div class="hero-label">🏆 Aggregate System Performance</div>
                <h3 style='margin: 5px 0 0 0; color: white; font-size: 28px; font-weight: 600;'>Global Dataset Accuracy Score</h3>
            </div>
            <div class="hero-value">{global_dataset_score:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("###")

    def render_metric_card(container, col_title, key_name):
        if key_name in variable_data:
            v = variable_data[key_name]
            container.markdown(f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">{col_title}</span>
                        <span class="badge {v['badge']}">Accuracy</span>
                    </div>
                    <div class="metric-value">{v['score']:.1f}%</div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width: {v['score']}%; background-color: {v['color']};"></div>
                    </div>
                    <p style='font-size: 12px; color: #64748b; margin: 8px 0 0 0;'>{v['passed']:,} / {v['total']:,} accurate rows</p>
                </div>
                """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<p style='font-weight:700; color:#475569; margin-bottom:15px; letter-spacing:1px;'>📍 INDIVIDUAL VARIABLE METRICS</p>", unsafe_allow_html=True)
        row1 = st.columns(3)
        render_metric_card(row1[0], "🆔 System ID", "ID")
        render_metric_card(row1[1], "📦 Warehouse Block", "Warehouse_block")
        render_metric_card(row1[2], "🚢 Shipment Mode", "Mode_of_Shipment")

        row2 = st.columns(3)
        render_metric_card(row2[0], "📞 Customer Care Calls", "Customer_care_calls")
        render_metric_card(row2[1], "⭐ Customer Rating", "Customer_rating")
        render_metric_card(row2[2], "💰 Cost of the Product", "Cost_of_the_Product")

        row3 = st.columns(3)
        render_metric_card(row3[0], "⏳ Prior Purchases Count", "Prior_purchases")
        render_metric_card(row3[1], "⚡ Product Importance Tier", "Product_importance")
        render_metric_card(row3[2], "⚧ Customer Gender", "Gender")

        row4 = st.columns(3)
        render_metric_card(row4[0], "🏷️ Discount Offered", "Discount_offered")
        render_metric_card(row4[1], "⚖️ Package Weight", "Weight_in_gms")
        render_metric_card(row4[2], "🏁 Reached On Time (Y/N)", "Reached.on.Time_Y.N")

    st.markdown("---")
    st.subheader("🔍 Visualization: Accuracy Registry Field Distribution Map")
    
    with st.container(border=True):
        viz_col1, viz_col2 = st.columns(2)
        plot_df = filtered_df.copy()
        if 'Warehouse_block' in registry_masks and 'Customer_rating' in registry_masks:
            plot_df['Warehouse_Status'] = np.where(registry_masks['Warehouse_block'], '🟢 Accurate', '🔴 Inaccurate')
            plot_df['Rating_Status'] = np.where(registry_masks['Customer_rating'], '🟢 Accurate', '🔴 Inaccurate')
            
            with viz_col1:
                st.write("#### Warehouse Block Logs: Accurate vs Inaccurate Segmentations")
                warehouse_chart = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X('Warehouse_block:N', title="Warehouse Block Code Value"),
                    y=alt.Y('count()', title="Record Log Volume Count"),
                    color=alt.Color('Warehouse_Status:N', scale=alt.Scale(domain=['🟢 Accurate', '🔴 Inaccurate'], range=['#16a34a', '#dc2626']))
                ).properties(height=280)
                st.altair_chart(warehouse_chart, use_container_width=True)
                
            with viz_col2:
                st.write("#### Customer Rating Options: Accurate Boundaries vs Failure Deviations")
                rating_chart = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X('Customer_rating:O', title="Observed Input Rating Value"),
                    y=alt.Y('count()', title="Record Log Volume Count"),
                    color=alt.Color('Rating_Status:N', scale=alt.Scale(domain=['🟢 Accurate', '🔴 Inaccurate'], range=['#16a34a', '#dc2626']))
                ).properties(height=280)
                st.altair_chart(rating_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 Threshold Alerts: Data Quality Processing Pipelines")
    
    high_priority_df = filtered_df[filtered_df['Cleaning_Priority'] == '🔴 High Priority (3+ Inaccuracies)']
    med_priority_df = filtered_df[filtered_df['Cleaning_Priority'] == '🟡 Medium Priority (1-2 Inaccuracies)']
    compliant_df = filtered_df[filtered_df['Cleaning_Priority'] == '🟢 Low Priority (Fully Compliant)']

    tab_high, tab_med, tab_green = st.tabs([
        f"🔴 High Priority ({len(high_priority_df):,})", 
        f"🟡 Medium Priority ({len(med_priority_df):,})",
        f"🟢 Compliant Data ({len(compliant_df):,})"
    ])
    
    with tab_high:
        st.error(f"⚠️ **Critical Failure Alert:** These structural rows fail your target threshold configuration. Immediate correction required.")
        st.dataframe(high_priority_df, use_container_width=True)
        
    with tab_med:
        st.warning("⚡ **Cautionary Pipeline Variance:** These records showcase minor boundary slips. Check upstream extraction steps.")
        st.dataframe(med_priority_df, use_container_width=True)

    with tab_green:
        st.success("✅ **Clean Pass Framework:** Entries conform cleanly to defined logic boundaries.")
        st.dataframe(compliant_df, use_container_width=True)


# ==========================================
# 4. PAGE 2: REGISTRY & REQUIREMENTS PROFILE
# ==========================================
def show_registry_page():
    st.title("📋 Registry: Accuracy Dimension")
    st.markdown("Detailed breakdown of data quality rules and their enforcement status.")
    
    with st.status("Analyzing compliance constraints...", expanded=False) as status_box:
        time.sleep(0.2)
        st.write("📊 Evaluating row validation constraints...")
        status_box.update(label=f"Registry Verified Against target threshold!", state="complete", expanded=False)

    registry_summary = []
    for col in REQUIREMENTS_TEXT.keys():
        if col in variable_data:
            score = variable_data[col]["score"]
            status_text = "❌ FAILED" if score < target_threshold else "✅ PASS"
            
            registry_summary.append({
                "Column": col,
                "Requirement": REQUIREMENTS_TEXT[col],
                "Score (%)": f"{score:.6f}",
                "Status": status_text
            })
            
    summary_df = pd.DataFrame(registry_summary)

    def style_registry_grid(val):
        if val == "❌ FAILED":
            return 'background-color: #fee2e2; color: #991b1b; font-weight: bold; border: 1px solid #fecaca;'
        elif val == "✅ PASS":
            return 'background-color: #f0fdf4; color: #166534; font-weight: bold; border: 1px solid #bbf7d0;'
        return ''

    styled_summary = (summary_df.style
                      .map(style_registry_grid, subset=['Status']))
    
    with st.container(border=True):
        st.dataframe(styled_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🔍 Interactive Row-by-Row Field Inspector Mapping")
    
    audit_display_df = filtered_df.copy()
    for col in REQUIREMENTS_TEXT.keys():
        if col in audit_display_df.columns:
            audit_display_df[f"{col}_Valid"] = registry_masks[col].map({True: "🟢 Valid", False: "🔴 Invalid"})

    ordered_cols = []
    for col in REQUIREMENTS_TEXT.keys():
        if col in filtered_df.columns:
            ordered_cols.extend([col, f"{col}_Valid"])
            
    with st.expander("ℹ️ Help Window"):
        st.markdown("Use this expandable section to audit explicit values alongside automated test results.")
        
    with st.container(border=True):
        st.dataframe(audit_display_df[ordered_cols], use_container_width=True)


# ==========================================
# 5. NEW PAGE 3: SYNTHETIC STRESS TESTER & ERROR SIMULATOR
# ==========================================
def show_stress_tester_page():
    st.title("🛠️ Synthetic Pipeline Stress Tester & Error Simulator")
    st.markdown("Inject artificial discrepancies into the database sandbox to observe metrics changes and evaluate systemic error alerts.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🛠️ Error Ingestion Matrix")
        st.caption("Slide elements to inject real-world system anomalies into your active rows.")
        
        # Capture variables directly to browser state blocks
        st.session_state.inject_text_typos = st.slider(
            "Inject Logging Text Typos (%)", 
            min_value=0, max_value=100, 
            value=st.session_state.inject_text_typos, 
            step=5,
            help="Simulates warehouse typos by writing unknown codes (e.g., Block 'X', Mode 'Drone') into database files."
        )
        
        st.session_state.inject_math_outliers = st.slider(
            "Inject Sensor Value Outliers (%)", 
            min_value=0, max_value=100, 
            value=st.session_state.inject_math_outliers, 
            step=5,
            help="Simulates hardware logging errors by outputting out-of-bound variables (e.g., negative costs and package masses)."
        )
        
        st.write("###")
        if st.button("🧼 Reset Sandbox Data (Flush Anomalies)", use_container_width=True, type="primary"):
            st.session_state.inject_text_typos = 0
            st.session_state.inject_math_outliers = 0
            st.rerun()

    with col2:
        st.markdown("#### 📈 Dynamic Live Breakdown Analysis")
        st.write("Observe how simulated degradation trends interact with your configured corporate target thresholds.")
        
        # Display summary visualizer cards based on generated conditions
        if global_dataset_score < target_threshold:
            st.error(f"❌ **System Status: BREACHED**\nThe dataset accuracy ({global_dataset_score:.2f}%) has fallen underneath your target baseline threshold of {target_threshold}%.")
        else:
            st.success(f"✅ **System Status: STABLE**\nDataset metrics are holding above the required validation limits.")
            
        # Chart breakdown visualization representing real-time stability
        status_chart_df = pd.DataFrame({
            'Category': ['Accurate Database Values', 'Flagged Failures'],
            'Count': [total_elements_passed, total_elements_audited - total_elements_passed]
        })
        
        pie_chart = alt.Chart(status_chart_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal", scale=alt.Scale(range=['#16a34a', '#dc2626'])),
            tooltip=['Category', 'Count']
        ).properties(height=240)
        
        st.altair_chart(pie_chart, use_container_width=True)


# ==========================================
# 6. PAGE 4: RAW DATASET VIEW (WITH INTERACTIVE DROPDOWN COLUMN FILTER)
# ==========================================
def show_raw_dataset_page():
    st.title("🗃️ Raw System Dataset")
    st.markdown("Direct preview of the unmodified database log files compiled across all 12 operational columns.")
    st.markdown("---")
    
    original_columns = list(REQUIREMENTS_TEXT.keys())
    
    # INTERACTIVE SINGLE COLUMN SELECTOR FEATURE
    st.markdown("### 🎯 Single-Column Inspection Filter")
    column_options = ["View All Columns"] + original_columns
    selected_column = st.selectbox(
        "Choose a specific column to isolate and analyze:", 
        options=column_options,
        index=0,
        help="Select any individual column to hide the rest of the database and focus on its raw values."
    )
    
    # Dynamically slice dataset based on selected dropdown options
    if selected_column == "View All Columns":
        raw_preview_df = filtered_df[original_columns]
    else:
        raw_preview_df = filtered_df[[selected_column]]
        
    st.write("###")
    
    with st.container(border=True):
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Rows Processed", f"{len(raw_preview_df):,}")
        m_col2.metric("Total Active Dimensions", f"{len(raw_preview_df.columns)}")
        with m_col3:
            st.markdown("<div style='margin-top:5px;'></div>", unsafe_allow_html=True)
            if st.download_button(
                label="📥 Export Staged CSV Log",
                data=raw_preview_df.to_csv(index=False).encode('utf-8'),
                file_name="filtered_logistics_dataset.csv",
                mime="text/csv",
                use_container_width=True
            ):
                st.balloons()
        
    st.write("###")
    st.dataframe(raw_preview_df, use_container_width=True)


# ==========================================
# 7. NAVIGATION HUB EXECUTION
# ==========================================
if df_raw is not None:
    pages = {
        "Data Control Panel": [
            st.Page(show_overview_page, title="Scorecard Overview Dashboard", icon="📊"),
            st.Page(show_registry_page, title="Data Registry & Requirements", icon="📋"),
            st.Page(show_stress_tester_page, title="Pipeline Error Simulator", icon="🛠️"),
            st.Page(show_raw_dataset_page, title="Raw Dataset View", icon="🗃️"),
        ]
    }
    pg = st.navigation(pages)
    pg.run()
else:
    st.error("Fatal System Error. Processing pipeline cannot resolve raw data structures.")
