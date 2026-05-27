import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration & Global Styling
st.set_page_config(
    page_title="Titanic Advanced BI Space",
    page_icon="🚢",
    layout="wide"
)

# Central Color Palette Strategy (Green for Survived, Red for Died)
COLOR_MAP = {'Survived': '#2ca02c', 'Died': '#d62728'}

st.title("🚢 Titanic Advanced BI & Predictive Machine Learning Suite")
st.markdown("An immersive data space exploring historical survival pathways, behavioral correlations, and machine learning scenarios.")

# 2. Data Engine & Feature Engineering
@st.cache_data
def load_and_engineer_data():
    df = pd.read_csv('titanic_cleaned.csv')
    
    # Standardizing classifications & Text Transformations
    df['Survival Status'] = df['is_survived'].map({1: 'Survived', 0: 'Died'})
    df['Ticket Class'] = df['passenger_class'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    df['Embarkation Port'] = df['embarkation_port'].map({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}).fillna('Unknown')
    
    # Feature Engineering 1: Family Layouts
    df['Family Size'] = df['siblings_spouses_count'] + df['parents_children_count']
    
    # Feature Engineering 2: Age Slices (Brackets) for clean Categorical Grouping
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child (0-12)', 'Teenager (13-18)', 'Young Adult (19-35)', 'Adult (36-60)', 'Senior (60+)']
    df['Age Bracket'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    # Feature Engineering 3: Extract Deck Level from Cabin Code
    df['Deck'] = df['cabin_code'].str[0].str.upper()
    df['Deck'] = df['Deck'].replace({'U': 'Unknown (No Cabin Data)'})
    
    # Feature Engineering 4: Title Extraction
    df['Title'] = df['passenger_name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Dr', 'Rev', 'Mlle', 'Major', 'Col', 'Countess', 'Capt', 'Ms', 'Sir', 'Lady', 'Mme', 'Don', 'Jonkheer']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare/Noble')
    
    return df

# 3. Machine Learning Training Matrix
@st.cache_resource
def train_predictive_model(data):
    ml_df = data.copy()
    ml_df['gender_encoded'] = ml_df['gender'].map({'male': 0, 'female': 1})
    ml_df['embark_encoded'] = ml_df['embarkation_port'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)
    
    features = ['passenger_class', 'gender_encoded', 'age', 'fare_paid', 'Family Size', 'embark_encoded']
    X = ml_df[features]
    y = ml_df['is_survived']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)
    return model, features

# Execution Checks
try:
    df = load_and_engineer_data()
    ml_model, ml_features = train_predictive_model(df)
except FileNotFoundError:
    st.error("Missing dataset! Place 'titanic_cleaned.csv' inside the folder containing this app script.")
    st.stop()

# 4. Sidebar Controller Panel
st.sidebar.header("🕹️ Global Dashboard Filters")
search_query = st.sidebar.text_input("🔍 Search Passenger Name:", "")

# Filter Selections
genders = sorted(df['gender'].unique().tolist())
selected_genders = st.sidebar.multiselect("Gender Selection:", options=genders, default=genders)

classes = sorted(df['Ticket Class'].unique().tolist())
selected_classes = st.sidebar.multiselect("Class Tier Selection:", options=classes, default=classes)

decks = sorted(df['Deck'].unique().tolist())
selected_decks = st.sidebar.multiselect("Ship Deck Filter:", options=decks, default=decks)

# Slicing active frame
filtered_df = df[
    (df['gender'].isin(selected_genders)) &
    (df['Ticket Class'].isin(selected_classes)) &
    (df['Deck'].isin(selected_decks))
]

if search_query:
    filtered_df = filtered_df[filtered_df['passenger_name'].str.contains(search_query, case=False, na=False)]

# 5. Metric Dashboard Ribbon Layout
if filtered_df.empty:
    st.warning("No records matched your specific filter configurations. Readjust your sidebar limits.")
else:
    col1, col2, col3, col4 = st.columns(4)
    total_count = len(filtered_df)
    survived_count = (filtered_df['Survival Status'] == 'Survived').sum()
    survival_rate = (survived_count / total_count * 100) if total_count > 0 else 0
    
    with col1:
        st.metric("Filtered Population", f"{total_count:,} passengers")
    with col2:
        st.metric("Survival Probability", f"{survival_rate:.1f}%")
    with col3:
        st.metric("Avg Fare Cost", f"${filtered_df['fare_paid'].mean():.2f}")
    with col4:
        st.metric("Dominant Social Demographic", f"{filtered_df['Title'].mode()[0]}")

    st.markdown("---")

    # 6. Structured Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Advanced Visualization Space", "🔮 AI Survival Simulator", "📋 Detailed Data Matrix"])

    # ---- TAB 1: ADVANCED VISUALIZATIONS ----
    with tab1:
        # High Impact Row: Parallel Categorical Pathways
        st.subheader("Demographic Survival Ribbons Flow")
        st.markdown("*Trace the structural flows of survival lines directly across ticket tiers and gender classifications.*")
        
        fig_parallel = px.parallel_categories(
            filtered_df,
            dimensions=['Ticket Class', 'gender', 'Survival Status'],
            color='is_survived',
            color_continuous_scale=[[0, '#d62728'], [1, '#2ca02c']],
            template="plotly_white"
        )
        fig_parallel.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_parallel, use_container_width=True)
        
        st.markdown("---")
        
       # Heatmap
        st.subheader("Survival Probability Hotspots (Heatmap)")
        st.markdown("*Concentration of survival counts mapped across Ticket Classes vs. Age Brackets.*")

        # Filter specifically for survivors to see concentrations via the heatmap matrix
        survivor_only_df = filtered_df[filtered_df['Survival Status'] == 'Survived']

        fig_heatmap = px.density_heatmap(
            survivor_only_df,
            x="Age Bracket",
            y="Ticket Class",
            z="is_survived",
            histfunc="sum",
            category_orders={
                "Ticket Class": ["1st Class", "2nd Class", "3rd Class"],
                "Age Bracket": [
                    'Child (0-12)',
                    'Teenager (13-18)',
                    'Young Adult (19-35)',
                    'Adult (36-60)',
                    'Senior (60+)'
                ]
            },
            color_continuous_scale="Viridis",
            template="plotly_white",
            labels={'is_survived': 'Survivors'}
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)


        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Financial Fare Density Splitting")
            st.markdown("*Pinpointing the exact price distribution overlay where life odds turned favorable.*")
            fig_density = px.histogram(
                filtered_df,
                x="fare_paid",
                color="Survival Status",
                barmode="overlay",
                nbins=40,
                color_discrete_map=COLOR_MAP,
                template="plotly_white",
                labels={'fare_paid': 'Ticket Fare Amount ($)'}
            )
            fig_density.update_layout(yaxis_title="Passenger Volume Headcount")
            st.plotly_chart(fig_density, use_container_width=True)
            
        with col_g2:
            st.subheader("Survival Proportions by Ship Deck Level")
            st.markdown("*Analyzing cabin vertical positions relative to main deck lifeboat access.*")
            fig_deck = px.histogram(
                filtered_df,
                x="Deck",
                color="Survival Status",
                barnorm="percent",
                category_orders={"Deck": ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown (No Cabin Data)"]},
                color_discrete_map=COLOR_MAP,
                template="plotly_white"
            )
            fig_deck.update_layout(yaxis_title="Proportional Ratio (%)")
            st.plotly_chart(fig_deck, use_container_width=True)

    # ---- TAB 2: MACHINE LEARNING SIMULATOR ----
    with tab2:
        st.subheader("🤖 Live Interactive Machine Learning Profile Simulator")
        st.markdown("Adjust hypothetical attributes below to have the backend Random Forest model run a custom inference calculation.")
        
        sim_col1, sim_col2 = st.columns([1, 1])
        
        with sim_col1:
            st.markdown("##### Define Hypothetical Profile Structure:")
            sim_class = st.selectbox("Select Class Placement:", options=[1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else f"{x}nd Class" if x==2 else f"{x}rd Class")
            sim_gender = st.radio("Select Gender Profile:", options=["female", "male"], horizontal=True)
            sim_age = st.slider("Select Simulation Age:", min_value=1, max_value=80, value=25)
            sim_fare = st.number_input("Set Simulation Ticket Fare ($):", min_value=0.0, max_value=600.0, value=30.0)
            sim_fam = st.slider("Accompanied Family Count:", min_value=0, max_value=10, value=0)
            sim_port = st.selectbox("Port of Boarding:", options=["S", "C", "Q"], format_func=lambda x: "Southampton" if x=="S" else "Cherbourg" if x=="C" else "Queenstown")
            
            # Map values for model compliance
            gender_binary = 1 if sim_gender == "female" else 0
            port_binary = 0 if sim_port == "S" else 1 if sim_port == "C" else 2
            
            # Form single-line vector row
            input_vector = pd.DataFrame([[sim_class, gender_binary, sim_age, sim_fare, sim_fam, port_binary]], columns=ml_features)
            
            # Prediction processing
            prediction_prob = ml_model.predict_proba(input_vector)[0][1] * 100
            prediction_label = "SURVIVED" if prediction_prob >= 50 else "DIED"
            
        with sim_col2:
            st.markdown("##### AI Inference Calculations & Metric Importance:")
            
            if prediction_label == "SURVIVED":
                st.success(f"### Predicted Outcome: {prediction_label}\n#### Model Confidence Score: {prediction_prob:.1f}% Probability")
            else:
                st.error(f"### Predicted Outcome: {prediction_label}\n#### Model Risk Score: {100 - prediction_prob:.1f}% Fatality Probability")
                
            st.markdown("---")
            st.markdown("##### Global Model Drivers (What the AI prioritized):")
            
            importances = pd.DataFrame({
                'Feature': ['Ticket Class', 'Gender Flag', 'Age Bracket', 'Fare Value', 'Family Layout Size', 'Port Entry'],
                'Importance Score': ml_model.feature_importances_
            }).sort_values(by='Importance Score', ascending=True)
            
            fig_importance = px.bar(
                importances,
                x='Importance Score',
                y='Feature',
                orientation='h',
                template="plotly_white",
                color_discrete_sequence=["#007bff"]
            )
            st.plotly_chart(fig_importance, use_container_width=True)

    # ---- TAB 3: DATA MATRIX INFRASTRUCTURE ----
    with tab3:
        st.subheader("📋 Filtered Spreadsheet View")
        st.dataframe(
            filtered_df.drop(columns=['Survival Status', 'is_survived']),
            use_container_width=True
        )