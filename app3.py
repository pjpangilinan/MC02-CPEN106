import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import KNNImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM

st.set_page_config(page_title="Taal Lake Water Quality Dashboard", layout="wide")

st.image("lawatch.png", use_container_width=True)

st.markdown(f"""
<style>
.stTabs [data-baseweb="tab-panel"] h2 {{
    color: #003366;
    font-size: 26px;
    font-weight: bold; /* Make text bold */
    margin-top: 0px;
    margin-bottom: 20px; /* Increase space below the header */
    font-family: 'Arial', sans-serif; /* Clean and modern font */
    letter-spacing: 1px; /* Slight letter spacing */
    text-transform: uppercase; /* Capitalize for emphasis */
}}

/* Style for tabs on hover */
.stTabs [data-baseweb="tab"]:hover {{
    background-color: #126f39;
    color: #FFFFFF; /* Change text to white for better contrast */
    font-weight: bold; /* Bold text on hover */
    font-family: 'Arial', sans-serif;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3); /* Subtle shadow for better text contrast */
}}

/* Style for the tab list container */
.stTabs [data-baseweb="tab-list"] {{
    gap: 30px; /* Increase space between tabs */
    justify-content: center; /* Center the tabs */
}}

/* Style for individual tab headers */
.stTabs [data-baseweb="tab"] {{
    height: 50px; /* Increase height for tabs */
    white-space: pre-wrap; /* Allow wrapping if text is long */
    border-radius: 6px 6px 0px 0px; /* Rounded top corners */
    padding: 10px 20px; /* Add padding inside each tab */
    font-weight: bold; /* Make tab text bold */
    font-family: 'Arial', sans-serif; /* Clean and modern font */
    text-align: center; /* Center align the text */
    letter-spacing: 1px; /* Slight letter spacing */
    text-transform: uppercase; /* Capitalize text */
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2); /* Subtle text shadow */
    transition: all 0.3s ease; /* Smooth transition effect */
}}

/* Style for the button inside the tab list */
.stTabs [data-baseweb="tab-list"] button {{
    padding: 15px 30px; /* Adjust padding for a more spacious look */
}}

/* Highlight style for the selected tab */
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: green;
    height: 4px; /* Slightly thicker highlight */
    border-radius: 2px; /* Rounded corners for the highlight */
    color: dark-green;
}}

/* Style for selected tab */
.stTabs [aria-selected="true"] {{
    background-color: #5cc37f; /* Darker blue for selected tab */
    color: #FFFFFF !important; /* White text when selected */
    font-weight: bold; /* Bold text for selected tab */
    height: 55px; /* Slightly taller for emphasis */
}}
</style>
""", unsafe_allow_html=True) # Render the CSS


# Load data
df = pd.read_csv("Spreadmeat_WQxVAxMF.csv")

# Setup selections
sites = ['All Sites'] + list(df["Site"].unique())
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
years = list(range(2013, 2024))
feature_options = ["Water Parameters Only", "Water + External Factors"]
water_parameters = ['Surface Temp', 'Middle Temp', 'Bottom Temp', 'pH', 'Dissolved Oxygen', 'Nitrite', 'Nitrate', 'Ammonia', 'Phosphate']
categorical = ['Wind Direction', 'Weather Condition']
external_params = ['Sulfide', 'Carbon Dioxide', 'Air Temperature']

# Clean data
null_counts_per_row = df.isnull().sum(axis=1)
df_new = df[null_counts_per_row < 10].copy()
df_new.drop_duplicates(inplace=True)

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
float_cols = df_new.select_dtypes(include=['float']).columns
df_new[float_cols] = imputer.fit_transform(df_new[float_cols])

# Process date columns
df_new['Month'] = df_new['Month'].astype(str).str.capitalize()
df_new['Month'] = pd.to_datetime(df_new['Month'], format='%B').dt.month
df_new['Date'] = pd.to_datetime(df_new[['Year', 'Month']].assign(DAY=1))
df_new = df_new.drop(columns=['Month', 'Year'])

# Move 'Date' to second column
cols = list(df_new.columns)
cols.insert(1, cols.pop(cols.index('Date')))
df_new = df_new[cols]

# Save unscaled version for EDA
df_eda = df_new.copy()

# Normalize numeric columns
scaler = MinMaxScaler()
numeric_cols = df_new.select_dtypes(include=['number']).columns
df_new[numeric_cols] = scaler.fit_transform(df_new[numeric_cols])

df_scaled = df_new.copy()

# Main UI
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Introduction",
    "Exploratory Data Analysis", 
    "Prediction & Model Comparison", 
    "Prediction & WQI Calculation",
    "Recommendations"
])


with tab1:
    col1, col2 = st.columns([1.5, 1.2])

    # -------- LEFT COLUMN: INTRODUCTION --------
    with col1:
        st.markdown("<h3 style='border-radius: 10px; margin-top: -10px; background-color: green; color: white; padding: 20px;'>Data Mining and Data Visualization for Water Quality Prediction in Taal Lake</h3>", unsafe_allow_html=True)

        st.markdown("<h3>🎯 Objectives</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: justify;'>This laboratory aims to apply data mining techniques to extract actionable insights from environmental datasets. 
        It focuses on developing predictive models for water quality in Taal Lake using machine learning, 
        particularly ensemble learning methods such as CNN, LSTM, and Hybrid CNN-LSTM.</p>

        <ul>
        <li>Apply data mining techniques to extract useful insights from environmental datasets.</li>
        <li>Develop predictive models for water quality using machine learning.</li>
        <li>Compare ensemble learning techniques such as CNN, LSTM, and Hybrid CNN-LSTM.</li>
        <li>Visualize trends and patterns in water quality parameters.</li>
        <li>Interpret the impact of environmental and volcanic activity on water quality.</li>
        <li>Predict Water Quality Index (WQI) and Water Pollutant Levels with actionable insights for environmental monitoring and intervention.</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("<h3>🛠️ Materials & Tools</h3>", unsafe_allow_html=True)

        st.markdown("""
        <p style='text-align: justify;'>The analysis was conducted using Python in Google Colab.
        Essential libraries include Pandas and NumPy for data processing, Matplotlib and Seaborn for visualization,
        Scikit-learn for traditional machine learning, TensorFlow/Keras for deep learning models,
        and Streamlit for building an interactive dashboard.</p>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style='text-align: justify;'>The dataset used in this study consists of Taal Lake water quality reports from 2013 to 2023, 
        provided by the Bureau of Fisheries and Aquatic Resources (BFAR). It includes both environmental 
        and volcanic activity indicators.</p>
        <ul>
            <li><p><em>Weather Factors:</em> Weather condition, wind direction, and air temperature.<br></li>
            <li><em>Water Quality Parameters:</em> Surface, middle, and bottom water temperatures, pH, dissolved oxygen, 
        nitrogen (as nitrite or nitrate), ammonia, and phosphate.<br></li>
            <li><em>Volcanic Activity Indicators:</em> Sulfide and carbon dioxide concentrations.</p></li>
        </ul>""", unsafe_allow_html=True)

    with col2:
        st.image("lake.jpg")

        st.markdown(
            "<p style='text-align: center; color: lightgray; font-size: 16px;'>Taal Lake <br> https://shoestringdiary.wordpress.com/2024/08/14/golden-hour-to-sunset-at-taal-lake/</p>",
            unsafe_allow_html=True
        )

        st.image("mmber.jpg")

        st.markdown("""<p><strong>Group Members, from left to right:</strong></p>
            <ul>
                <li>Papa, Mark Jamir C.</li>
                <li>Vidad, Ranjo B.</li>
                <li>Mojica, Warreon Dave A.</li>
                <li>Dizon, Rockwell E.</li>
                <li>Pangilinan, Patrick James A.</li>
                <li>Lineses, Dann B.</li>
            </ul>
        """, unsafe_allow_html=True)


with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.write(df_eda.head())

        st.subheader("Summary Statistics")
        st.write(df_eda.describe())

        # Full width for line charts
        st.markdown("""
            <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; max-width: 1000px; justify-content: center; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0;">Line Charts for Time-Series Data</h3>
            </div>
        """, unsafe_allow_html=True)
        df_plot = df_eda.copy()
        df_plot.set_index('Date', inplace=True)

        numeric_columns = df_plot.select_dtypes(include=np.number).columns.tolist()

        # Streamlit dropdown for parameter selection
        selected_param = st.selectbox("Select Parameter to Plot:", numeric_columns)

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        color = random.choice(['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'gray'])
        ax.plot(df_plot.index, df_plot[selected_param], color=color)
        ax.set_xlabel("Date")
        ax.set_ylabel(selected_param)
        ax.set_title(f"Time Series Plot of {selected_param}")
        ax.set_xticks(pd.to_datetime([f"{year}-01-01" for year in range(2013, 2025, 2)]))
        ax.set_xticklabels([str(year) for year in range(2013, 2025, 2)], rotation=45)
        ax.grid(True)

        # Show plot in Streamlit
        st.pyplot(fig)
    with col2:
        st.subheader("Correlation Heatmap")
        df_filtered = df_eda.select_dtypes(include=np.number)
        correlation_matrix = df_filtered.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
        st.pyplot(fig_corr)

        with st.expander("📘 Correlation Analysis Summary", expanded=False):
            st.markdown("""
            <ul>
            <li>The analysis shows that most parameters have low negative correlations, while a few show strong relationships.</li>
            <li><strong>Water Temperatures</strong> strongly correlate with <strong>Air Temperature</strong>, indicating the impact of atmospheric conditions, especially on surface waters.</li>
            <li><strong>Ammonia</strong> and <strong>Phosphate</strong> levels rise together, likely due to pollution or biological processes.</li>
            <li><strong>pH</strong>, <strong>Dissolved Oxygen</strong>, <strong>Phosphate</strong>, and <strong>CO₂</strong> show declining trends over time, possibly linked to acidification, warming, or reduced biological activity.</li>
            <li>Moderate correlations also exist between:
                <ul>
                <li><strong>pH</strong> & <strong>Dissolved Oxygen</strong></li>
                <li><strong>Sulfide</strong> & <strong>Ammonia</strong></li>
                <li><strong>CO₂</strong> & <strong>Water Temperatures</strong> / <strong>Dissolved Oxygen</strong></li>
                </ul>
            </li>
            </ul>
            """, unsafe_allow_html=True)

        df_eda['Weather Condition'] = df_eda['Weather Condition'].astype(str)
        sorted_conditions = sorted(df_eda['Weather Condition'].unique())
        df_eda['Weather Condition'] = pd.Categorical(df_eda['Weather Condition'], categories=sorted_conditions, ordered=True)

        # Streamlit header
        st.markdown("""
            <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; max-width: 1000px; justify-content: center; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0;">Scatter Plots for Parameter Relationships</h3>
            </div>
        """, unsafe_allow_html=True)

        numeric_columns = df_new.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = ['Weather Condition', 'Wind Direction']

        x_axis = st.selectbox("Select X-axis parameter", categorical_columns + numeric_columns)
        y_axis = st.selectbox("Select Y-axis parameter", numeric_columns)

        fig, ax = plt.subplots(figsize=(10, 6))

        if x_axis in categorical_columns:
            sns.scatterplot(data=df_new, x=x_axis, y=y_axis, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            ax.scatter(df_new[x_axis], df_new[y_axis], alpha=0.7)
            ax.set_xlabel(x_axis)

        ax.set_ylabel(y_axis)
        ax.set_title(f"{x_axis} vs {y_axis}")
        ax.grid(True)

        # Display plot
        st.pyplot(fig)

with tab3:
    st.markdown(
        """
        <div style='background-color: green; padding: 10px; border-radius: 10px;'>
            <h1 style='font-size: 2.3rem; color: white; text-align: center;'>Water Quality Prediction & Model Comparison Dashboard for Taal Lake</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Sidebar Filters
    st.sidebar.header("Selection Filters for Water Quality Prediction")
    selected_site = st.sidebar.selectbox("Select Site", sites)

    date_range = st.sidebar.select_slider(
        "Select Date Range",
        options=pd.date_range(start="2013-01-01", end="2023-12-01", freq="MS").strftime('%Y-%m').tolist(),
        value=("2013-01", "2023-12")
    )

    # Convert date range to datetime
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    selected_features = st.sidebar.selectbox("Select Feature Set", feature_options)
    selected_parameter = st.sidebar.selectbox("Select Water Parameter to Predict", water_parameters)

    with st.expander("Selected Options", expanded=True):
        st.markdown(f"""
        - **Site**: {selected_site}
        - **Date Range**: {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}
        - **Feature Set**: {selected_features}
        - **Selected Water Parameter to Predict**: {selected_parameter}
        """)

    df_filtered = df_scaled.copy()

    # Apply site filter
    if selected_site != "All Sites":
        df_filtered = df_filtered[df_filtered["Site"] == selected_site]

    # Apply date filter
    df_filtered = df_filtered[(df_filtered["Date"] >= start_date) & (df_filtered["Date"] <= end_date)]

    if df_filtered.shape[0] < 10:
        st.error("Not enough data for the selected filters. Please choose another site or expand the date range.")
        st.stop()

    # Feature processing
    if selected_features == "Water Parameters Only":
        X = df_filtered[water_parameters]
    else:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cat_data = encoder.fit_transform(df_filtered[categorical])
        encoded_cat_cols = encoder.get_feature_names_out(categorical)
        encoded_cat_df = pd.DataFrame(encoded_cat_data, columns=encoded_cat_cols, index=df_filtered.index)

        selected_features = water_parameters + external_params
        X = pd.concat([df_filtered[selected_features], encoded_cat_df], axis=1)

    y = df_filtered[selected_parameter]
    X = np.array(X).reshape(X.shape[0], X.shape[1], 1)

    # Train-valid-test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Model builder
    def build_model(model_type, input_shape):
        model = Sequential()
        if model_type == "CNN":
            model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
        elif model_type == "LSTM":
            model.add(LSTM(64, input_shape=input_shape))
        elif model_type == "CNN-LSTM":
            model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(64))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def plot_actual_vs_predicted(y_test, y_pred, title):
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(title)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        st.pyplot(plt)

    def train_and_evaluate(model_type, X_train, X_valid, X_test, y_train, y_valid, y_test, selected_param):
        st.markdown(f"""
            <div style="background-color: green; border-radius: 10px; color: white; text-align: center; width: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 10px;">
                <h1 style="color: white; padding: 0; margin: 0;">{model_type} Model</h1>
                <h3 style="color: white; padding: 0; margin: 0;">Predicting {selected_param}</h3>
            </div>
        """, unsafe_allow_html=True)
        model = build_model(model_type, (X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.subheader("🔎 Model Evaluation")
        st.write(f"**MAE:** {mae:.3f}")
        st.write(f"**RMSE:** {rmse:.3f}")

        y_pred_flattened = y_pred.flatten()
        st.subheader("Sample Predictions")
        pred_df = pd.DataFrame({
            "Actual": y_test.values[:10],
            "Predicted": y_pred.flatten()[:10]
        })
        st.write(pred_df)

        plot_actual_vs_predicted(y_test, y_pred_flattened, f'{model_type} Model')

        return model

    col1, col2, col3 = st.columns(3)

    with col1:
        train_and_evaluate("CNN", X_train, X_valid, X_test, y_train, y_valid, y_test, selected_parameter)
    with col2:
        train_and_evaluate("LSTM", X_train, X_valid, X_test, y_train, y_valid, y_test, selected_parameter)
    with col3:
        train_and_evaluate("CNN-LSTM", X_train, X_valid, X_test, y_train, y_valid, y_test, selected_parameter)

with tab4:
    df_unstandardize = df.copy()

    def build_model(model_type, input_shape, output_dim):
        model = Sequential()
        if model_type == "CNN":
            model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
        elif model_type == "LSTM":
            model.add(LSTM(64, input_shape=input_shape))
        elif model_type == "CNN-LSTM":
            model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(64))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_monthly_data(df, site, target_vars, look_back=12):
        site_data = df[df['Site'] == site].copy()
        site_data['Date'] = pd.to_datetime(site_data['Date'])
        site_data.set_index('Date', inplace=True)
        
        # Monthly average
        monthly_data = site_data[target_vars].resample('M').mean().dropna()
        
        if monthly_data.empty:
            st.error(f"No data available for the site '{site}' at the monthly granularity.")
            return None, None, 0, None
        
        scaler.fit(monthly_data)
        scaled_data = scaler.transform(monthly_data)
        
        X, Y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back])
            Y.append(scaled_data[i + look_back])
            
        return np.array(X), np.array(Y), len(X), monthly_data
    
    def prepare_yearly_data(df, site, target_vars, look_back=5):
        site_data = df[df['Site'] == site].copy()
        site_data['Date'] = pd.to_datetime(site_data['Date'])
        site_data.set_index('Date', inplace=True)
        
        # Yearly average
        yearly_data = site_data[target_vars].resample('Y').mean().dropna()
        
        if yearly_data.empty:
            st.error(f"No data available for the site '{site}' at the yearly granularity.")
            return None, None, 0, None
        
        scaler.fit(yearly_data)
        scaled_data = scaler.transform(yearly_data)
        
        X, Y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back])
            Y.append(scaled_data[i + look_back])
            
        return np.array(X), np.array(Y), len(X), yearly_data

    st.markdown(
        """
        <div style='background-color: green; padding: 10px; border-radius: 10px;'>
            <h1 style='font-size: 2.3rem; color: white; text-align: center;'>⏳ Time-Based Prediction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    # UI Inputs
    with col1:
        prediction_type = st.radio("Select Time Granularity", ["Monthly", "Yearly"])
    with col2:
        selected_model = st.selectbox("Choose Model", ["CNN", "LSTM", "CNN-LSTM"])
    with col3:
        location = st.selectbox("Select Site", df_new['Site'].unique())

    target_vars = ['Surface Temp', 'Middle Temp', 'Bottom Temp', 'pH',
                   'Dissolved Oxygen', 'Nitrite', 'Nitrate', 'Ammonia', 'Phosphate']

    look_back = 12 if prediction_type == "Monthly" else 5
    future_periods = 12 if prediction_type == "Monthly" else 5

    # Prepare data
    if prediction_type == "Monthly":
        X, Y, count, data_used = prepare_monthly_data(df_new, location, target_vars)
    else:
        X, Y, count, data_used = prepare_yearly_data(df_new, location, target_vars, look_back)

    if X is None or Y is None:
        st.error("Data preparation failed due to insufficient data. Please check your data and site selection.")
    else:
        st.write(f"Total usable records: {count}")
        num_features = X.shape[2]

        # Build and train model
        model = build_model(selected_model, input_shape=(look_back, num_features), output_dim=len(target_vars))
        model.fit(X, Y, epochs=20, batch_size=32, verbose=0)

        # Prediction loop
        last_data = data_used[target_vars].values[-look_back:]
        predictions = []
        for _ in range(future_periods):
            input_data = last_data[-look_back:].reshape(1, look_back, len(target_vars))
            pred = model.predict(input_data, verbose=0)[0]
            predictions.append(pred)
            last_data = np.vstack([last_data, pred])

        predictions_df = pd.DataFrame(predictions, columns=target_vars)

        # Inverse scale predictions
        scaler.fit(df_unstandardize[target_vars])
        descaled_df = pd.DataFrame(scaler.inverse_transform(predictions_df),
                                   columns=target_vars)

        # Display predictions
        st.markdown("""
            <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; max-width: 1400px; justify-content: center; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0;">Predictions (Original Units)</h3>
            </div>
        """, unsafe_allow_html=True)
        st.dataframe(descaled_df.style.format("{:.2f}"))

        st.subheader("Model Performance (on Last Known Data)")

        metrics = []
        for column in predictions_df.columns:
            # Get actual and predicted values for the current column
            actual_values = Y[-future_periods:, predictions_df.columns.get_loc(column)]
            
            # Recreate predictions on the last `future_periods` input sequences
            test_inputs = X[-future_periods:]
            predicted_values = model.predict(test_inputs, verbose=0)[:, predictions_df.columns.get_loc(column)]

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mae = mean_absolute_error(actual_values, predicted_values)

            metrics.append((column, round(rmse, 4), round(mae, 4)))

        metrics_df = pd.DataFrame(metrics, columns=["Variable", "RMSE", "MAE"])
        st.dataframe(metrics_df)

        # --- WQI Calculation Function ---
        def calculate_wqi(df, parameters):
            weights = {
                'pH': 0.15,
                'Dissolved Oxygen': 0.25,
                'Nitrite': 0.10,
                'Nitrate': 0.10,
                'Ammonia': 0.15,
                'Phosphate': 0.10,
                'Surface Temp': 0.05,
                'Middle Temp': 0.05,
                'Bottom Temp': 0.05,
            }

            missing_params = [param for param in parameters if param not in df.columns]
            if missing_params:
                st.error(f"Missing parameters in DataFrame: {missing_params}")
                return pd.Series([None] * len(df))

            weighted_values = df[parameters].apply(lambda x: x * weights[x.name], axis=0)
            return weighted_values.sum(axis=1)

        # --- Classification Function ---
        def classify_water_quality(wqi):
            if wqi >= 90:
                return 'Excellent'
            elif wqi >= 70:
                return 'Good'
            elif wqi >= 50:
                return 'Fair'
            elif wqi >= 25:
                return 'Poor'
            else:
                return 'Very Poor'

        # --- Calculate and Display WQI ---
        st.subheader("Water Quality Index (WQI) and Classification")

        # Parameters used in WQI
        wqi_params = ['pH', 'Dissolved Oxygen', 'Nitrite', 'Nitrate', 'Ammonia',
                    'Phosphate', 'Surface Temp', 'Middle Temp', 'Bottom Temp']

        # Calculate WQI from descaled predictions
        descaled_df['WQI'] = calculate_wqi(descaled_df, wqi_params)
        descaled_df['Water Quality'] = descaled_df['WQI'].apply(classify_water_quality)

        # Format numeric columns before displaying
        def format_numeric_columns(df):
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].apply(lambda x: f"{x:.2f}")
            return df

        # Apply formatting
        formatted_df = format_numeric_columns(descaled_df)

        # Display the formatted dataframe
        st.dataframe(formatted_df)

with tab5:
    st.subheader("Recommendations")

    st.markdown(
        """
        <ul>
          <li>
            <strong>Getting more data from PHILVOCS and PAGASA</strong> – Improve the dataset with environmental and meteorological data to improve prediction accuracy.
          </li>
          <li>
            <strong>Daily Water Quality Prediction</strong> – Have daily prediction for each parameter, for each location.
          </li>
          <li>
            <strong>Adding the Year 2024</strong> – Incorporate the most recent data for continuity and better trend analysis.
          </li>
          <li>
            <strong>Tweaking the Parameters of Machine Learning</strong> – Fine-tune hyperparameters to boost model performance and reduce prediction errors.
          </li>
          <li>
            <strong>Use Latest Predictive Models like Transformer or AutoML</strong> – Include advanced models to enhance sequential learning and automate model optimization.
          </li>
        </ul>
        """,
        unsafe_allow_html=True
    )
