import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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
    font-weight: bold;
    margin-top: 0px;
    margin-bottom: 20px;
    font-family: 'Arial', sans-serif;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* Style for tabs on hover */
.stTabs [data-baseweb="tab"]:hover {{
    background-color: #126f39;
    color: #FFFFFF;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}}

/* Style for the tab list container */
.stTabs [data-baseweb="tab-list"] {{
    gap: 30px;
    justify-content: center;
    width: 100% !important;  /* Ensure full width */
    overflow: visible !important;  /* Prevent clipping/fading */
    flex-wrap: nowrap !important;  /* Prevent line break */
    margin-right: 0 !important;  /* Remove default margins */
}}

/* Style for individual tab headers */
.stTabs [data-baseweb="tab"] {{
    height: 50px;
    flex: 1;
    white-space: pre-wrap;
    border-radius: 6px 6px 0px 0px;
    padding: 10px 20px;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
    text-align: center;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    box-sizing: border-box;  /* Fix layout overflow */
}}

/* Style for the button inside the tab list */
.stTabs [data-baseweb="tab-list"] button {{
    padding: 15px 30px;
}}

/* Highlight style for the selected tab */
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: green;
    height: 4px;
    border-radius: 2px;
    color: dark-green;
}}

/* Style for selected tab */
.stTabs [aria-selected="true"] {{
    background-color: #5cc37f;
    color: #FFFFFF !important;
    font-weight: bold;
    height: 55px;
}}
</style>
""", unsafe_allow_html=True)

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
tab1, tab2 = st.tabs([
    "Exploratory Data Analysis",
    "Water Quality Prediction"
])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Dataset Overview")

        num_rows = st.slider("Select number of rows to view", min_value=5, max_value=100, value=10)

        columns_to_view = st.multiselect(
            "Select columns to display",
            options=df_eda.columns.tolist(),
            default=df_eda.columns.tolist()[:5]  
        )

        st.dataframe(df_eda[columns_to_view].head(num_rows), height=400)

        with st.expander("📊 Summary Statistics", expanded=False):
            st.write(df_eda.describe().T.style.format("{:.2f}"))

        st.markdown("""
            <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; justify-content: center; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0;">Parameter Distribution</h3>
            </div>
        """, unsafe_allow_html=True)
        dist_col = st.selectbox("Select column for Distribution", df_eda.select_dtypes(include=['float64', 'int64']).columns,  index=list(df_eda.select_dtypes(include=['float64', 'int64']).columns).index("pH"), key="dist")
        fig1 = px.histogram(
            df_eda, x=dist_col, nbins=50,
            color_discrete_sequence=['skyblue'], opacity=0.75
        )
        fig1.update_layout(margin=dict(t=20,b=20), bargap=0.1)
        st.plotly_chart(fig1, use_container_width=True)

        # Full width for line charts
        st.markdown("""
            <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; max-width: 1000px; justify-content: center; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0;">Line Charts for Time-Series Data</h3>
            </div>
        """, unsafe_allow_html=True)
        df_plot = df_eda.copy()
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
        df_plot = df_plot.sort_values('Date') 

        numeric_columns = df_plot.select_dtypes(include=np.number).columns.tolist()

        selected_param = st.selectbox("Select Parameter to Plot:", numeric_columns)

        fig_line = px.line(
            df_plot,
            x="Date",
            y=selected_param,
            title=" ",
            line_shape="spline",  
        )

        fig_line.update_layout(
            margin=dict(t=20,b=20),  
            xaxis_title="Date",
            yaxis_title=selected_param,
            title_x=0.5,
            height=500,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.subheader("🔥 Correlation Heatmap")
        df_filtered = df_eda.select_dtypes(include=np.number)
        correlation_matrix = df_filtered.corr()
       
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=".2f",        
            color_continuous_scale="RdBu_r",  
            title=" ",
            aspect="auto"
        )

        fig_corr.update_layout(
            title_x=0.5,
            height=590,
            margin=dict(t=10, b=10)
        )

        # Display in Streamlit
        st.plotly_chart(fig_corr, use_container_width=True)

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

        st.markdown("""
            <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; justify-content: center; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0;">Parameter Box Plot</h3>
            </div>
        """, unsafe_allow_html=True)
        box_col = st.selectbox("Select column for Box Plot", df_eda.select_dtypes(include=['float64', 'int64']).columns, key="box")
        fig_box = px.box(
            df_eda,
            y=box_col,
            title=" ",
            points="outliers",
            color_discrete_sequence=["lightgreen"]
        )

        fig_box.update_layout(
            yaxis_title=box_col,
            title_x=0.5,
            height=450,
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(fig_box, use_container_width=True)
        
        # Streamlit header
        st.markdown("""
            <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; justify-content: center; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0;">Scatter Plots for Parameter Relationships</h3>
            </div>
        """, unsafe_allow_html=True)

        # Select columns for X and Y
        numeric_columns = df_new.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = ['Weather Condition', 'Wind Direction']
        x, y = st.columns(2)

        with x:
            x_axis = st.selectbox(
                "Select X-axis parameter",
                categorical_columns + numeric_columns,
                index=(categorical_columns + numeric_columns).index("pH") 
            )

        with y:
            y_axis = st.selectbox(
                "Select Y-axis parameter",
                numeric_columns,
                index=numeric_columns.index("Dissolved Oxygen")  
        )
            
        fig_scatter = px.strip(df_new, x=x_axis, y=y_axis, title=" ") if x_axis in categorical_columns else px.scatter(
            df_new,
            x=x_axis,
            y=y_axis,
            opacity=0.7,
            title=f" "
        )

        fig_scatter.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            title_x=0.5,
            height=500,
            margin=dict(t=30, b=30)
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("""
        <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; justify-content: center; border-radius: 8px; text-align: center;">
            <h3 style="margin: 0;">Water Quality Index Distribution</h3>
        </div>
    """, unsafe_allow_html=True)

    wqi = pd.read_csv("wqi.csv")

    fig_wqi = px.histogram(
        wqi,
        x='WQI',
        nbins=30,
        title=" ",
        color_discrete_sequence=['teal'],
        opacity=0.75
    )

    fig_wqi.update_layout(
        margin=dict(t=20,b=20),  
        xaxis_title="WQI",
        yaxis_title="Frequency",
        title_x=0.5,
        height=400,
        bargap=0.1
    )

    st.plotly_chart(fig_wqi, use_container_width=True)
    
with tab2:
    mode = st.selectbox("Select Prediction Mode", [
        "Water Quality Prediction & Model Comparison",
        "Time Based Prediction & WQI Calculation"
    ])

    if mode == "Water Quality Prediction & Model Comparison":
        st.markdown(
            """
            <div style='background-color: green; padding: 10px; border-radius: 10px;'>
                <h1 style='font-size: 2.3rem; color: white; text-align: center;'>Water Quality Prediction & Model Comparison</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.container():
            st.markdown("### 🔎 Selection Filters for Water Quality Prediction")
            col1, col2 = st.columns(2)

            with col1:
                selected_site = st.selectbox("Select Site", sites)
                selected_features = st.selectbox("Select Feature Set", feature_options)

            with col2:
                selected_parameter = st.selectbox("Select Water Parameter to Predict", water_parameters)
                date_range = st.select_slider(
                    "Select Date Range",
                    options=pd.date_range(start="2013-01-01", end="2023-12-01", freq="MS").strftime('%Y-%m').tolist(),
                    value=("2013-01", "2023-12")
                )

        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        with st.expander("Selected Options", expanded=True):
            st.markdown(f"""
            - **Site**: {selected_site}
            - **Date Range**: {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}
            - **Feature Set**: {selected_features}
            - **Selected Water Parameter to Predict**: {selected_parameter}
            """)

        df_filtered = df_scaled.copy()

        if selected_site != "All Sites":
            df_filtered = df_filtered[df_filtered["Site"] == selected_site]

        df_filtered = df_filtered[(df_filtered["Date"] >= start_date) & (df_filtered["Date"] <= end_date)]

        if df_filtered.shape[0] < 10:
            st.error("Not enough data for the selected filters. Please choose another site or expand the date range.")
            st.stop()

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

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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

        predictions_df = pd.DataFrame()

        def train_and_evaluate(model_type, X_train, X_valid, X_test, y_train, y_valid, y_test, selected_param):
            st.markdown(f"""
                <div style="background-color: green; border-radius: 10px; color: white; text-align: center; width: 100%; padding: 10px;">
                    <h1 style="color: white;">{model_type} Model</h1>
                    <h3 style="color: white;">Predicting {selected_param}</h3>
                </div>
            """, unsafe_allow_html=True)
            model = build_model(model_type, (X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
            y_pred = model.predict(X_test).flatten()
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.subheader("🔍 Model Evaluation")
            st.write(f"**MAE:** {mae:.3f}")
            st.write(f"**RMSE:** {rmse:.3f}")

            sample_preds = pd.DataFrame({
                "Actual": y_test.values[:10],
                "Predicted": y_pred[:10]
            })

            st.subheader("Sample Predictions")
            st.write(sample_preds)

            plot_actual_vs_predicted(y_test, y_pred, f'{model_type} Model')

            return y_pred

        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: green;
                    color: white;
                    width: 100%;
                    padding: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                div.stButton > button:first-child:hover {
                    background-color: darkgreen;
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)

        if st.button("Start Prediction"):
            col1, col2, col3 = st.columns(3)

            with col1:
                train_and_evaluate("CNN", X_train, X_valid, X_test, y_train, y_valid, y_test, selected_parameter)
            with col2:
                train_and_evaluate("LSTM", X_train, X_valid, X_test, y_train, y_valid, y_test, selected_parameter)
            with col3:
                train_and_evaluate("CNN-LSTM", X_train, X_valid, X_test, y_train, y_valid, y_test, selected_parameter)

    elif mode == "Time Based Prediction & WQI Calculation":
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

            st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: green;
                    color: white;
                    width: 100%;
                    padding: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                div.stButton > button:first-child:hover {
                    background-color: darkgreen;
                    color: white;
                }
            </style>
            """, unsafe_allow_html=True)

            if st.button("Start Prediction"):
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

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("""
                        <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; justify-content: center; border-radius: 8px; text-align: center;">
                            <h3 style="margin: 0;">Predictions (Original Units)</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(descaled_df.style.format("{:.2f}"))

                with c2:
                    st.markdown("""
                        <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; justify-content: center; border-radius: 8px; text-align: center;">
                            <h3 style="margin: 0;">Model Performance (on Last Known Data)</h3>
                        </div>
                    """, unsafe_allow_html=True)

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
                    metrics_long = metrics_df.melt(id_vars='Variable', value_vars=['RMSE', 'MAE'],
                                    var_name='Metric', value_name='Value')

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=metrics_long, x='Variable', y='Value', hue='Metric', ax=ax)

                    ax.set_title("Model Performance Metrics by Variable")
                    ax.set_ylabel("Error Value")
                    ax.set_xlabel("Variable")
                    ax.legend(title="Metric")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    for p in ax.patches:
                        height = p.get_height()
                        if height > 0.01:  
                            ax.annotate(f'{height:.3f}',
                                        (p.get_x() + p.get_width() / 2, height),
                                        ha='center', va='bottom',
                                        fontsize=9, color='black',
                                        xytext=(0, 3), textcoords='offset points')

                    st.pyplot(fig)

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

                st.markdown("""
                    <div style="background-color: green; color: white; padding: 0; margin-bottom: 10px; justify-content: center; border-radius: 8px; text-align: center;">
                        <h3 style="margin: 0;">Water Quality Index & Classification for Predictions</h3>
                    </div>
                """, unsafe_allow_html=True)

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

                w1, w2 = st.columns([1.1, 0.9])
                with w1:
                    st.dataframe(formatted_df[['WQI', 'Water Quality']], height=500)
                with w2:
                    st.markdown("""
                        <div style="background-color: white; color: black; padding: 0; margin-bottom: 10px; justify-content: center; border-radius: 8px; text-align: center;">
                            <h3 style="margin: 0;">Recommendations for Very Poor Water Quality</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Monitor Regularly"):
                        st.write("Check water quality often, especially for pH, oxygen, temperature, and pollutants from the volcano.")

                    with st.expander("Improve Wastewater Treatment"):
                        st.write("Make sure nearby homes and factories clean their water before releasing it into the lake.")

                    with st.expander("Plant Vegetation"):
                        st.write("Add trees and plants along the lake’s edge to stop erosion and filter dirty water.")

                    with st.expander("Add Helpful Fish"):
                        st.write("Protect or introduce native fish that eat algae to keep the water balanced.")

                    with st.expander("Use Aerators"):
                        st.write("Add machines that increase oxygen in the water to stop harmful algae.")

                    with st.expander("Educate the Community"):
                        st.write("Teach locals how to protect the lake and why it matters.")

                    with st.expander("Enforce Rules"):
                        st.write("Make sure pollution and land-use laws are followed around the lake.")
