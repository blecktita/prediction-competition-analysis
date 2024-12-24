"""
A Streamlit-based web application for analyzing competition categories.
This module provides an interactive interface for predicting Amazon's presence
in different market categories using machine learning models.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import subprocess
import time
import requests
import sys
import os
import json

from typing import Dict
from test_scenarios import TestCases

# Configuring page layout:
st.set_page_config(
    page_title="Competition Category Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #232F3E;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
    }
    
    /* Metrics */
    .metric-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .stCard {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #FF9900;
        color: white;
        font-weight: 500;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF8800;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #232F3E;
    }
    
    /* Input fields */
    .stNumberInput input {
        border-radius: 5px;
    }
    
    .stSlider {
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class AmazonCategoryApp:
    def __init__(self):
        self.service_url = "http://localhost:9696/predict"
        self.service_process = None

    def start_prediction_service(self):
        """Starting prediction service"""
        try:
            with st.spinner("Starting prediction service..."):
                # Check if model file exists
                if not os.path.exists('model_v1.bin'):
                    st.error("📛 Model file (model_v1.bin) not found!")
                    return False

                # Start the Flask service
                self.service_process = subprocess.Popen(
                    [sys.executable, "prediction_service.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Wait for service to start
                max_retries = 5
                for i in range(max_retries):
                    try:
                        # Test if service is responding
                        requests.get(f"http://localhost:9696/")
                        st.success("✅ Prediction service started successfully!")
                        return True
                    except requests.exceptions.ConnectionError:
                        if i < max_retries - 1:
                            time.sleep(2)
                        else:
                            st.error("❌ Failed to start prediction service!")
                            return False

        except Exception as e:
            st.error(f"❌ Error starting service: {str(e)}")
            return False

    def stop_prediction_service(self):
        """Stop the Flask prediction service"""
        if self.service_process:
            self.service_process.terminate()
            self.service_process = None

    def make_prediction(self, data: Dict) -> Dict:
        """Make a prediction request to the service"""
        try:
            response = requests.post(self.service_url, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Prediction request failed: {str(e)}")
            return None

    def create_gauge_chart(self, value: float, title: str) -> go.Figure:
        """Create a gauge chart for probability visualization"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            title={'text': title},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF9900"},
                'steps': [
                    {'range': [0, 30], 'color': "#FFE5CC"},
                    {'range': [30, 70], 'color': "#FFCC99"},
                    {'range': [70, 100], 'color': "#FFB366"}
                ]
            }
        ))
        fig.update_layout(height=250)
        return fig

    def show_prediction_results(self, result: Dict, input_data: Dict = None):
        """Display enhanced prediction results"""
        st.markdown("### 📊 Analysis Results")

        cols = st.columns(3)

        # Probability Gauge
        with cols[0]:
            fig = self.create_gauge_chart(
                result['amazon_probability'],
                "Amazon Presence Probability"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Prediction Status
        with cols[1]:
            prediction = "Amazon Present" if result['amazon_guess'] else "Amazon-free"
            color = "#32CD32" if result['amazon_guess'] else "#FF4500"
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: white; 
                border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: {color};'>{prediction}</h3>
                </div>
            """, unsafe_allow_html=True)

        # Confidence Score
        with cols[2]:
            confidence = abs(result['amazon_probability'] - 0.5) * 2
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: white; 
                border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3>Confidence: {confidence:.1%}</h3>
                </div>
            """, unsafe_allow_html=True)

        # Only show insights if we have input data
        if input_data:
            st.markdown("### 🔍 Key Insights")
            insights = self.generate_insights(input_data)
            for insight in insights:
                st.info(insight)

    def generate_insights(self, data: Dict) -> list:
        """Generate insights based on input data"""
        insights = []

        if data['item_count'] > 3000:
            insights.append("📈 Large market size indicates significant potential")
        elif data['item_count'] < 500:
            insights.append("🎯 Niche market with specialized potential")

        if data['rating_mean'] > 4.0:
            insights.append("⭐ High average ratings suggest strong market quality")
        elif data['rating_mean'] < 3.5:
            insights.append("⚠️ Lower ratings indicate room for improvement")

        if data['vol_purchase_total'] > 1000000:
            insights.append("💰 High purchase volume shows strong market demand")

        if data['best_seller_count'] > 20:
            insights.append("🏆 Multiple bestsellers indicate market maturity")

        return insights

    def show_manual_input(self):
        """Enhanced manual input interface"""
        st.markdown("## 📝 Manual Category Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Market Metrics")
            item_count = st.number_input(
                "Number of Items",
                min_value=0,
                value=1000,
                help="Total number of products in the category"
            )

            vol_purchase_total = st.number_input(
                "Total Purchase Volume ($)",
                min_value=0.0,
                value=100000.0,
                help="Total monetary value of purchases"
            )

            best_seller_count = st.number_input(
                "Number of Bestsellers",
                min_value=0,
                value=10,
                help="Count of bestselling products"
            )

        with col2:
            st.markdown("### ⭐ Rating Metrics")
            rating_mean = st.slider(
                "Average Rating",
                1.0, 5.0, 4.0,
                help="Mean product rating"
            )

            rating_std = st.slider(
                "Rating Standard Deviation",
                0.0, 2.0, 0.8,
                help="Variation in product ratings"
            )

            high_rating_perc = st.slider(
                "High Rating Percentage",
                0.0, 1.0, 0.5,
                help="Proportion of highly-rated products"
            )

        if st.button("🔍 Analyze Category", use_container_width=True):
            data = {
                "item_count": item_count,
                "vol_purchase_total": vol_purchase_total,
                "high_rating_perc": high_rating_perc,
                "rating_mean": rating_mean,
                "rating_std": rating_std,
                "best_seller_count": best_seller_count
            }

            with st.spinner("Analyzing category data..."):
                result = self.make_prediction(data)
                if result:
                    self.show_prediction_results(result, data)

    def show_test_scenarios(self):
        """Enhanced test scenarios interface"""
        st.markdown("## 🧪 Test Scenarios")

        scenarios = TestCases.get_test_scenarios()
        selected_scenario = st.selectbox(
            "Select a Test Scenario",
            options=range(len(scenarios)),
            format_func=lambda x: scenarios[x].name
        )

        scenario = scenarios[selected_scenario]

        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
        """, unsafe_allow_html=True)
        st.write("**Description:**", scenario.description)
        st.write("**Expected Behavior:**", scenario.expected_behavior)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Run Scenario Analysis", use_container_width=True):
            with st.spinner("Analyzing scenario..."):
                result = self.make_prediction(scenario.data)
                if result:
                    self.show_prediction_results(result, scenario.data)

                    st.markdown("### 🎯 Expected vs Actual")
                    st.info(scenario.expected_behavior)

    def show_batch_analysis(self):
        """Enhanced batch analysis interface"""
        st.markdown("## 📈 Batch Category Analysis")

        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                Upload a CSV file containing multiple categories to analyze them all at once.
                The CSV should include all required metrics for each category.
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload CSV with category data",
            type=['csv']
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                if st.button("🔍 Analyze All Categories", use_container_width=True):
                    results = []
                    progress_bar = st.progress(0)

                    for idx, row in df.iterrows():
                        data = row.to_dict()
                        result = self.make_prediction(data)
                        if result:
                            results.append({
                                **data,
                                'amazon_probability': result['amazon_probability'],
                                'amazon_present': result['amazon_guess']
                            })
                        progress_bar.progress((idx + 1) / len(df))

                    results_df = pd.DataFrame(results)
                    st.success(f"✅ Analysis complete for {len(results)} categories!")

                    st.markdown("### Results Overview")
                    st.dataframe(results_df)

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "amazon_category_analysis.csv",
                        "text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    def run(self):
        """Run the Streamlit application"""
        st.markdown("""
            <h1>
                <img src='https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg' 
                     style='height: 40px; margin-right: 10px;'>
                Category Prediction Analysis
            </h1>
        """, unsafe_allow_html=True)

        # Sidebar navigation
        st.sidebar.markdown("## 🎯 Navigation")
        page = st.sidebar.selectbox(
            "Choose a Mode",
            ["📊 Manual Input", "🧪 Test Scenarios", "📈 Batch Analysis"]
        )

        # Start the prediction service if not running
        if not hasattr(st.session_state, 'service_running'):
            st.session_state.service_running = False

        if not st.session_state.service_running:
            if self.start_prediction_service():
                st.session_state.service_running = True
                st.sidebar.success("✅ Prediction service running")
            else:
                st.sidebar.error("❌ Prediction service not running")
                return


        if page == "📊 Manual Input":
            self.show_manual_input()
        elif page == "🧪 Test Scenarios":
            self.show_test_scenarios()
        else:
            self.show_batch_analysis()

def main():
    app = AmazonCategoryApp()
    app.run()

if __name__ == "__main__":
    main()
