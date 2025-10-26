"""
SmartFarm AI Dashboard - Interactive Interface for Farmers
Integrates all AI modules: Disease Detection, Yield Prediction, Smart Irrigation, Market Intelligence
Addresses UN SDG 2: Zero Hunger through accessible technology
"""

import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
from PIL import Image
import io
import base64
from datetime import datetime, timedelta
import json

# Import our AI modules (would be actual imports in production)
# from crop_disease_detection import CropDiseaseDetector
# from yield_prediction import YieldPredictionSystem
# from smart_irrigation import SmartIrrigationController
# from market_intelligence import MarketIntelligenceSystem

# Configure Streamlit page
st.set_page_config(
    page_title="SmartFarm AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class SmartFarmDashboard:
    """
    Main dashboard class that integrates all AI modules
    """
    
    def __init__(self):
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'farm_data' not in st.session_state:
            st.session_state.farm_data = self.get_default_farm_data()
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'irrigation_history' not in st.session_state:
            st.session_state.irrigation_history = self.generate_irrigation_history()
        if 'yield_predictions' not in st.session_state:
            st.session_state.yield_predictions = self.generate_yield_data()
        if 'market_data' not in st.session_state:
            st.session_state.market_data = self.generate_market_data()
    
    def get_default_farm_data(self):
        """Get default farm configuration"""
        return {
            'farm_name': 'Green Valley Farm',
            'location': 'Punjab, India',
            'total_area': 50,  # hectares
            'crop_type': 'Rice',
            'planting_date': '2024-06-15',
            'expected_harvest': '2024-11-30',
            'soil_type': 'Clay loam',
            'irrigation_system': 'Drip irrigation'
        }
    
    def generate_irrigation_history(self):
        """Generate sample irrigation history data"""
        dates = pd.date_range(start='2024-06-15', end='2024-10-25', freq='D')
        np.random.seed(42)
        
        data = []
        for i, date in enumerate(dates):
            soil_moisture = 30 + 20 * np.sin(i/10) + np.random.normal(0, 5)
            irrigation = max(0, np.random.normal(8, 3)) if soil_moisture < 45 else max(0, np.random.normal(3, 2))
            
            data.append({
                'date': date,
                'soil_moisture': np.clip(soil_moisture, 15, 75),
                'irrigation_mm': irrigation,
                'rainfall_mm': max(0, np.random.exponential(2)),
                'temperature': 25 + 5 * np.sin(i/15) + np.random.normal(0, 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_yield_data(self):
        """Generate sample yield prediction data"""
        return {
            'current_prediction': 7.8,
            'confidence_interval': (7.2, 8.4),
            'historical_average': 7.2,
            'factors': {
                'Weather Conditions': 85,
                'Soil Health': 78,
                'Irrigation Management': 92,
                'Pest Management': 88,
                'Nutrient Management': 82
            }
        }
    
    def generate_market_data(self):
        """Generate sample market data"""
        dates = pd.date_range(start='2024-01-01', end='2024-10-25', freq='D')
        base_price = 320
        prices = [base_price]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0.001, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return {
            'prices': pd.DataFrame({
                'date': dates,
                'price_usd_per_ton': prices
            }),
            'current_price': prices[-1],
            'weekly_change': ((prices[-1] - prices[-7]) / prices[-7]) * 100,
            'market_sentiment': 'Positive',
            'recommendations': [
                'Market showing upward trend',
                'Consider holding for better prices',
                'Strong export demand expected'
            ]
        }
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🌱 SmartFarm AI Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3>AI-Powered Agriculture for UN SDG 2: Zero Hunger</h3>
            <p>Real-time insights for crop management, irrigation optimization, and market intelligence</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with farm configuration"""
        st.sidebar.title("🏡 Farm Configuration")
        
        farm_data = st.session_state.farm_data
        
        # Farm basic info
        st.sidebar.subheader("Basic Information")
        farm_data['farm_name'] = st.sidebar.text_input("Farm Name", farm_data['farm_name'])
        farm_data['location'] = st.sidebar.text_input("Location", farm_data['location'])
        farm_data['total_area'] = st.sidebar.number_input("Total Area (hectares)", value=farm_data['total_area'], min_value=1)
        
        # Crop information
        st.sidebar.subheader("Crop Information")
        farm_data['crop_type'] = st.sidebar.selectbox("Crop Type", 
            ['Rice', 'Wheat', 'Corn', 'Soybean', 'Cotton'], 
            index=['Rice', 'Wheat', 'Corn', 'Soybean', 'Cotton'].index(farm_data['crop_type']))
        
        # Current conditions
        st.sidebar.subheader("Current Conditions")
        current_soil_moisture = st.sidebar.slider("Soil Moisture (%)", 0, 100, 45)
        current_temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 28)
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("🚨 Generate Alert", type="primary"):
            st.session_state.alerts.append({
                'time': datetime.now(),
                'type': 'info',
                'message': 'Manual alert generated from dashboard'
            })
            st.rerun()
        
        if st.sidebar.button("🔄 Refresh Data"):
            st.session_state.irrigation_history = self.generate_irrigation_history()
            st.rerun()
        
        return current_soil_moisture, current_temperature
    
    def render_overview_metrics(self):
        """Render key metrics overview"""
        st.subheader("📊 Farm Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Yield Prediction",
                value=f"{st.session_state.yield_predictions['current_prediction']:.1f} tons/ha",
                delta=f"+{st.session_state.yield_predictions['current_prediction'] - st.session_state.yield_predictions['historical_average']:.1f}"
            )
        
        with col2:
            current_price = st.session_state.market_data['current_price']
            weekly_change = st.session_state.market_data['weekly_change']
            st.metric(
                label="Market Price",
                value=f"${current_price:.0f}/ton",
                delta=f"{weekly_change:+.1f}%"
            )
        
        with col3:
            total_irrigation = st.session_state.irrigation_history['irrigation_mm'].sum()
            st.metric(
                label="Water Used (Season)",
                value=f"{total_irrigation:.0f} mm",
                delta="Optimized"
            )
        
        with col4:
            avg_soil_moisture = st.session_state.irrigation_history['soil_moisture'].mean()
            st.metric(
                label="Avg Soil Moisture",
                value=f"{avg_soil_moisture:.1f}%",
                delta="Optimal" if 40 <= avg_soil_moisture <= 60 else "Monitor"
            )
    
    def render_alerts(self):
        """Render alerts and notifications"""
        if st.session_state.alerts:
            st.subheader("🚨 Alerts & Notifications")
            for alert in st.session_state.alerts[-3:]:  # Show last 3 alerts
                alert_class = "alert-warning" if alert['type'] == 'warning' else "alert-success"
                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>{alert['time'].strftime('%H:%M')}</strong> - {alert['message']}
                </div>
                """, unsafe_allow_html=True)
    
    def render_disease_detection(self):
        """Render crop disease detection interface"""
        st.subheader("🔍 Crop Disease Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Upload crop image for disease analysis**")
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Analyze Image"):
                    # Simulate disease detection
                    with st.spinner("Analyzing image..."):
                        # Mock analysis results
                        results = {
                            'disease': 'Bacterial Blight',
                            'confidence': 87.5,
                            'treatment': 'Apply copper-based bactericides. Improve field drainage.',
                            'severity': 'Moderate'
                        }
                    
                    st.success(f"Analysis Complete!")
                    st.write(f"**Detected Disease:** {results['disease']}")
                    st.write(f"**Confidence:** {results['confidence']:.1f}%")
                    st.write(f"**Severity:** {results['severity']}")
                    st.write(f"**Treatment:** {results['treatment']}")
        
        with col2:
            st.write("**Recent Disease Detections**")
            recent_detections = [
                {'date': '2024-10-23', 'disease': 'Leaf Blast', 'confidence': 92.3, 'status': 'Treated'},
                {'date': '2024-10-20', 'disease': 'Brown Spot', 'confidence': 78.9, 'status': 'Monitoring'},
                {'date': '2024-10-18', 'disease': 'Healthy', 'confidence': 95.1, 'status': 'Good'}
            ]
            
            for detection in recent_detections:
                with st.expander(f"{detection['date']} - {detection['disease']}"):
                    st.write(f"Confidence: {detection['confidence']:.1f}%")
                    st.write(f"Status: {detection['status']}")
    
    def render_irrigation_dashboard(self):
        """Render smart irrigation dashboard"""
        st.subheader("💧 Smart Irrigation Control")
        
        # Current recommendation
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Irrigation history chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Soil Moisture & Irrigation', 'Temperature & Rainfall'),
                specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
            )
            
            # Soil moisture and irrigation
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.irrigation_history['date'],
                    y=st.session_state.irrigation_history['soil_moisture'],
                    name='Soil Moisture (%)',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=st.session_state.irrigation_history['date'],
                    y=st.session_state.irrigation_history['irrigation_mm'],
                    name='Irrigation (mm)',
                    marker_color='blue',
                    opacity=0.6
                ),
                row=1, col=1, secondary_y=True
            )
            
            # Temperature and rainfall
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.irrigation_history['date'],
                    y=st.session_state.irrigation_history['temperature'],
                    name='Temperature (°C)',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=st.session_state.irrigation_history['date'],
                    y=st.session_state.irrigation_history['rainfall_mm'],
                    name='Rainfall (mm)',
                    marker_color='lightblue'
                ),
                row=2, col=1, secondary_y=True
            )
            
            fig.update_layout(height=500, title_text="Irrigation Management History")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Today's Recommendation**")
            
            # Mock irrigation recommendation
            recommendation = {
                'amount': 12.5,
                'timing': 'Early morning',
                'reasoning': 'Soil moisture below optimal range',
                'efficiency_score': 85
            }
            
            st.metric("Recommended Irrigation", f"{recommendation['amount']:.1f} mm")
            st.write(f"**Best Timing:** {recommendation['timing']}")
            st.write(f"**Reasoning:** {recommendation['reasoning']}")
            st.progress(recommendation['efficiency_score'])
            st.write(f"Efficiency Score: {recommendation['efficiency_score']}%")
            
            if st.button("Apply Recommendation", type="primary"):
                st.success("Irrigation scheduled!")
                st.session_state.alerts.append({
                    'time': datetime.now(),
                    'type': 'success',
                    'message': f'Irrigation scheduled: {recommendation["amount"]:.1f}mm'
                })
    
    def render_yield_prediction(self):
        """Render yield prediction interface"""
        st.subheader("📈 Yield Prediction & Optimization")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Yield prediction visualization
            prediction = st.session_state.yield_predictions
            
            fig = go.Figure()
            
            # Historical average
            fig.add_hline(
                y=prediction['historical_average'],
                line_dash="dash",
                line_color="gray",
                annotation_text="Historical Average"
            )
            
            # Current prediction with confidence interval
            fig.add_trace(go.Scatter(
                x=['Current Prediction'],
                y=[prediction['current_prediction']],
                mode='markers',
                marker=dict(size=20, color='green'),
                name='Predicted Yield',
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[prediction['confidence_interval'][1] - prediction['current_prediction']],
                    arrayminus=[prediction['current_prediction'] - prediction['confidence_interval'][0]]
                )
            ))
            
            fig.update_layout(
                title="Yield Prediction with Confidence Interval",
                yaxis_title="Yield (tons/hectare)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Factors influencing yield
            st.write("**Factors Influencing Yield**")
            
            factors = prediction['factors']
            for factor, score in factors.items():
                st.write(f"**{factor}**")
                progress_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                st.progress(score/100)
                st.write(f"Score: {score}%")
                st.write("")
            
            # Recommendations for improvement
            st.write("**Recommendations for Improvement**")
            recommendations = [
                "Optimize nitrogen application timing",
                "Monitor pest pressure closely",
                "Maintain optimal soil moisture",
                "Consider foliar nutrient application"
            ]
            
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def render_market_intelligence(self):
        """Render market intelligence dashboard"""
        st.subheader("💰 Market Intelligence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price trend chart
            market_data = st.session_state.market_data
            price_data = market_data['prices']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=price_data['price_usd_per_ton'],
                mode='lines',
                name='Price (USD/ton)',
                line=dict(color='blue', width=2)
            ))
            
            # Add trend line
            z = np.polyfit(range(len(price_data)), price_data['price_usd_per_ton'], 1)
            trend_line = np.poly1d(z)(range(len(price_data)))
            
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Market Price Trend",
                xaxis_title="Date",
                yaxis_title="Price (USD/ton)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Market Summary**")
            
            current_price = market_data['current_price']
            weekly_change = market_data['weekly_change']
            sentiment = market_data['market_sentiment']
            
            st.metric("Current Price", f"${current_price:.0f}/ton", f"{weekly_change:+.1f}%")
            st.write(f"**Market Sentiment:** {sentiment}")
            
            st.write("**Market Recommendations**")
            for rec in market_data['recommendations']:
                st.write(f"• {rec}")
            
            # Selling decision tool
            st.write("**Selling Decision Tool**")
            target_price = st.number_input("Target Price (USD/ton)", value=int(current_price), min_value=100)
            
            if target_price <= current_price:
                st.success(f"✅ Current price meets your target! Consider selling.")
            else:
                price_gap = ((target_price - current_price) / current_price) * 100
                st.warning(f"❌ Target price is {price_gap:.1f}% above current market.")
    
    def render_analytics_summary(self):
        """Render analytics summary"""
        st.subheader("📊 Analytics Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Water Efficiency**")
            # Calculate water efficiency metrics
            total_irrigation = st.session_state.irrigation_history['irrigation_mm'].sum()
            days_with_irrigation = (st.session_state.irrigation_history['irrigation_mm'] > 0).sum()
            avg_irrigation = total_irrigation / days_with_irrigation if days_with_irrigation > 0 else 0
            
            st.metric("Total Water Used", f"{total_irrigation:.0f} mm")
            st.metric("Avg. Daily Irrigation", f"{avg_irrigation:.1f} mm")
            st.metric("Water Efficiency Score", "85%")
        
        with col2:
            st.write("**Crop Health**")
            st.metric("Disease Incidents", "3")
            st.metric("Treatment Success Rate", "95%")
            st.metric("Overall Health Score", "88%")
        
        with col3:
            st.write("**Economic Performance**")
            current_price = st.session_state.market_data['current_price']
            predicted_yield = st.session_state.yield_predictions['current_prediction']
            farm_area = st.session_state.farm_data['total_area']
            estimated_revenue = current_price * predicted_yield * farm_area
            
            st.metric("Estimated Revenue", f"${estimated_revenue:,.0f}")
            st.metric("Revenue per Hectare", f"${current_price * predicted_yield:,.0f}")
            st.metric("Profit Margin", "22%")
    
    def run(self):
        """Main dashboard execution"""
        self.render_header()
        
        # Sidebar
        soil_moisture, temperature = self.render_sidebar()
        
        # Main content
        self.render_overview_metrics()
        self.render_alerts()
        
        # Create tabs for different modules
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Disease Detection",
            "💧 Smart Irrigation", 
            "📈 Yield Prediction",
            "💰 Market Intelligence",
            "📊 Analytics"
        ])
        
        with tab1:
            self.render_disease_detection()
        
        with tab2:
            self.render_irrigation_dashboard()
        
        with tab3:
            self.render_yield_prediction()
        
        with tab4:
            self.render_market_intelligence()
        
        with tab5:
            self.render_analytics_summary()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>SmartFarm AI Dashboard - Addressing UN SDG 2: Zero Hunger through AI-powered agriculture</p>
            <p>Integrating Computer Vision, Machine Learning, Reinforcement Learning, and NLP for sustainable farming</p>
        </div>
        """, unsafe_allow_html=True)

# Main execution
def main():
    """Main function to run the dashboard"""
    dashboard = SmartFarmDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()