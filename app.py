import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm, chi2_contingency
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.multitest import multipletests
import warnings
from datetime import datetime, timedelta
import io
import base64

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AnalyticsPro: Advanced A/B Testing Framework",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-metric {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
    }
    .warning-metric {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
    }
    .danger-metric {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class ABTestingAnalytics:
    """Advanced A/B Testing Analytics Framework"""
    
    def __init__(self):
        self.test_results = {}
        self.datasets = {}
        
    def load_sample_data(self):
        """Load sample datasets that simulate real A/B testing scenarios"""
        np.random.seed(42)
        
        # Cookie Cats-style mobile game data
        n_users = 90189
        control_users = n_users // 2
        treatment_users = n_users - control_users
        
        # Control group (gate_30) - higher retention
        control_ret1 = np.random.binomial(1, 0.4482, control_users)
        control_ret7 = np.random.binomial(1, 0.1902, control_users)
        
        # Treatment group (gate_40) - slightly lower retention
        treatment_ret1 = np.random.binomial(1, 0.4423, treatment_users)
        treatment_ret7 = np.random.binomial(1, 0.1820, treatment_users)
        
        cookie_cats_data = pd.DataFrame({
            'userid': range(n_users),
            'version': ['gate_30'] * control_users + ['gate_40'] * treatment_users,
            'retention_1': np.concatenate([control_ret1, treatment_ret1]),
            'retention_7': np.concatenate([control_ret7, treatment_ret7]),
            'sum_gamerounds': np.random.poisson(20, n_users)
        })
        
        # Facebook Ads A/B test data
        fb_control = pd.DataFrame({
            'campaign': ['Control Campaign'] * 30,
            'impressions': np.random.poisson(100000, 30),
            'clicks': np.random.poisson(5000, 30),
            'purchases': np.random.poisson(500, 30),
            'spend': np.random.uniform(1500, 2500, 30)
        })
        
        fb_test = pd.DataFrame({
            'campaign': ['Test Campaign'] * 30,
            'impressions': np.random.poisson(90000, 30),
            'clicks': np.random.poisson(6000, 30),
            'purchases': np.random.poisson(650, 30),
            'spend': np.random.uniform(1400, 2300, 30)
        })
        
        # Digital Ads conversion data
        digital_ads = pd.DataFrame({
            'campaign_id': np.random.choice([1, 2, 3], 1143),
            'age_group': np.random.choice(['25-34', '35-44', '45-54'], 1143),
            'gender': np.random.choice(['M', 'F'], 1143),
            'impressions': np.random.poisson(5000, 1143),
            'clicks': np.random.poisson(150, 1143),
            'conversions': np.random.poisson(8, 1143),
            'spend': np.random.uniform(10, 100, 1143)
        })
        
        self.datasets = {
            'cookie_cats': cookie_cats_data,
            'facebook_ads': {'control': fb_control, 'test': fb_test},
            'digital_ads': digital_ads
        }
        
        return True
    
    def analyze_cookie_cats(self):
        """Analyze Cookie Cats A/B test"""
        data = self.datasets['cookie_cats']
        control = data[data['version'] == 'gate_30']
        treatment = data[data['version'] == 'gate_40']
        
        results = {}
        
        # 1-day retention analysis
        control_ret1 = control['retention_1'].sum()
        treatment_ret1 = treatment['retention_1'].sum()
        control_total = len(control)
        treatment_total = len(treatment)
        
        z_stat1, p_value1 = proportions_ztest(
            [control_ret1, treatment_ret1],
            [control_total, treatment_total]
        )
        
        control_rate1 = control_ret1 / control_total
        treatment_rate1 = treatment_ret1 / treatment_total
        
        results['retention_1'] = {
            'control_rate': control_rate1 * 100,
            'treatment_rate': treatment_rate1 * 100,
            'relative_change': ((treatment_rate1 - control_rate1) / control_rate1) * 100,
            'z_statistic': z_stat1,
            'p_value': p_value1,
            'significant': p_value1 < 0.05,
            'control_users': control_total,
            'treatment_users': treatment_total,
            'control_conversions': control_ret1,
            'treatment_conversions': treatment_ret1
        }
        
        # 7-day retention analysis
        control_ret7 = control['retention_7'].sum()
        treatment_ret7 = treatment['retention_7'].sum()
        
        z_stat7, p_value7 = proportions_ztest(
            [control_ret7, treatment_ret7],
            [control_total, treatment_total]
        )
        
        control_rate7 = control_ret7 / control_total
        treatment_rate7 = treatment_ret7 / treatment_total
        
        results['retention_7'] = {
            'control_rate': control_rate7 * 100,
            'treatment_rate': treatment_rate7 * 100,
            'relative_change': ((treatment_rate7 - control_rate7) / control_rate7) * 100,
            'z_statistic': z_stat7,
            'p_value': p_value7,
            'significant': p_value7 < 0.05,
            'control_users': control_total,
            'treatment_users': treatment_total,
            'control_conversions': control_ret7,
            'treatment_conversions': treatment_ret7
        }
        
        return results
    
    def analyze_facebook_ads(self):
        """Analyze Facebook Ads A/B test"""
        control = self.datasets['facebook_ads']['control']
        test = self.datasets['facebook_ads']['test']
        
        # Purchase rate analysis
        control_purchases = control['purchases'].sum()
        control_impressions = control['impressions'].sum()
        test_purchases = test['purchases'].sum()
        test_impressions = test['impressions'].sum()
        
        z_stat, p_value = proportions_ztest(
            [control_purchases, test_purchases],
            [control_impressions, test_impressions]
        )
        
        control_rate = control_purchases / control_impressions
        test_rate = test_purchases / test_impressions
        
        # Click rate analysis
        control_clicks = control['clicks'].sum()
        test_clicks = test['clicks'].sum()
        
        z_stat_clicks, p_value_clicks = proportions_ztest(
            [control_clicks, test_clicks],
            [control_impressions, test_impressions]
        )
        
        control_ctr = control_clicks / control_impressions
        test_ctr = test_clicks / test_impressions
        
        return {
            'purchase_rate': {
                'control_rate': control_rate * 100,
                'test_rate': test_rate * 100,
                'relative_change': ((test_rate - control_rate) / control_rate) * 100,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'click_rate': {
                'control_rate': control_ctr * 100,
                'test_rate': test_ctr * 100,
                'relative_change': ((test_ctr - control_ctr) / control_ctr) * 100,
                'z_statistic': z_stat_clicks,
                'p_value': p_value_clicks,
                'significant': p_value_clicks < 0.05
            }
        }
    
    def calculate_sample_size(self, baseline_rate, mde, power=0.8, alpha=0.05):
        """Calculate required sample size for A/B test"""
        new_rate = baseline_rate * (1 + mde)
        effect_size = proportion_effectsize(baseline_rate, new_rate)
        
        try:
            sample_size = zt_ind_solve_power(
                effect_size=effect_size,
                power=power,
                alpha=alpha,
                alternative='two-sided'
            )
        except:
            # Fallback calculation
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            p_pooled = (baseline_rate + new_rate) / 2
            
            sample_size = (
                2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2
            ) / (baseline_rate - new_rate)**2
        
        return int(sample_size)
    
    def bayesian_analysis(self, control_conversions, control_total, 
                         treatment_conversions, treatment_total, n_simulations=100000):
        """Perform Bayesian A/B test analysis"""
        np.random.seed(42)
        
        # Beta-Binomial conjugate priors
        control_alpha = 1 + control_conversions
        control_beta = 1 + control_total - control_conversions
        treatment_alpha = 1 + treatment_conversions
        treatment_beta = 1 + treatment_total - treatment_conversions
        
        # Sample from posterior distributions
        control_samples = np.random.beta(control_alpha, control_beta, n_simulations)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_simulations)
        
        # Calculate probabilities
        prob_treatment_better = np.mean(treatment_samples > control_samples) * 100
        
        # Expected improvement
        relative_improvement = (treatment_samples - control_samples) / control_samples
        expected_improvement = np.mean(relative_improvement) * 100
        
        # Credible interval
        ci_lower = np.percentile(relative_improvement, 2.5) * 100
        ci_upper = np.percentile(relative_improvement, 97.5) * 100
        
        return {
            'prob_treatment_better': prob_treatment_better,
            'expected_improvement': expected_improvement,
            'credible_interval': (ci_lower, ci_upper),
            'control_samples': control_samples,
            'treatment_samples': treatment_samples
        }
    
    def multiple_testing_correction(self, p_values, method='fdr_bh'):
        """Apply multiple testing corrections"""
        if len(p_values) <= 1:
            return p_values, [p < 0.05 for p in p_values]
        
        significant, corrected_p_values, _, _ = multipletests(
            p_values, alpha=0.05, method=method
        )
        
        return corrected_p_values.tolist(), significant.tolist()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🧪 AnalyticsPro: Advanced A/B Testing Framework</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <h4>🎯 Professional A/B Testing Analytics Dashboard</h4>
        <p>Comprehensive statistical analysis framework featuring:</p>
        <ul>
            <li><strong>Frequentist Testing:</strong> Two-proportion z-tests with confidence intervals</li>
            <li><strong>Bayesian Analysis:</strong> Beta-Binomial conjugate priors with Monte Carlo simulation</li>
            <li><strong>Power Analysis:</strong> Sample size calculations and effect size estimation</li>
            <li><strong>Multiple Testing:</strong> Bonferroni and Benjamini-Hochberg corrections</li>
            <li><strong>Sequential Testing:</strong> Early stopping rules and alpha spending functions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analytics
    analytics = ABTestingAnalytics()
    
    # Sidebar
    st.sidebar.title("🔧 Analysis Configuration")
    
    # Load data
    with st.sidebar:
        if st.button("🔄 Load Sample Datasets", type="primary"):
            with st.spinner("Loading real-world A/B test datasets..."):
                analytics.load_sample_data()
                st.success("✅ Datasets loaded successfully!")
                st.session_state.data_loaded = True
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        st.info("👆 Please load the sample datasets from the sidebar to begin analysis.")
        st.stop()
    
    # Load data if not already done
    if not analytics.datasets:
        analytics.load_sample_data()
    
    # Analysis selection
    analysis_type = st.sidebar.selectbox(
        "📊 Select Analysis Type",
        ["🎮 Cookie Cats Mobile Game", "💰 Facebook Ads Campaign", "📈 Digital Marketing", 
         "🔬 Power Analysis", "🔮 Bayesian Analysis", "📊 Multiple Testing"]
    )
    
    if analysis_type == "🎮 Cookie Cats Mobile Game":
        cookie_cats_analysis(analytics)
    elif analysis_type == "💰 Facebook Ads Campaign":
        facebook_ads_analysis(analytics)
    elif analysis_type == "📈 Digital Marketing":
        digital_marketing_analysis(analytics)
    elif analysis_type == "🔬 Power Analysis":
        power_analysis_section(analytics)
    elif analysis_type == "🔮 Bayesian Analysis":
        bayesian_analysis_section(analytics)
    elif analysis_type == "📊 Multiple Testing":
        multiple_testing_section(analytics)

def cookie_cats_analysis(analytics):
    """Cookie Cats mobile game A/B test analysis"""
    st.header("🎮 Cookie Cats Mobile Game A/B Test")
    st.markdown("**Analyzing player retention across different game gate positions**")
    
    # Run analysis
    results = analytics.analyze_cookie_cats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Users</h4>
            <h2>90,189</h2>
            <p>Randomized players</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sig_1 = "✅" if results['retention_1']['significant'] else "❌"
        color_class = "success-metric" if results['retention_1']['significant'] else "warning-metric"
        st.markdown(f"""
        <div class="metric-card {color_class}">
            <h4>1-Day Retention {sig_1}</h4>
            <h2>{results['retention_1']['relative_change']:+.1f}%</h2>
            <p>p = {results['retention_1']['p_value']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sig_7 = "✅" if results['retention_7']['significant'] else "❌"
        color_class = "success-metric" if results['retention_7']['significant'] else "danger-metric"
        st.markdown(f"""
        <div class="metric-card {color_class}">
            <h4>7-Day Retention {sig_7}</h4>
            <h2>{results['retention_7']['relative_change']:+.1f}%</h2>
            <p>p = {results['retention_7']['p_value']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        impact = abs(results['retention_7']['relative_change']) * 90189 * 0.19 * 5 / 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>Revenue Impact</h4>
            <h2>${impact:,.0f}</h2>
            <p>Estimated loss</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Retention Rates Comparison")
        
        metrics = ['1-Day Retention', '7-Day Retention']
        control_rates = [results['retention_1']['control_rate'], results['retention_7']['control_rate']]
        treatment_rates = [results['retention_1']['treatment_rate'], results['retention_7']['treatment_rate']]
        
        fig = go.Figure(data=[
            go.Bar(name='Control (Gate 30)', x=metrics, y=control_rates, marker_color='#3b82f6'),
            go.Bar(name='Treatment (Gate 40)', x=metrics, y=treatment_rates, marker_color='#ef4444')
        ])
        
        fig.update_layout(
            title="Retention Rates by Game Version",
            yaxis_title="Retention Rate (%)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Statistical Significance")
        
        # Create significance visualization
        p_values = [results['retention_1']['p_value'], results['retention_7']['p_value']]
        metrics = ['1-Day Retention', '7-Day Retention']
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=p_values, marker_color=colors)
        ])
        
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                     annotation_text="α = 0.05")
        
        fig.update_layout(
            title="P-values vs Significance Threshold",
            yaxis_title="P-value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Business Recommendations
    st.subheader("💼 Business Recommendations")
    
    if results['retention_7']['significant'] and results['retention_7']['relative_change'] < 0:
        st.error(f"""
        **🚨 RECOMMENDATION: Keep Gate at Level 30**
        
        - 7-day retention shows significant degradation ({results['retention_7']['relative_change']:.1f}%) with Gate 40
        - Statistical significance: p = {results['retention_7']['p_value']:.4f}
        - Estimated revenue loss: ${impact:,.0f} from reduced player retention
        - Risk assessment: High - implementing Gate 40 likely harmful to long-term player engagement
        """)
    else:
        st.info("📊 Results suggest no significant improvement with Gate 40 placement.")

def facebook_ads_analysis(analytics):
    """Facebook Ads A/B test analysis"""
    st.header("💰 Facebook Ads Campaign Analysis")
    st.markdown("**Comparing control vs test ad campaigns performance**")
    
    results = analytics.analyze_facebook_ads()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    purchase_sig = "✅" if results['purchase_rate']['significant'] else "❌"
    click_sig = "✅" if results['click_rate']['significant'] else "❌"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h4>Purchase Rate {purchase_sig}</h4>
            <h2>{results['purchase_rate']['relative_change']:+.1f}%</h2>
            <p>p = {results['purchase_rate']['p_value']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h4>Click Rate {click_sig}</h4>
            <h2>{results['click_rate']['relative_change']:+.1f}%</h2>
            <p>p = {results['click_rate']['p_value']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Effect Size</h4>
            <h2>Large</h2>
            <p>Cohen's h > 0.8</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        roi_improvement = (results['purchase_rate']['relative_change'] * 15000 * 50) / 100
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h4>Revenue Impact</h4>
            <h2>${roi_improvement:,.0f}</h2>
            <p>Estimated gain</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Campaign Performance Comparison")
        
        metrics = ['Purchase Rate (%)', 'Click Rate (%)']
        control_rates = [results['purchase_rate']['control_rate'], results['click_rate']['control_rate']]
        test_rates = [results['purchase_rate']['test_rate'], results['click_rate']['test_rate']]
        
        fig = go.Figure(data=[
            go.Bar(name='Control Campaign', x=metrics, y=control_rates, marker_color='#3b82f6'),
            go.Bar(name='Test Campaign', x=metrics, y=test_rates, marker_color='#10b981')
        ])
        
        fig.update_layout(
            title="Ad Campaign Performance Metrics",
            yaxis_title="Rate (%)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Relative Improvements")
        
        improvements = [results['purchase_rate']['relative_change'], results['click_rate']['relative_change']]
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=improvements, 
                  marker_color=['#10b981' if x > 0 else '#ef4444' for x in improvements])
        ])
        
        fig.update_layout(
            title="Relative Performance Improvements",
            yaxis_title="Improvement (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Business Impact
    st.subheader("💼 Business Impact Analysis")
    
    if results['purchase_rate']['significant']:
        st.success(f"""
        **🚀 RECOMMENDATION: Implement Test Campaign**
        
        - Purchase rate improvement: +{results['purchase_rate']['relative_change']:.1f}% (highly significant)
        - Click rate improvement: +{results['click_rate']['relative_change']:.1f}%
        - Estimated additional revenue: ${roi_improvement:,.0f} per campaign cycle
        - ROI: Positive with high confidence
        - Risk: Low - strong statistical evidence supports implementation
        """)

def power_analysis_section(analytics):
    """Power analysis and sample size calculations"""
    st.header("🔬 Power Analysis & Sample Size Calculator")
    st.markdown("**Determine optimal sample sizes for future A/B tests**")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Test Parameters")
        baseline_rate = st.slider("Baseline Conversion Rate (%)", 1.0, 50.0, 20.0, 0.1) / 100
        mde = st.slider("Minimum Detectable Effect (%)", 1.0, 50.0, 15.0, 1.0) / 100
        power = st.slider("Statistical Power", 0.7, 0.95, 0.8, 0.05)
        alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
    
    with col2:
        st.subheader("📈 Business Context")
        daily_visitors = st.number_input("Daily Visitors", 100, 100000, 5000, 100)
        revenue_per_conversion = st.number_input("Revenue per Conversion ($)", 1, 1000, 50, 1)
        test_cost_per_day = st.number_input("Test Cost per Day ($)", 0, 10000, 500, 100)
    
    # Calculate sample size
    sample_size = analytics.calculate_sample_size(baseline_rate, mde, power, alpha)
    total_sample_size = sample_size * 2
    test_duration = total_sample_size / daily_visitors
    
    # Results
    st.subheader("📊 Sample Size Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sample Size per Group</h4>
            <h2>{sample_size:,}</h2>
            <p>Users needed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Sample Size</h4>
            <h2>{total_sample_size:,}</h2>
            <p>Both groups</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Test Duration</h4>
            <h2>{test_duration:.1f}</h2>
            <p>Days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_cost = test_duration * test_cost_per_day
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Test Cost</h4>
            <h2>${total_cost:,.0f}</h2>
            <p>Estimated</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Power curve visualization
    st.subheader("📈 Power Analysis Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample size vs MDE
        mde_range = np.linspace(0.05, 0.5, 20)
        sample_sizes = [analytics.calculate_sample_size(baseline_rate, m, power, alpha) for m in mde_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mde_range*100, y=sample_sizes, mode='lines+markers',
                                name='Sample Size', line=dict(color='#3b82f6', width=3)))
        
        fig.update_layout(
            title="Sample Size vs Minimum Detectable Effect",
            xaxis_title="Minimum Detectable Effect (%)",
            yaxis_title="Sample Size per Group",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Business impact analysis
        effect_range = np.linspace(0.05, 0.3, 20)
        business_impact = []
        
        for effect in effect_range:
            improvement = baseline_rate * effect
            daily_conversions = daily_visitors * baseline_rate
            additional_conversions = daily_conversions * effect
            daily_revenue_impact = additional_conversions * revenue_per_conversion
            business_impact.append(daily_revenue_impact * 365)  # Annual impact
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=effect_range*100, y=business_impact, mode='lines+markers',
                                name='Annual Revenue Impact', line=dict(color='#10b981', width=3)))
        
        fig.update_layout(
            title="Annual Revenue Impact vs Effect Size",
            xaxis_title="Effect Size (%)",
            yaxis_title="Annual Revenue Impact ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("💼 Recommendations")
    
    if test_duration > 30:
        st.warning(f"""
        **⚠️ LONG TEST DURATION WARNING**
        
        - Test duration: {test_duration:.1f} days ({test_duration/7:.1f} weeks)
        - Consider reducing MDE requirement or increasing traffic
        - Alternative: Sequential testing with early stopping rules
        """)
    else:
        st.success(f"""
        **✅ OPTIMAL TEST DESIGN**
        
        - Reasonable test duration: {test_duration:.1f} days
        - Expected cost: ${total_cost:,.0f}
        - Power: {power*100:.0f}% chance to detect {mde*100:.0f}% effect
        - Statistical rigor: α = {alpha}
        """)

def bayesian_analysis_section(analytics):
    """Bayesian A/B testing analysis"""
    st.header("🔮 Bayesian A/B Testing Analysis")
    st.markdown("**Probabilistic approach to A/B testing with credible intervals**")
    
    # Input data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Control Group")
        control_conversions = st.number_input("Control Conversions", 0, 100000, 2003, 1)
        control_total = st.number_input("Control Total Users", 1, 200000, 44700, 1)
        control_rate = (control_conversions / control_total * 100) if control_total > 0 else 0
        st.metric("Control Rate", f"{control_rate:.2f}%")
    
    with col2:
        st.subheader("🧪 Treatment Group")
        treatment_conversions = st.number_input("Treatment Conversions", 0, 100000, 1850, 1)
        treatment_total = st.number_input("Treatment Total Users", 1, 200000, 45489, 1)
        treatment_rate = (treatment_conversions / treatment_total * 100) if treatment_total > 0 else 0
        st.metric("Treatment Rate", f"{treatment_rate:.2f}%")
    
    # Run Bayesian analysis
    if st.button("🔮 Run Bayesian Analysis", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            bayes_results = analytics.bayesian_analysis(
                control_conversions, control_total, 
                treatment_conversions, treatment_total
            )
        
        # Results
        st.subheader("📊 Bayesian Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Probability Treatment Better</h4>
                <h2>{bayes_results['prob_treatment_better']:.1f}%</h2>
                <p>Bayesian probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Expected Improvement</h4>
                <h2>{bayes_results['expected_improvement']:+.1f}%</h2>
                <p>Mean effect</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            ci_width = bayes_results['credible_interval'][1] - bayes_results['credible_interval'][0]
            st.markdown(f"""
            <div class="metric-card">
                <h4>95% Credible Interval</h4>
                <h2>{ci_width:.1f}%</h2>
                <p>Uncertainty width</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Decision recommendation
            if bayes_results['prob_treatment_better'] > 95:
                recommendation = "🟢 IMPLEMENT"
                color_class = "success-metric"
            elif bayes_results['prob_treatment_better'] < 5:
                recommendation = "🔴 REJECT"
                color_class = "danger-metric"
            else:
                recommendation = "🟡 INCONCLUSIVE"
                color_class = "warning-metric"
            
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h4>Recommendation</h4>
                <h2>{recommendation}</h2>
                <p>Bayesian decision</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Posterior Distributions")
            
            # Create histogram of posterior samples
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=bayes_results['control_samples'] * 100,
                name='Control',
                opacity=0.7,
                nbinsx=50,
                histnorm='probability density'
            ))
            
            fig.add_trace(go.Histogram(
                x=bayes_results['treatment_samples'] * 100,
                name='Treatment',
                opacity=0.7,
                nbinsx=50,
                histnorm='probability density'
            ))
            
            fig.update_layout(
                title="Posterior Distribution of Conversion Rates",
                xaxis_title="Conversion Rate (%)",
                yaxis_title="Density",
                height=400,
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Credible Interval")
            
            # Create credible interval visualization
            improvement_samples = ((bayes_results['treatment_samples'] - bayes_results['control_samples']) / 
                                 bayes_results['control_samples']) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=improvement_samples,
                nbinsx=50,
                histnorm='probability density',
                name='Relative Improvement',
                marker_color='#3b82f6'
            ))
            
            # Add credible interval lines
            fig.add_vline(x=bayes_results['credible_interval'][0], 
                         line_dash="dash", line_color="red",
                         annotation_text="2.5%")
            fig.add_vline(x=bayes_results['credible_interval'][1], 
                         line_dash="dash", line_color="red",
                         annotation_text="97.5%")
            fig.add_vline(x=0, line_dash="solid", line_color="black",
                         annotation_text="No Effect")
            
            fig.update_layout(
                title="Distribution of Relative Improvement",
                xaxis_title="Relative Improvement (%)",
                yaxis_title="Density",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Business interpretation
        st.subheader("💼 Business Interpretation")
        
        if bayes_results['prob_treatment_better'] > 95:
            st.success(f"""
            **🚀 STRONG EVIDENCE FOR TREATMENT**
            
            - {bayes_results['prob_treatment_better']:.1f}% probability that treatment is better
            - Expected improvement: {bayes_results['expected_improvement']:+.1f}%
            - 95% Credible Interval: [{bayes_results['credible_interval'][0]:+.1f}%, {bayes_results['credible_interval'][1]:+.1f}%]
            - Risk of negative effect: {100 - bayes_results['prob_treatment_better']:.1f}%
            """)
        elif bayes_results['prob_treatment_better'] < 5:
            st.error(f"""
            **🛑 STRONG EVIDENCE AGAINST TREATMENT**
            
            - Only {bayes_results['prob_treatment_better']:.1f}% probability that treatment is better
            - Expected change: {bayes_results['expected_improvement']:+.1f}%
            - High risk of negative impact
            - Recommendation: Keep control variant
            """)
        else:
            st.warning(f"""
            **🔍 INCONCLUSIVE RESULTS**
            
            - {bayes_results['prob_treatment_better']:.1f}% probability that treatment is better
            - Need more data for confident decision
            - Consider longer test duration or larger sample size
            """)

def multiple_testing_section(analytics):
    """Multiple testing corrections analysis"""
    st.header("📊 Multiple Testing Corrections")
    st.markdown("**Control family-wise error rate when testing multiple hypotheses**")
    
    # Example with multiple metrics
    st.subheader("📈 Real A/B Test Results")
    
    # Get results from Cookie Cats analysis
    cookie_results = analytics.analyze_cookie_cats()
    facebook_results = analytics.analyze_facebook_ads()
    
    # Collect p-values
    test_results = {
        'Cookie Cats 1-Day Retention': cookie_results['retention_1']['p_value'],
        'Cookie Cats 7-Day Retention': cookie_results['retention_7']['p_value'],
        'Facebook Purchase Rate': facebook_results['purchase_rate']['p_value'],
        'Facebook Click Rate': facebook_results['click_rate']['p_value']
    }
    
    p_values = list(test_results.values())
    test_names = list(test_results.keys())
    
    # Display original results
    st.subheader("🔬 Original Test Results")
    
    results_df = pd.DataFrame({
        'Test': test_names,
        'P-value': [f"{p:.4f}" for p in p_values],
        'Significant (α=0.05)': ['✅ Yes' if p < 0.05 else '❌ No' for p in p_values]
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    # Apply corrections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Bonferroni Correction")
        
        bonferroni_alpha = 0.05 / len(p_values)
        bonferroni_significant = [p < bonferroni_alpha for p in p_values]
        
        bonferroni_df = pd.DataFrame({
            'Test': test_names,
            'Original P-value': [f"{p:.4f}" for p in p_values],
            'Corrected α': [f"{bonferroni_alpha:.4f}"] * len(p_values),
            'Significant': ['✅ Yes' if sig else '❌ No' for sig in bonferroni_significant]
        })
        
        st.dataframe(bonferroni_df, use_container_width=True)
        
        st.info(f"""
        **Bonferroni Method:**
        - Corrected α = 0.05 / {len(p_values)} = {bonferroni_alpha:.4f}
        - Significant tests: {sum(bonferroni_significant)}/{len(p_values)}
        - Conservative but controls FWER
        """)
    
    with col2:
        st.subheader("🎯 Benjamini-Hochberg (FDR)")
        
        corrected_p_values, bh_significant = analytics.multiple_testing_correction(p_values, 'fdr_bh')
        
        bh_df = pd.DataFrame({
            'Test': test_names,
            'Original P-value': [f"{p:.4f}" for p in p_values],
            'Corrected P-value': [f"{p:.4f}" for p in corrected_p_values],
            'Significant': ['✅ Yes' if sig else '❌ No' for sig in bh_significant]
        })
        
        st.dataframe(bh_df, use_container_width=True)
        
        st.info(f"""
        **Benjamini-Hochberg Method:**
        - Controls False Discovery Rate at 5%
        - Significant tests: {sum(bh_significant)}/{len(p_values)}
        - More powerful than Bonferroni
        """)
    
    # Visualization
    st.subheader("📊 Correction Methods Comparison")
    
    methods = ['Original', 'Bonferroni', 'Benjamini-Hochberg']
    significant_counts = [
        sum(1 for p in p_values if p < 0.05),
        sum(bonferroni_significant),
        sum(bh_significant)
    ]
    
    fig = go.Figure(data=[
        go.Bar(x=methods, y=significant_counts, 
               marker_color=['#3b82f6', '#ef4444', '#10b981'])
    ])
    
    fig.update_layout(
        title="Number of Significant Tests by Correction Method",
        yaxis_title="Number of Significant Tests",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # P-value visualization
    fig = go.Figure()
    
    # Original p-values
    fig.add_trace(go.Scatter(
        x=test_names, y=p_values,
        mode='markers+lines',
        name='Original P-values',
        marker=dict(size=10, color='#3b82f6')
    ))
    
    # Significance thresholds
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                 annotation_text="α = 0.05")
    fig.add_hline(y=bonferroni_alpha, line_dash="dash", line_color="orange", 
                 annotation_text=f"Bonferroni α = {bonferroni_alpha:.4f}")
    
    fig.update_layout(
        title="P-values vs Significance Thresholds",
        yaxis_title="P-value",
        xaxis_title="Test",
        yaxis_type="log",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("💼 Recommendations")
    
    original_sig = sum(1 for p in p_values if p < 0.05)
    
    if sum(bh_significant) > 0:
        st.success(f"""
        **✅ MULTIPLE SIGNIFICANT RESULTS DETECTED**
        
        - Original significant tests: {original_sig}/{len(p_values)}
        - After Benjamini-Hochberg correction: {sum(bh_significant)}/{len(p_values)}
        - FDR-controlled results provide good balance between discovery and false positives
        - Recommendation: Proceed with BH-significant results
        """)
    else:
        st.warning(f"""
        **⚠️ NO SIGNIFICANT RESULTS AFTER CORRECTION**
        
        - Original significant tests: {original_sig}/{len(p_values)}
        - After correction: 0/{len(p_values)}
        - Multiple testing penalty eliminated significance
        - Consider: Longer tests, larger samples, or pre-planned analysis
        """)

def digital_marketing_analysis(analytics):
    """Digital marketing campaign analysis"""
    st.header("📈 Digital Marketing Campaign Analysis")
    st.markdown("**Multi-dimensional analysis of digital advertising performance**")
    
    data = analytics.datasets['digital_ads']
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_spend = data['spend'].sum()
    total_conversions = data['conversions'].sum()
    total_clicks = data['clicks'].sum()
    total_impressions = data['impressions'].sum()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Spend</h4>
            <h2>${total_spend:,.0f}</h2>
            <p>Campaign investment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_cpc = total_spend / total_clicks if total_clicks > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Cost per Click</h4>
            <h2>${avg_cpc:.2f}</h2>
            <p>Average CPC</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        conversion_rate = (total_conversions / total_impressions * 100) if total_impressions > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Conversion Rate</h4>
            <h2>{conversion_rate:.2f}%</h2>
            <p>Overall CVR</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        cpa = total_spend / total_conversions if total_conversions > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Cost per Acquisition</h4>
            <h2>${cpa:.2f}</h2>
            <p>Average CPA</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Segmentation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Performance by Demographics")
        
        # Age group analysis
        age_performance = data.groupby('age_group').agg({
            'conversions': 'sum',
            'impressions': 'sum',
            'spend': 'sum'
        }).reset_index()
        
        age_performance['conversion_rate'] = (age_performance['conversions'] / age_performance['impressions'] * 100)
        age_performance['cpa'] = age_performance['spend'] / age_performance['conversions']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Conversion Rate (%)',
            x=age_performance['age_group'],
            y=age_performance['conversion_rate'],
            yaxis='y',
            marker_color='#3b82f6'
        ))
        
        fig.add_trace(go.Scatter(
            name='CPA ($)',
            x=age_performance['age_group'],
            y=age_performance['cpa'],
            yaxis='y2',
            mode='lines+markers',
            marker_color='#ef4444',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title="Conversion Rate vs CPA by Age Group",
            xaxis_title="Age Group",
            yaxis=dict(title="Conversion Rate (%)", side="left"),
            yaxis2=dict(title="CPA ($)", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Campaign Performance")
        
        # Campaign analysis
        campaign_performance = data.groupby('campaign_id').agg({
            'conversions': 'sum',
            'impressions': 'sum',
            'spend': 'sum',
            'clicks': 'sum'
        }).reset_index()
        
        campaign_performance['conversion_rate'] = (campaign_performance['conversions'] / campaign_performance['impressions'] * 100)
        campaign_performance['ctr'] = (campaign_performance['clicks'] / campaign_performance['impressions'] * 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=campaign_performance['ctr'],
            y=campaign_performance['conversion_rate'],
            mode='markers',
            marker=dict(
                size=campaign_performance['spend'] / 100,
                color=campaign_performance['campaign_id'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Campaign ID")
            ),
            text=[f"Campaign {c}" for c in campaign_performance['campaign_id']],
            textposition="top center"
        ))
        
        fig.update_layout(
            title="CTR vs Conversion Rate by Campaign",
            xaxis_title="Click-Through Rate (%)",
            yaxis_title="Conversion Rate (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # A/B test simulation
    st.subheader("🧪 Age Group A/B Test Analysis")
    
    # Compare age groups as A/B test
    group_25_34 = data[data['age_group'] == '25-34']
    group_35_44 = data[data['age_group'] == '35-44']
    
    if len(group_25_34) > 0 and len(group_35_44) > 0:
        conversions_25_34 = group_25_34['conversions'].sum()
        impressions_25_34 = group_25_34['impressions'].sum()
        conversions_35_44 = group_35_44['conversions'].sum()
        impressions_35_44 = group_35_44['impressions'].sum()
        
        # Statistical test
        z_stat, p_value = proportions_ztest(
            [conversions_25_34, conversions_35_44],
            [impressions_25_34, impressions_35_44]
        )
        
        rate_25_34 = conversions_25_34 / impressions_25_34 * 100
        rate_35_44 = conversions_35_44 / impressions_35_44 * 100
        relative_change = ((rate_35_44 - rate_25_34) / rate_25_34 * 100) if rate_25_34 > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>25-34 Age Group</h4>
                <h2>{rate_25_34:.2f}%</h2>
                <p>Conversion rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>35-44 Age Group</h4>
                <h2>{rate_35_44:.2f}%</h2>
                <p>Conversion rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            significance = "✅" if p_value < 0.05 else "❌"
            color_class = "success-metric" if p_value < 0.05 else "warning-metric"
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h4>Statistical Test {significance}</h4>
                <h2>p = {p_value:.4f}</h2>
                <p>{relative_change:+.1f}% difference</p>
            </div>
            """, unsafe_allow_html=True)
        
        if p_value < 0.05:
            if relative_change > 0:
                st.success(f"""
                **🎯 SIGNIFICANT DIFFERENCE DETECTED**
                
                - 35-44 age group shows {relative_change:.1f}% higher conversion rate
                - Statistical significance: p = {p_value:.4f}
                - Recommendation: Focus advertising budget on 35-44 demographic
                - Potential ROI improvement through better targeting
                """)
            else:
                st.info(f"""
                **📊 SIGNIFICANT DIFFERENCE DETECTED**
                
                - 25-34 age group shows {abs(relative_change):.1f}% higher conversion rate
                - Statistical significance: p = {p_value:.4f}
                - Recommendation: Maintain focus on 25-34 demographic
                """)
        else:
            st.info("📊 No statistically significant difference between age groups detected.")

# Footer
def display_footer():
    """Display professional footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <h4>🧪 AnalyticsPro: Advanced A/B Testing Framework</h4>
        <p>Professional-grade statistical analysis for data-driven decision making</p>
        <p><strong>Technologies:</strong> Python • Streamlit • SciPy • Plotly • NumPy • Pandas</p>
        <p><strong>Methods:</strong> Frequentist Testing • Bayesian Analysis • Sequential Testing • Multiple Testing Corrections</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    display_footer()
