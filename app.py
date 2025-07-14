import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm, beta
from scipy.stats import proportions_ztest
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.multitest import multipletests
import datetime
import io
import base64

# Configure page
st.set_page_config(
    page_title="AnalyticsPro | Advanced A/B Testing Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .highlight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedABTesting:
    """Professional A/B Testing Framework"""
    
    def __init__(self):
        self.results = {}
        
    def load_sample_data(self, dataset_choice):
        """Load sample datasets for demonstration"""
        
        if dataset_choice == "Cookie Cats (Mobile Game)":
            # Simulate Cookie Cats data structure
            np.random.seed(42)
            n_control = 44700
            n_treatment = 45489
            
            # Control group - gate_30
            control_retention_1 = np.random.binomial(1, 0.4482, n_control)
            control_retention_7 = np.random.binomial(1, 0.1902, n_control)
            
            # Treatment group - gate_40 (slightly worse performance)
            treatment_retention_1 = np.random.binomial(1, 0.4423, n_treatment)
            treatment_retention_7 = np.random.binomial(1, 0.1820, n_treatment)
            
            # Create combined dataset
            control_df = pd.DataFrame({
                'user_id': range(1, n_control + 1),
                'version': 'gate_30',
                'retention_1': control_retention_1,
                'retention_7': control_retention_7
            })
            
            treatment_df = pd.DataFrame({
                'user_id': range(n_control + 1, n_control + n_treatment + 1),
                'version': 'gate_40',
                'retention_1': treatment_retention_1,
                'retention_7': treatment_retention_7
            })
            
            data = pd.concat([control_df, treatment_df], ignore_index=True)
            
            return data, {
                'control_group': 'gate_30',
                'treatment_group': 'gate_40',
                'metrics': ['retention_1', 'retention_7'],
                'description': 'Mobile game A/B test comparing two gate positions'
            }
            
        elif dataset_choice == "E-commerce Conversion":
            # Simulate e-commerce data
            np.random.seed(123)
            n_control = 5000
            n_treatment = 5000
            
            # Control group
            control_conversions = np.random.binomial(1, 0.125, n_control)
            control_purchases = np.random.binomial(1, 0.023, n_control)
            
            # Treatment group (better performance)
            treatment_conversions = np.random.binomial(1, 0.142, n_treatment)
            treatment_purchases = np.random.binomial(1, 0.028, n_treatment)
            
            control_df = pd.DataFrame({
                'user_id': range(1, n_control + 1),
                'variant': 'control',
                'converted': control_conversions,
                'purchased': control_purchases
            })
            
            treatment_df = pd.DataFrame({
                'user_id': range(n_control + 1, n_control + n_treatment + 1),
                'variant': 'treatment',
                'converted': treatment_conversions,
                'purchased': treatment_purchases
            })
            
            data = pd.concat([control_df, treatment_df], ignore_index=True)
            
            return data, {
                'control_group': 'control',
                'treatment_group': 'treatment',
                'metrics': ['converted', 'purchased'],
                'description': 'E-commerce website A/B test'
            }
            
        elif dataset_choice == "Email Campaign":
            # Simulate email campaign data
            np.random.seed(456)
            n_control = 8000
            n_treatment = 8000
            
            # Control group
            control_opens = np.random.binomial(1, 0.18, n_control)
            control_clicks = np.random.binomial(1, 0.045, n_control)
            
            # Treatment group
            treatment_opens = np.random.binomial(1, 0.22, n_treatment)
            treatment_clicks = np.random.binomial(1, 0.055, n_treatment)
            
            control_df = pd.DataFrame({
                'email_id': range(1, n_control + 1),
                'campaign': 'control',
                'opened': control_opens,
                'clicked': control_clicks
            })
            
            treatment_df = pd.DataFrame({
                'email_id': range(n_control + 1, n_control + n_treatment + 1),
                'campaign': 'treatment',
                'opened': treatment_opens,
                'clicked': treatment_clicks
            })
            
            data = pd.concat([control_df, treatment_df], ignore_index=True)
            
            return data, {
                'control_group': 'control',
                'treatment_group': 'treatment',
                'metrics': ['opened', 'clicked'],
                'description': 'Email marketing campaign A/B test'
            }
    
    def calculate_basic_stats(self, data, config, metric):
        """Calculate basic descriptive statistics"""
        
        control_col = config['control_group']
        treatment_col = config['treatment_group']
        
        if 'version' in data.columns:
            group_col = 'version'
        elif 'variant' in data.columns:
            group_col = 'variant'
        else:
            group_col = 'campaign'
            
        control_data = data[data[group_col] == control_col]
        treatment_data = data[data[group_col] == treatment_col]
        
        control_conversions = control_data[metric].sum()
        control_total = len(control_data)
        treatment_conversions = treatment_data[metric].sum()
        treatment_total = len(treatment_data)
        
        control_rate = control_conversions / control_total
        treatment_rate = treatment_conversions / treatment_total
        
        return {
            'control': {
                'conversions': control_conversions,
                'total': control_total,
                'rate': control_rate
            },
            'treatment': {
                'conversions': treatment_conversions,
                'total': treatment_total,
                'rate': treatment_rate
            },
            'difference': {
                'absolute': treatment_rate - control_rate,
                'relative': ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0
            }
        }
    
    def frequentist_test(self, stats_data):
        """Perform frequentist statistical test"""
        
        control = stats_data['control']
        treatment = stats_data['treatment']
        
        # Two-proportion z-test
        z_stat, p_value = proportions_ztest(
            [control['conversions'], treatment['conversions']],
            [control['total'], treatment['total']]
        )
        
        # Effect size (Cohen's h)
        effect_size = proportion_effectsize(control['rate'], treatment['rate'])
        
        # Confidence interval
        p1, p2 = control['rate'], treatment['rate']
        n1, n2 = control['total'], treatment['total']
        
        se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        diff = p2 - p1
        margin_error = 1.96 * se_diff
        
        ci_lower = diff - margin_error
        ci_upper = diff + margin_error
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'confidence_interval': {
                'lower': ci_lower * 100,
                'upper': ci_upper * 100
            }
        }
    
    def bayesian_analysis(self, stats_data, prior_alpha=1, prior_beta=1, n_simulations=100000):
        """Perform Bayesian analysis"""
        
        control = stats_data['control']
        treatment = stats_data['treatment']
        
        # Posterior parameters
        control_alpha = prior_alpha + control['conversions']
        control_beta = prior_beta + control['total'] - control['conversions']
        
        treatment_alpha = prior_alpha + treatment['conversions']
        treatment_beta = prior_beta + treatment['total'] - treatment['conversions']
        
        # Monte Carlo simulation
        np.random.seed(42)
        control_samples = np.random.beta(control_alpha, control_beta, n_simulations)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_simulations)
        
        # Key probabilities
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        
        # Relative improvement
        relative_improvement = (treatment_samples - control_samples) / control_samples
        
        # Credible interval
        ci_lower = np.percentile(relative_improvement, 2.5)
        ci_upper = np.percentile(relative_improvement, 97.5)
        
        # Expected loss
        loss_if_choose_treatment = np.mean(np.maximum(control_samples - treatment_samples, 0))
        loss_if_choose_control = np.mean(np.maximum(treatment_samples - control_samples, 0))
        
        # Risk assessment
        prob_negative = np.mean(relative_improvement < 0) * 100
        prob_large_effect = np.mean(relative_improvement > 0.1) * 100
        
        return {
            'prob_treatment_better': prob_treatment_better * 100,
            'credible_interval': {
                'lower': ci_lower * 100,
                'upper': ci_upper * 100
            },
            'expected_loss': {
                'choose_treatment': loss_if_choose_treatment * 100,
                'choose_control': loss_if_choose_control * 100
            },
            'risk_assessment': {
                'prob_negative': prob_negative,
                'prob_large_effect': prob_large_effect
            },
            'samples': {
                'control': control_samples,
                'treatment': treatment_samples,
                'relative_improvement': relative_improvement
            }
        }
    
    def power_analysis(self, baseline_rate, mde, alpha=0.05, power=0.8):
        """Calculate required sample size"""
        
        new_rate = baseline_rate * (1 + mde)
        effect_size = proportion_effectsize(baseline_rate, new_rate)
        
        try:
            sample_size = zt_ind_solve_power(
                effect_size=effect_size,
                power=power,
                alpha=alpha
            )
        except:
            # Fallback calculation
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            p_pooled = (baseline_rate + new_rate) / 2
            
            sample_size = (
                2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2
            ) / (baseline_rate - new_rate)**2
        
        return {
            'sample_size_per_group': int(sample_size),
            'total_sample_size': int(sample_size * 2),
            'baseline_rate': baseline_rate * 100,
            'target_rate': new_rate * 100,
            'effect_size': effect_size,
            'mde': mde * 100
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🧪 AnalyticsPro | Advanced A/B Testing Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Professional-Grade Statistical Analysis for Business Experimentation**")
    
    # Author info
    with st.expander("👨‍💼 About the Developer"):
        st.markdown("""
        **Peter Chika Ozo-Ogueji** - Senior Data Scientist & Analytics Professional
        
        This platform demonstrates advanced A/B testing methodologies including:
        - ✅ Frequentist & Bayesian Statistical Analysis
        - ✅ Power Analysis & Sample Size Calculation  
        - ✅ Sequential Testing & Early Stopping
        - ✅ Multiple Testing Corrections
        - ✅ Business Impact Assessment
        - ✅ Interactive Visualizations & Reporting
        
        **Technologies:** Python, Streamlit, Plotly, SciPy, NumPy, Pandas
        """)
    
    # Initialize framework
    ab_tester = AdvancedABTesting()
    
    # Sidebar navigation
    st.sidebar.title("🔬 Analysis Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🏠 Overview", "📊 Dataset Analysis", "🧮 Power Calculator", "📈 Live Experiment", "📋 Reports"]
    )
    
    if analysis_type == "🏠 Overview":
        show_overview()
    elif analysis_type == "📊 Dataset Analysis":
        show_dataset_analysis(ab_tester)
    elif analysis_type == "🧮 Power Calculator":
        show_power_calculator(ab_tester)
    elif analysis_type == "📈 Live Experiment":
        show_live_experiment(ab_tester)
    elif analysis_type == "📋 Reports":
        show_reports()

def show_overview():
    """Display platform overview"""
    
    st.header("🎯 Platform Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Statistical Methods</h3>
            <ul>
                <li>Frequentist Testing</li>
                <li>Bayesian Analysis</li>
                <li>Sequential Testing</li>
                <li>Multiple Comparisons</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class