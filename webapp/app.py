# ============================================================================
# PRISM-MB: Probabilistic Response-factor Informed Structural Mass Balance
# Interactive Web Application
# ============================================================================
# Author: Aryan Ranjan
# Institution: VIT Bhopal University
# Version: 1.0.0
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="PRISM-MB Calculator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/aryanranjan/PRISM-MB',
        'Report a bug': 'https://github.com/aryanranjan/PRISM-MB/issues',
        'About': '# PRISM-MB Calculator\nDeveloped by Aryan Ranjan | VIT Bhopal University'
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #F59E0B;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #EF4444;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #3B82F6;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class DegradationLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class Decision(Enum):
    ACCEPT = "ACCEPT"
    INVESTIGATE = "INVESTIGATE"
    REVISE = "REVISE METHOD"


@dataclass
class DegradationData:
    api_initial: float
    api_stressed: float
    degradants_initial: float
    degradants_stressed: float
    api_uncertainty: float = 1.5
    deg_uncertainty: float = 5.0
    lod: float = 0.05
    loq: float = 0.10
    stress_condition: str = "Not specified"
    drug_name: str = "API"
    
    @property
    def api_loss(self) -> float:
        return self.api_initial - self.api_stressed
    
    @property
    def api_loss_percent(self) -> float:
        return (self.api_loss / self.api_initial) * 100 if self.api_initial > 0 else 0
    
    @property
    def degradant_increase(self) -> float:
        return self.degradants_stressed - self.degradants_initial
    
    @property
    def degradation_level(self) -> DegradationLevel:
        loss_pct = self.api_loss_percent
        if loss_pct < 5:
            return DegradationLevel.LOW
        elif loss_pct < 15:
            return DegradationLevel.MODERATE
        elif loss_pct < 30:
            return DegradationLevel.HIGH
        else:
            return DegradationLevel.EXTREME
    
    @property
    def total_initial(self) -> float:
        return self.api_initial + self.degradants_initial
    
    @property
    def total_stressed(self) -> float:
        return self.api_stressed + self.degradants_stressed


@dataclass
class MassBalanceResult:
    method_name: str
    value: float
    formula: str
    is_acceptable: bool
    threshold: float
    method_type: str = "conventional"


# ============================================================================
# CONVENTIONAL MASS BALANCE METHODS
# ============================================================================

class ConventionalMB:
    
    @staticmethod
    def simple_mb(data: DegradationData) -> MassBalanceResult:
        value = data.api_stressed + data.degradants_stressed
        return MassBalanceResult(
            method_name="Simple Mass Balance",
            value=round(value, 2),
            formula="SMB = API_stressed + Degradants_stressed",
            is_acceptable=value >= 95.0,
            threshold=95.0,
            method_type="conventional"
        )
    
    @staticmethod
    def absolute_mb(data: DegradationData) -> MassBalanceResult:
        value = (data.total_stressed / data.total_initial) * 100 if data.total_initial > 0 else 0
        return MassBalanceResult(
            method_name="Absolute Mass Balance",
            value=round(value, 2),
            formula="AMB = (Stressed Total / Initial Total) × 100",
            is_acceptable=value >= 95.0,
            threshold=95.0,
            method_type="conventional"
        )
    
    @staticmethod
    def absolute_mb_deficiency(data: DegradationData) -> MassBalanceResult:
        amb = ConventionalMB.absolute_mb(data).value
        value = 100 - amb
        return MassBalanceResult(
            method_name="Absolute MB Deficiency",
            value=round(value, 2),
            formula="AMBD = 100 - AMB",
            is_acceptable=value <= 5.0,
            threshold=5.0,
            method_type="conventional"
        )
    
    @staticmethod
    def relative_mb(data: DegradationData) -> MassBalanceResult:
        if data.api_loss <= 0:
            value = 100.0
        else:
            value = (data.degradant_increase / data.api_loss) * 100
        return MassBalanceResult(
            method_name="Relative Mass Balance",
            value=round(value, 2),
            formula="RMB = (ΔDegradants / ΔAPI) × 100",
            is_acceptable=80.0 <= value <= 120.0,
            threshold=80.0,
            method_type="conventional"
        )
    
    @staticmethod
    def relative_mb_deficiency(data: DegradationData) -> MassBalanceResult:
        rmb = ConventionalMB.relative_mb(data).value
        value = max(0, 100 - min(rmb, 100))
        return MassBalanceResult(
            method_name="Relative MB Deficiency",
            value=round(value, 2),
            formula="RMBD = 100 - RMB",
            is_acceptable=value <= 20.0,
            threshold=20.0,
            method_type="conventional"
        )
    
    @classmethod
    def calculate_all(cls, data: DegradationData) -> Dict[str, MassBalanceResult]:
        return {
            'SMB': cls.simple_mb(data),
            'AMB': cls.absolute_mb(data),
            'AMBD': cls.absolute_mb_deficiency(data),
            'RMB': cls.relative_mb(data),
            'RMBD': cls.relative_mb_deficiency(data)
        }


# ============================================================================
# PRISM ENHANCED METHODS
# ============================================================================

class PRISMMB:
    
    @staticmethod
    def get_rf_estimate(degradation_level: DegradationLevel, stress_type: str = "general") -> Tuple[float, float]:
        stress_rf = {
            "Oxidation (H2O2)": (0.70, 0.15),
            "Acid Hydrolysis": (0.82, 0.12),
            "Base Hydrolysis": (0.78, 0.14),
            "Thermal": (0.80, 0.12),
            "Photolysis": (0.65, 0.18),
            "Humidity": (0.85, 0.10),
            "Other": (0.75, 0.15)
        }
        
        base_rf, base_std = stress_rf.get(stress_type, (0.75, 0.15))
        
        level_adjustment = {
            DegradationLevel.LOW: 0.05,
            DegradationLevel.MODERATE: 0.0,
            DegradationLevel.HIGH: -0.05,
            DegradationLevel.EXTREME: -0.10
        }
        
        adjusted_rf = base_rf + level_adjustment.get(degradation_level, 0)
        return (adjusted_rf, base_std)
    
    @staticmethod
    def rf_corrected_mb(data: DegradationData, custom_rf: Optional[float] = None, 
                        stress_type: str = "general") -> MassBalanceResult:
        if custom_rf:
            avg_rf = custom_rf
        else:
            avg_rf, _ = PRISMMB.get_rf_estimate(data.degradation_level, stress_type)
        
        corrected_deg = data.degradants_stressed / avg_rf
        corrected_total = data.api_stressed + corrected_deg
        value = (corrected_total / data.total_initial) * 100 if data.total_initial > 0 else 0
        
        return MassBalanceResult(
            method_name=f"RF-Corrected MB (RF={avg_rf:.2f})",
            value=round(value, 2),
            formula=f"RFCMB = (API + Deg/RF) / Initial × 100",
            is_acceptable=value >= 95.0,
            threshold=95.0,
            method_type="prism"
        )
    
    @staticmethod
    def get_weights(degradation_level: DegradationLevel) -> Tuple[float, float, float]:
        weights = {
            DegradationLevel.LOW: (0.50, 0.10, 0.40),
            DegradationLevel.MODERATE: (0.30, 0.30, 0.40),
            DegradationLevel.HIGH: (0.20, 0.40, 0.40),
            DegradationLevel.EXTREME: (0.15, 0.45, 0.40)
        }
        return weights[degradation_level]
    
    @staticmethod
    def weighted_composite_mb(data: DegradationData, custom_rf: Optional[float] = None,
                              stress_type: str = "general") -> MassBalanceResult:
        amb = ConventionalMB.absolute_mb(data).value
        rmb = min(100, max(0, ConventionalMB.relative_mb(data).value))
        rfcmb = PRISMMB.rf_corrected_mb(data, custom_rf, stress_type).value
        
        w1, w2, w3 = PRISMMB.get_weights(data.degradation_level)
        value = w1 * amb + w2 * rmb + w3 * rfcmb
        
        return MassBalanceResult(
            method_name="Weighted Composite MB",
            value=round(value, 2),
            formula=f"WCMB = {w1:.2f}×AMB + {w2:.2f}×RMB + {w3:.2f}×RFCMB",
            is_acceptable=value >= 95.0,
            threshold=95.0,
            method_type="prism"
        )
    
    @staticmethod
    def detection_adjusted_mb(data: DegradationData, custom_rf: Optional[float] = None,
                              volatile_est: float = 1.0, non_chromo_est: float = 1.5,
                              stress_type: str = "general") -> MassBalanceResult:
        rfcmb = PRISMMB.rf_corrected_mb(data, custom_rf, stress_type).value
        total_adjustment = volatile_est + non_chromo_est
        value = min(100, rfcmb + total_adjustment)
        
        return MassBalanceResult(
            method_name="Detection-Adjusted MB",
            value=round(value, 2),
            formula=f"DAMB = RFCMB + {total_adjustment:.1f}%",
            is_acceptable=value >= 95.0,
            threshold=95.0,
            method_type="prism"
        )
    
    @classmethod
    def calculate_all(cls, data: DegradationData, custom_rf: Optional[float] = None,
                      volatile: float = 1.0, non_chromo: float = 1.5,
                      stress_type: str = "general") -> Dict[str, MassBalanceResult]:
        return {
            'RFCMB': cls.rf_corrected_mb(data, custom_rf, stress_type),
            'WCMB': cls.weighted_composite_mb(data, custom_rf, stress_type),
            'DAMB': cls.detection_adjusted_mb(data, custom_rf, volatile, non_chromo, stress_type)
        }


# ============================================================================
# MONTE CARLO ENGINE
# ============================================================================

class MonteCarloEngine:
    
    @staticmethod
    @st.cache_data(ttl=300)
    def run_simulation(
        api_initial: float,
        api_stressed: float,
        deg_initial: float,
        deg_stressed: float,
        api_uncertainty: float,
        deg_uncertainty: float,
        avg_rf: float,
        rf_std: float,
        n_simulations: int = 10000
    ) -> Tuple:
        
        np.random.seed(42)
        results = []
        
        for _ in range(n_simulations):
            api_init_s = np.random.normal(api_initial, api_initial * api_uncertainty / 100)
            api_stress_s = np.random.normal(api_stressed, api_stressed * api_uncertainty / 100)
            deg_init_s = max(0, np.random.normal(deg_initial, max(0.05, deg_initial * deg_uncertainty / 100)))
            deg_stress_s = max(0, np.random.normal(deg_stressed, deg_stressed * deg_uncertainty / 100))
            rf_s = np.clip(np.random.normal(avg_rf, rf_std), 0.1, 2.0)
            
            corrected_deg = deg_stress_s / rf_s
            initial_total = api_init_s + deg_init_s
            stressed_total = api_stress_s + corrected_deg
            mb = (stressed_total / initial_total) * 100 if initial_total > 0 else 0
            results.append(mb)
        
        results = np.array(results)
        
        return (
            round(np.mean(results), 2),
            round(np.std(results), 2),
            round(np.percentile(results, 2.5), 2),
            round(np.percentile(results, 97.5), 2),
            round(np.mean(results >= 95), 3),
            round(np.mean(results >= 90), 3),
            round(np.mean(results >= 85), 3),
            round(np.mean(results >= 80), 3),
            results,
            n_simulations
        )


# ============================================================================
# DECISION ENGINE
# ============================================================================

class DecisionEngine:
    
    @staticmethod
    def make_decision(p_above_95: float, p_above_90: float, p_above_85: float) -> Tuple[Decision, str, List[str]]:
        
        if p_above_95 >= 0.80 or p_above_90 >= 0.95:
            decision = Decision.ACCEPT
            rationale = f"High confidence in mass balance (P(MB≥95%) = {p_above_95:.1%})"
            recommendations = [
                "✅ Mass balance acceptable for stability-indicating method claim",
                "✅ Document PRISM analysis as scientific justification in stability report",
                "💡 Consider confirming response factors for key degradants to strengthen documentation"
            ]
        
        elif p_above_90 >= 0.50 and p_above_85 >= 0.80:
            decision = Decision.INVESTIGATE
            rationale = f"Moderate confidence (P(MB≥90%) = {p_above_90:.1%}). Investigation recommended."
            recommendations = [
                "🔍 Confirm response factors for major degradation peaks using reference standards",
                "🔍 Run LC-MS analysis to identify unknown peaks",
                "🔍 Check for volatile degradation products (headspace GC-MS)",
                "🔍 Review degradation pathway for expected products not detected",
                "🔍 Consider CAD/ELSD detection for non-UV active species"
            ]
        
        else:
            decision = Decision.REVISE
            rationale = f"Low confidence (P(MB≥85%) = {p_above_85:.1%}). Method revision needed."
            recommendations = [
                "⚠️ Significant mass balance gap detected - method revision required",
                "🔧 Extend chromatographic run time to check for late-eluting peaks",
                "🔧 Use alternative detection methods (MS, CAD, ELSD)",
                "🔧 Investigate sample preparation losses (extraction, filtration)",
                "🔧 Check for precipitation or adsorption during stress testing",
                "🔧 Verify stress conditions are not causing secondary reactions"
            ]
        
        return decision, rationale, recommendations


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_method_comparison_chart(conventional: Dict, prism: Dict) -> go.Figure:
    methods = ['SMB', 'AMB', 'RMB', 'RFCMB', 'WCMB', 'DAMB']
    values = [
        conventional['SMB'].value,
        conventional['AMB'].value,
        min(100, conventional['RMB'].value),
        prism['RFCMB'].value,
        prism['WCMB'].value,
        prism['DAMB'].value
    ]
    
    colors = ['#EF4444', '#EF4444', '#EF4444', '#10B981', '#10B981', '#10B981']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=methods,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_hline(y=95, line_dash="dash", line_color="green", line_width=2,
                  annotation_text="95% Target", annotation_position="right")
    fig.add_hline(y=90, line_dash="dash", line_color="orange", line_width=2,
                  annotation_text="90% Acceptable", annotation_position="right")
    fig.add_hline(y=85, line_dash="dot", line_color="red", line_width=1,
                  annotation_text="85% Minimum", annotation_position="right")
    
    fig.update_layout(
        title=dict(text="<b>Mass Balance Method Comparison</b>", font_size=20),
        xaxis_title="Method",
        yaxis_title="Mass Balance (%)",
        yaxis_range=[0, 115],
        showlegend=False,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.add_annotation(x=1, y=108, text="<b>Conventional</b>", showarrow=False,
                      font=dict(size=14, color='#EF4444'))
    fig.add_annotation(x=4, y=108, text="<b>PRISM</b>", showarrow=False,
                      font=dict(size=14, color='#10B981'))
    
    return fig


def create_monte_carlo_distribution(distribution: np.ndarray, mean: float, 
                                    ci_lower: float, ci_upper: float) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=distribution,
        nbinsx=60,
        name='Distribution',
        marker_color='#3B82F6',
        opacity=0.7
    ))
    
    fig.add_vrect(x0=ci_lower, x1=ci_upper, fillcolor="rgba(59, 130, 246, 0.2)",
                  line_width=0, annotation_text="95% CI", annotation_position="top left")
    
    fig.add_vline(x=95, line_dash="dash", line_color="green", line_width=3,
                  annotation_text="95%", annotation_position="top")
    fig.add_vline(x=90, line_dash="dash", line_color="orange", line_width=2,
                  annotation_text="90%", annotation_position="top")
    fig.add_vline(x=mean, line_dash="solid", line_color="red", line_width=3,
                  annotation_text=f"Mean: {mean}%", annotation_position="top")
    
    fig.update_layout(
        title=dict(text="<b>Monte Carlo Probability Distribution</b>", font_size=18),
        xaxis_title="Mass Balance (%)",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    return fig


def create_probability_gauges(p95: float, p90: float, p85: float) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        horizontal_spacing=0.1
    )
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=p95 * 100,
        number={'suffix': '%', 'font': {'size': 24}},
        title={'text': "P(MB ≥ 95%)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#10B981" if p95 >= 0.8 else ("#F59E0B" if p95 >= 0.5 else "#EF4444")},
            'steps': [
                {'range': [0, 50], 'color': "#FEE2E2"},
                {'range': [50, 80], 'color': "#FEF3C7"},
                {'range': [80, 100], 'color': "#D1FAE5"}
            ],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.8, 'value': 80}
        }
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=p90 * 100,
        number={'suffix': '%', 'font': {'size': 24}},
        title={'text': "P(MB ≥ 90%)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#10B981" if p90 >= 0.5 else "#F59E0B"},
            'steps': [
                {'range': [0, 50], 'color': "#FEE2E2"},
                {'range': [50, 80], 'color': "#FEF3C7"},
                {'range': [80, 100], 'color': "#D1FAE5"}
            ]
        }
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=p85 * 100,
        number={'suffix': '%', 'font': {'size': 24}},
        title={'text': "P(MB ≥ 85%)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#10B981" if p85 >= 0.8 else "#F59E0B"},
            'steps': [
                {'range': [0, 50], 'color': "#FEE2E2"},
                {'range': [50, 80], 'color': "#FEF3C7"},
                {'range': [80, 100], 'color': "#D1FAE5"}
            ]
        }
    ), row=1, col=3)
    
    fig.update_layout(height=280, margin=dict(t=50, b=20, l=20, r=20))
    
    return fig


def create_improvement_waterfall(conventional: Dict, prism: Dict) -> go.Figure:
    amb = conventional['AMB'].value
    rfcmb = prism['RFCMB'].value
    damb = prism['DAMB'].value
    
    rf_improvement = rfcmb - amb
    detection_improvement = damb - rfcmb
    
    fig = go.Figure(go.Waterfall(
        name="Mass Balance",
        orientation="v",
        x=["Conventional<br>(AMB)", "RF<br>Correction", "Detection<br>Adjustment", "PRISM<br>(DAMB)"],
        y=[amb, rf_improvement, detection_improvement, 0],
        measure=["absolute", "relative", "relative", "total"],
        text=[f"{amb:.1f}%", f"+{rf_improvement:.1f}%", f"+{detection_improvement:.1f}%", f"{damb:.1f}%"],
        textposition="outside",
        textfont=dict(size=14, family='Arial Black'),
        connector={"line": {"color": "#374151", "width": 2}},
        increasing={"marker": {"color": "#10B981"}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#3B82F6"}}
    ))
    
    fig.add_hline(y=95, line_dash="dash", line_color="green", line_width=2,
                  annotation_text="95% Target", annotation_position="right")
    
    total_improvement = damb - amb
    fig.add_annotation(
        x=2.5, y=max(amb, damb) + 5,
        text=f"<b>Total Improvement: +{total_improvement:.1f}%</b>",
        showarrow=False,
        font=dict(size=14, color="#10B981"),
        bgcolor="white",
        bordercolor="#10B981",
        borderwidth=2
    )
    
    fig.update_layout(
        title=dict(text="<b>PRISM-MB Improvement Breakdown</b>", font_size=18),
        yaxis_title="Mass Balance (%)",
        height=450,
        showlegend=False,
        yaxis_range=[0, max(110, damb + 10)]
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🔬 PRISM-MB Calculator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Probabilistic Response-factor Informed Structural Mass Balance Framework</p>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#9CA3AF; font-size:0.9rem;">Developed by Aryan Ranjan | VIT Bhopal University | Innovation Challenge 2024</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Input Parameters")
        
        st.subheader("📋 Sample Information")
        drug_name = st.text_input("Drug Name", value="Sample API")
        stress_condition = st.selectbox(
            "Stress Condition",
            ["Oxidation (H2O2)", "Acid Hydrolysis", "Base Hydrolysis", 
             "Thermal", "Photolysis", "Humidity", "Other"]
        )
        
        st.markdown("---")
        
        st.subheader("🧪 Initial Sample")
        api_initial = st.number_input("API Initial (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
        deg_initial = st.number_input("Degradants Initial (%)", min_value=0.0, max_value=50.0, value=0.5, step=0.1)
        
        st.subheader("⚗️ Stressed Sample")
        api_stressed = st.number_input("API Stressed (%)", min_value=0.0, max_value=100.0, value=82.5, step=0.1)
        deg_stressed = st.number_input("Degradants Stressed (%)", min_value=0.0, max_value=100.0, value=4.9, step=0.1)
        
        st.markdown("---")
        
        st.subheader("⚙️ Advanced Settings")
        with st.expander("Click to expand", expanded=False):
            custom_rf = st.slider("Response Factor (RF)", min_value=0.30, max_value=1.50, value=0.75, step=0.05)
            volatile_est = st.slider("Volatile Loss Estimate (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
            non_chromo_est = st.slider("Non-chromophoric Estimate (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.25)
            n_simulations = st.select_slider("Monte Carlo Iterations", options=[1000, 5000, 10000, 25000], value=10000)
            api_uncertainty = st.slider("API Measurement RSD (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
            deg_uncertainty = st.slider("Degradant Measurement RSD (%)", min_value=1.0, max_value=15.0, value=5.0, step=1.0)
        
        st.markdown("---")
        
        analyze_button = st.button("🚀 Run PRISM Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.subheader("📚 Example Data")
        if st.button("Load Problem Statement Example", use_container_width=True):
            st.session_state.load_example = True
            st.rerun()
    
    # Check for example load
    if 'load_example' in st.session_state and st.session_state.load_example:
        api_initial = 98.0
        api_stressed = 82.5
        deg_initial = 0.5
        deg_stressed = 4.9
        st.session_state.load_example = False
        analyze_button = True
    
    # Main Content
    if analyze_button:
        if api_initial <= 0:
            st.error("⚠️ API Initial must be greater than 0")
            return
        
        data = DegradationData(
            api_initial=api_initial,
            api_stressed=api_stressed,
            degradants_initial=deg_initial,
            degradants_stressed=deg_stressed,
            api_uncertainty=api_uncertainty,
            deg_uncertainty=deg_uncertainty,
            drug_name=drug_name,
            stress_condition=stress_condition
        )
        
        conventional = ConventionalMB.calculate_all(data)
        prism = PRISMMB.calculate_all(data, custom_rf, volatile_est, non_chromo_est, stress_condition)
        
        avg_rf, rf_std = PRISMMB.get_rf_estimate(data.degradation_level, stress_condition)
        if custom_rf:
            avg_rf = custom_rf
            rf_std = 0.10
        
        with st.spinner("🔄 Running Monte Carlo simulation..."):
            mc_results = MonteCarloEngine.run_simulation(
                api_initial, api_stressed, deg_initial, deg_stressed,
                api_uncertainty, deg_uncertainty, avg_rf, rf_std, n_simulations
            )
        
        mean, std, ci_lower, ci_upper, p95, p90, p85, p80, distribution, n_sims = mc_results
        decision, rationale, recommendations = DecisionEngine.make_decision(p95, p90, p85)
        
        # Display Results
        st.header("📈 Analysis Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("API Loss", f"{data.api_loss:.1f}%", delta=f"-{data.api_loss_percent:.1f}%", delta_color="inverse")
        
        with col2:
            level_colors = {"low": "🟢", "moderate": "🟡", "high": "🟠", "extreme": "🔴"}
            st.metric("Degradation Level", f"{level_colors.get(data.degradation_level.value, '')} {data.degradation_level.value.upper()}")
        
        with col3:
            st.metric("Conventional (AMB)", f"{conventional['AMB'].value:.1f}%")
        
        with col4:
            improvement = prism['RFCMB'].value - conventional['AMB'].value
            st.metric("PRISM (RFCMB)", f"{prism['RFCMB'].value:.1f}%", delta=f"+{improvement:.1f}%")
        
        with col5:
            decision_icons = {Decision.ACCEPT: "✅", Decision.INVESTIGATE: "⚠️", Decision.REVISE: "❌"}
            st.metric("Decision", f"{decision_icons.get(decision, '')} {decision.value}")
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Method Comparison", "📈 Uncertainty Analysis", "🎯 Decision",
            "🔍 Missing Mass", "📋 Full Report"
        ])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_method_comparison_chart(conventional, prism)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Conventional Methods")
                for name, result in conventional.items():
                    status = "✅" if result.is_acceptable else "❌"
                    st.markdown(f"{status} **{name}**: {result.value}%")
                
                st.markdown("### PRISM Methods")
                for name, result in prism.items():
                    status = "✅" if result.is_acceptable else "⚠️"
                    st.markdown(f"{status} **{name}**: {result.value}%")
            
            st.subheader("PRISM Improvement Breakdown")
            fig_waterfall = create_improvement_waterfall(conventional, prism)
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with tab2:
            st.subheader("Monte Carlo Uncertainty Quantification")
            st.info(f"📊 Based on {n_sims:,} Monte Carlo simulations")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig_dist = create_monte_carlo_distribution(distribution, mean, ci_lower, ci_upper)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.markdown("### Statistical Summary")
                st.dataframe(pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', '95% CI Lower', '95% CI Upper'],
                    'Value': [f"{mean}%", f"±{std}%", f"{ci_lower}%", f"{ci_upper}%"]
                }), hide_index=True, use_container_width=True)
                
                st.markdown("### Probability Thresholds")
                st.dataframe(pd.DataFrame({
                    'Threshold': ['P(MB ≥ 95%)', 'P(MB ≥ 90%)', 'P(MB ≥ 85%)', 'P(MB ≥ 80%)'],
                    'Probability': [f"{p95:.1%}", f"{p90:.1%}", f"{p85:.1%}", f"{p80:.1%}"]
                }), hide_index=True, use_container_width=True)
            
            st.subheader("Probability Gauges")
            fig_gauges = create_probability_gauges(p95, p90, p85)
            st.plotly_chart(fig_gauges, use_container_width=True)
        
        with tab3:
            if decision == Decision.ACCEPT:
                st.markdown(f'<div class="success-box"><h2>✅ Decision: {decision.value}</h2><p>{rationale}</p></div>', unsafe_allow_html=True)
            elif decision == Decision.INVESTIGATE:
                st.markdown(f'<div class="warning-box"><h2>⚠️ Decision: {decision.value}</h2><p>{rationale}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-box"><h2>❌ Decision: {decision.value}</h2><p>{rationale}</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📝 Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        with tab4:
            st.subheader("🔍 Missing Mass Hypothesis")
            
            total_missing = 100 - conventional['AMB'].value
            rf_explained = prism['RFCMB'].value - conventional['AMB'].value
            detection_explained = volatile_est + non_chromo_est
            remaining = max(0, 100 - prism['DAMB'].value)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_missing = go.Figure()
                categories = ['Total Missing', 'RF Correction', 'Detection Adj.', 'Remaining']
                values = [total_missing, rf_explained, detection_explained, remaining]
                colors = ['#EF4444', '#10B981', '#10B981', '#F59E0B']
                
                fig_missing.add_trace(go.Bar(x=categories, y=values, marker_color=colors,
                    text=[f'{v:.1f}%' for v in values], textposition='outside'))
                fig_missing.update_layout(height=350, yaxis_title="Percentage (%)")
                st.plotly_chart(fig_missing, use_container_width=True)
            
            with col2:
                st.markdown("### Hypothesis Details")
                st.dataframe(pd.DataFrame([
                    {"Component": "Total Missing (Conventional)", "Value": f"{total_missing:.2f}%"},
                    {"Component": "RF Correction", "Value": f"+{rf_explained:.2f}%"},
                    {"Component": "Volatile Products", "Value": f"+{volatile_est:.2f}%"},
                    {"Component": "Non-chromophoric", "Value": f"+{non_chromo_est:.2f}%"},
                    {"Component": "Remaining Gap", "Value": f"{remaining:.2f}%"}
                ]), hide_index=True, use_container_width=True)
        
        with tab5:
            st.subheader("📋 Complete Analysis Report")
            
            report_text = f"""
================================================================================
                         PRISM-MB ANALYSIS REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Software: PRISM-MB Calculator v1.0.0
Author: Aryan Ranjan | VIT Bhopal University

================================================================================
SAMPLE INFORMATION
================================================================================
Drug Name: {drug_name}
Stress Condition: {stress_condition}

================================================================================
INPUT DATA
================================================================================
Initial: API = {api_initial}%, Degradants = {deg_initial}%
Stressed: API = {api_stressed}%, Degradants = {deg_stressed}%
API Loss: {data.api_loss:.2f}% ({data.api_loss_percent:.2f}%)
Degradation Level: {data.degradation_level.value.upper()}

================================================================================
CONVENTIONAL METHODS
================================================================================
SMB: {conventional['SMB'].value}% ({'Acceptable' if conventional['SMB'].is_acceptable else 'Not Acceptable'})
AMB: {conventional['AMB'].value}% ({'Acceptable' if conventional['AMB'].is_acceptable else 'Not Acceptable'})
AMBD: {conventional['AMBD'].value}%
RMB: {conventional['RMB'].value}%
RMBD: {conventional['RMBD'].value}%

================================================================================
PRISM METHODS
================================================================================
RFCMB: {prism['RFCMB'].value}% (Improvement: +{prism['RFCMB'].value - conventional['AMB'].value:.2f}%)
WCMB: {prism['WCMB'].value}%
DAMB: {prism['DAMB'].value}%

================================================================================
UNCERTAINTY ANALYSIS (n={n_sims:,})
================================================================================
Mean: {mean}% ± {std}%
95% CI: [{ci_lower}%, {ci_upper}%]
P(MB ≥ 95%): {p95:.1%}
P(MB ≥ 90%): {p90:.1%}
P(MB ≥ 85%): {p85:.1%}

================================================================================
DECISION: {decision.value}
================================================================================
{rationale}

Recommendations:
{chr(10).join(['- ' + r for r in recommendations])}

================================================================================
Generated by PRISM-MB Calculator | Innovation Challenge 2024
Author: Aryan Ranjan | VIT Bhopal University
================================================================================
            """
            
            st.text_area("Report", report_text, height=500)
            
            st.download_button(
                label="📥 Download Report (.txt)",
                data=report_text,
                file_name=f"PRISM_MB_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to PRISM-MB Calculator!</h3>
            <p>Enter your forced degradation data in the sidebar and click <b>Run PRISM Analysis</b> to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 What is Mass Balance?")
            st.markdown("""
            Mass Balance verifies that all components (API + degradants) are properly 
            accounted for during stress testing.
            
            **The Problem:** Current methods often show MB < 95% because they don't account for:
            - Response factor differences
            - Measurement uncertainty
            - Non-chromophoric products
            """)
        
        with col2:
            st.subheader("💡 PRISM Innovation")
            st.markdown("""
            **PRISM-MB** introduces:
            
            1. **Response Factor Correction** - Accounts for UV absorption differences
            2. **Uncertainty Quantification** - Monte Carlo simulation
            3. **Risk-Based Decisions** - Probabilistic thresholds
            """)
        
        st.info("👈 Click **'Load Problem Statement Example'** in the sidebar to see PRISM-MB in action!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 1rem;">
        <p><strong>PRISM-MB Calculator</strong> | Version 1.0.0</p>
        <p>Developed by <strong>Aryan Ranjan</strong> | B.Tech CSE | VIT Bhopal University</p>
        <p>Innovation Challenge 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()