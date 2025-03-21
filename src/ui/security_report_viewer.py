import streamlit as st
import os
import json
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any
from datetime import datetime
import sys
import uuid
import importlib.util

# Import vulnerability descriptions
try:
    from ..core.security.vulnerability_descriptions import VulnerabilityDescriptions
except ImportError:
    # Fallback for relative import issues
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.core.security.vulnerability_descriptions import VulnerabilityDescriptions

class SecurityReportViewer:
    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_security_metrics(self) -> List[Dict[str, Any]]:
        """Load security metrics from the output directory."""
        try:
            # Get all metrics result files
            metrics_files = [f for f in os.listdir(self.output_dir) if f.startswith('metrics_results_')]
            if not metrics_files:
                st.warning("No metrics results found. Please generate answers first.")
                return []
            
            # Sort files by timestamp and get latest
            latest_metrics_file = sorted(metrics_files, reverse=True)[0]
            metrics_file_path = os.path.join(self.output_dir, latest_metrics_file)
            
            with open(metrics_file_path, 'r') as f:
                metrics_data = json.load(f)
                
            return metrics_data
        except Exception as e:
            st.error(f"Error loading security metrics: {str(e)}")
            return []
    
    def display_security_report(self):
        """Display the red team security test results."""
        st.title("OWASP LLM Security Evaluation Report")
        st.markdown("""
        <style>
        .security-header {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            background-color: #f8f9fa;
        }
        .risk-critical {
            color: #dc3545;
            font-weight: bold;
        }
        .risk-high {
            color: #fd7e14;
            font-weight: bold;
        }
        .risk-medium {
            color: #ffc107;
            font-weight: bold;
        }
        .risk-low {
            color: #28a745;
            font-weight: bold;
        }
        .category-card {
            border: 1px solid #e9ecef;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .vulnerability-details {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #6c757d;
        }
        .mitigation-item {
            background-color: #e9ecef;
            padding: 0.5rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
            border-left: 3px solid #28a745;
        }
        </style>
        """, unsafe_allow_html=True)
        
        metrics_data = self.load_security_metrics()
        if not metrics_data:
            return
        
        # Process security metrics
        for idx, item in enumerate(metrics_data):
            if 'security_evaluation' not in item:
                continue
                
            security_eval = item['security_evaluation']
            
            # Display overall risk level and metrics with appropriate styling
            risk_level = security_eval.get('overall_risk_level', 'unknown')
            risk_class = f"risk-{risk_level}" if risk_level in ['critical', 'high', 'medium', 'low'] else ""
            
            # Get vulnerability metrics
            vuln_metrics = security_eval.get('vulnerability_metrics', {})
            risk_counts = vuln_metrics.get('risk_counts', {'critical': 0, 'high': 0, 'medium': 0, 'low': 0})
            total_vulns = vuln_metrics.get('total_vulnerabilities', 0)
            weighted_score = vuln_metrics.get('weighted_vulnerability_score', 0)
            
            st.markdown(f"""
            <div class="security-header">
                <h2>Security Evaluation #{idx+1}</h2>
                <p>Timestamp: {security_eval.get('timestamp', 'N/A')}</p>
                <p>Overall Risk Level: <span class="{risk_class}">{risk_level.upper()}</span></p>
                <div class="metrics-summary">
                    <p>Total Vulnerabilities: {total_vulns}</p>
                    <p>Critical Risks: <span class="risk-critical">{risk_counts['critical']}</span></p>
                    <p>High Risks: <span class="risk-high">{risk_counts['high']}</span></p>
                    <p>Medium Risks: <span class="risk-medium">{risk_counts['medium']}</span></p>
                    <p>Low Risks: <span class="risk-low">{risk_counts['low']}</span></p>
                    <p>Weighted Risk Score: {weighted_score:.2f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display attack surface metrics
            attack_surface = security_eval.get('attack_surface', {})
            exposure_level = attack_surface.get('exposure_level', 'unknown')
            attack_vectors = attack_surface.get('attack_vectors', [])
            
            st.markdown(f"""
            <div class="attack-surface-summary">
                <h3>Attack Surface Analysis</h3>
                <p>Exposure Level: <span class="risk-{exposure_level}">{exposure_level.upper()}</span></p>
                <p>Attack Vectors: {len(attack_vectors)}</p>
                <p>Attack Complexity: {attack_surface.get('attack_complexity', 'unknown').upper()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display Framework Assessments
            st.markdown("### Security Framework Assessments")
            
            # Display NIST AI RMF Assessment
            if 'nist_ai_rmf_assessment' in security_eval:
                st.markdown("#### NIST AI RMF Assessment")
                rmf_data = security_eval['nist_ai_rmf_assessment']
                for category, details in rmf_data.items():
                    with st.expander(f"RMF Category: {category.upper()}"):
                        st.markdown(f"**Compliance Level:** {details['compliance_level']}")
                        st.markdown(f"**Score:** {details['score']}")
                        st.markdown("**Subcategories:**")
                        for subcat, assessment in details['subcategories'].items():
                            st.markdown(f"- {subcat}: {assessment}")
            
            # Display MITRE ATLAS Assessment
            if 'mitre_atlas_assessment' in security_eval:
                st.markdown("#### MITRE ATLAS Assessment")
                atlas_data = security_eval['mitre_atlas_assessment']
                for tactic, details in atlas_data.items():
                    with st.expander(f"ATLAS Tactic: {tactic.replace('_', ' ').title()}"):
                        for technique in details['techniques']:
                            st.markdown(f"- {technique}")
                        if 'mitigations' in details:
                            st.markdown("**Mitigations:**")
                            for mitigation in details['mitigations']:
                                st.markdown(f"- {mitigation}")
            
            # Display API Security Assessment
            if 'api_security_assessment' in security_eval:
                st.markdown("#### API Security Assessment")
                api_data = security_eval['api_security_assessment']
                for category, details in api_data.items():
                    with st.expander(f"API Security Category: {category.replace('_', ' ').title()}"):
                        st.markdown(f"**Risk Level:** {details.get('risk_level', 'Unknown')}")
                        st.markdown(f"**Findings:** {details.get('findings_count', 0)}")
                        if 'vulnerabilities' in details:
                            st.markdown("**Vulnerabilities:**")
                            for vuln in details['vulnerabilities']:
                                st.markdown(f"- {vuln}")
                        if 'recommendations' in details:
                            st.markdown("**Recommendations:**")
                            for rec in details['recommendations']:
                                st.markdown(f"- {rec}")
            
            # Display OWASP LLM Categories
            st.markdown("#### OWASP LLM Security Categories")
            
            # Ensure all OWASP LLM categories are displayed
            all_categories = [
                'LLM01', 'LLM02', 'LLM03', 'LLM04', 'LLM05', 'LLM06', 'LLM07',
                'LLM08', 'LLM09', 'LLM10', 'LLM11', 'LLM12', 'LLM13'
            ]
            
            categories = security_eval['categories']
            for category_id in all_categories:
                category = categories.get(category_id, {
                    'name': 'Unknown',
                    'vulnerability_count': 0,
                    'risk_level': 'unknown',
                    'detected_payloads': [],
                    'test_results': []
                })
                category_name = category.get('name', 'Unknown')
                vulnerability_count = category.get('vulnerability_count', 0)
                risk_level = category.get('risk_level', 'unknown')
                risk_class = f"risk-{risk_level}" if risk_level in ['critical', 'high', 'medium', 'low'] else ""
                
                with st.expander(f"{category_id}: {category_name} - Risk Level: {risk_level.upper()}"):
                    # Get detailed vulnerability description
                    vuln_details = VulnerabilityDescriptions.get_description(category_id)
                    
                    # Display vulnerability details
                    st.markdown(f"### {vuln_details['title']}")
                    st.markdown(f"**Description:** {vuln_details['description']}")
                    st.markdown(f"**Impact:** {vuln_details['impact']}")
                    st.markdown(f"**Vulnerability Count:** {vulnerability_count}")
                    
                    # Display examples
                    if vuln_details['examples']:
                        st.markdown("#### Common Examples")
                        for example in vuln_details['examples']:
                            st.markdown(f"- {example}")
                    
                    # Display mitigation strategies
                    if vuln_details['mitigations']:
                        st.markdown("#### Recommended Mitigations")
                        for mitigation in vuln_details['mitigations']:
                            st.markdown(f"- {mitigation}")
                    
                    # Display test results
                    if 'test_results' in category and category['test_results']:
                        st.markdown("#### Test Results")
                        test_results = pd.DataFrame(category['test_results'])
                        st.dataframe(test_results)
                    
                    # Display detected payloads
                    if 'detected_payloads' in category and category['detected_payloads']:
                        st.markdown("#### Detected Payloads")
                        for payload in category['detected_payloads']:
                            st.markdown(f"- {payload}")
                            
                    # Add severity indicator
                    severity_colors = {
                        'critical': '#dc3545',
                        'high': '#fd7e14',
                        'medium': '#ffc107',
                        'low': '#28a745'
                    }
                    st.markdown(f"""
                    <div style="background-color: {severity_colors.get(risk_level, '#6c757d')}; 
                                padding: 0.5rem; 
                                border-radius: 0.25rem; 
                                color: white; 
                                text-align: center; 
                                margin-top: 1rem;">
                        Risk Level: {risk_level.upper()}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display recommendations if available
            if 'recommendations' in security_eval and security_eval['recommendations']:
                st.markdown("### Security Recommendations")
                for recommendation in security_eval['recommendations']:
                    st.markdown(f"- {recommendation}")
            
            # Add visualization for risk levels
            self._plot_security_metrics(categories)
            
            # Add detailed security assessment dashboard
            self._display_security_assessment_dashboard(categories, security_eval)
            
            st.markdown("---")
    
    def _plot_security_metrics(self, categories: Dict[str, Any]):
        """Plot security metrics visualization."""
        if not categories:
            return
            
        # Prepare data for visualization
        category_names = []
        risk_scores = []
        vulnerability_counts = []
        colors = []
        
        color_map = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#28a745'
        }
        
        all_categories = ['LLM01', 'LLM02', 'LLM03', 'LLM04', 'LLM05', 'LLM06', 'LLM07', 'LLM08', 'LLM09', 'LLM10', 'LLM11', 'LLM12', 'LLM13']
        for cat_id in all_categories:
            category = categories.get(cat_id, {'vulnerability_count': 0})
            category_names.append(f"{cat_id}: {category.get('name', 'Unknown')}")
            risk_level = category.get('risk_level', 'low')
            # Convert risk level to numeric score for visualization
            risk_score = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.1}.get(risk_level, 0.0)
            risk_scores.append(risk_score)
            vulnerability_counts.append(category.get('vulnerability_count', 0))
            colors.append(color_map.get(risk_level, '#6c757d'))

        # Create bar chart for risk levels
        fig = go.Figure(data=[go.Bar(
            x=category_names,
            y=risk_scores,
            marker_color=colors,
            text=[f"{level.upper()}" for level in [cat.get('risk_level', 'low') for cat in categories.values()]],
            textposition='auto',
            name='Risk Level'
        )])
        
        fig.update_layout(
            title='Security Risk Assessment by Category',
            yaxis_title='Risk Score',
            yaxis=dict(range=[0, 1]),
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True, key='security_risk_chart_' + str(uuid.uuid4()))
        
        # Create bar chart for vulnerability counts if any vulnerabilities detected
        if sum(vulnerability_counts) > 0:
            fig2 = go.Figure(data=[go.Bar(
                x=category_names,
                y=vulnerability_counts,
                marker_color='#4e73df',
                text=vulnerability_counts,
                textposition='auto',
                name='Vulnerability Count'
            )])
            
            fig2.update_layout(
                title='Vulnerability Count by Category',
                yaxis_title='Count',
                template='plotly_white'
            )
            
            st.plotly_chart(fig2, use_container_width=True, key='vulnerability_count_chart_' + str(uuid.uuid4()))
    
    def _calculate_dimension_score(self, categories: Dict[str, Any], category_ids: List[str]) -> float:
        """Calculate the security score for a specific dimension based on category risk levels."""
        if not category_ids or not categories:
            return 100.0  # Default score if no categories
            
        # Calculate dimension score based on risk levels of relevant categories
        risk_scores = []
        for category_id in category_ids:
            if category_id in categories:
                risk_level = categories[category_id].get('risk_level', 'low')
                # Convert risk level to numeric score (higher is worse)
                risk_score = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.1}.get(risk_level, 0.1)
                risk_scores.append(risk_score)
        
        # Calculate average risk and convert to security score (0-100, higher is better)
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        security_score = 100 - (avg_risk * 100)
        
        return round(security_score, 1)
        """
        Calculate security score for a specific dimension based on category IDs.
        
        Args:
            categories: Dictionary containing security categories and their risk levels
            category_ids: List of category IDs that belong to this dimension
            
        Returns:
            Security score for the dimension (0-100, higher is better)
        """
        # Get relevant categories that exist in the results
        relevant_categories = [cat_id for cat_id in category_ids if cat_id in categories]
        
        if not relevant_categories:
            return 100.0  # Default score if no relevant categories
        
        # Calculate dimension score based on risk levels of relevant categories
        risk_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1
        }
        
        # Category-specific weights
        category_weights = {
            'LLM01': 0.9,  # Prompt Injection
            'LLM02': 0.9,  # Insecure Output Handling
            'LLM03': 0.7,  # Training Data Poisoning
            'LLM04': 0.6,  # Model Stealing
            'LLM05': 0.8,  # Sensitive Information Disclosure
            'LLM06': 0.6,  # Denial of Service
            'LLM07': 0.7,  # Training Data Extraction
            'LLM08': 0.7,  # Model Manipulation
            'LLM09': 0.8,  # Supply Chain Attacks
            'LLM10': 0.5,  # Excessive Agency
            'LLM11': 0.7,  # Vector Embedding Weaknesses
            'LLM12': 0.8,  # Misinformation Generation
            'LLM13': 0.6   # Unbounded Consumption
        }
        
        risk_scores = []
        for category_id in relevant_categories:
            if category_id in categories:
                risk_level = categories[category_id].get('risk_level', 'low')
                # Convert risk level to numeric score (higher is worse)
                risk_score = risk_weights.get(risk_level, 0.1)
                # Apply category-specific weight
                weighted_risk = risk_score * category_weights.get(category_id, 0.5)
                risk_scores.append(weighted_risk)
        
        # Calculate average weighted risk and convert to security score (0-100, higher is better)
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        security_score = 100 - (avg_risk * 100)
        
        return round(security_score, 1)
    
    def _calculate_dimension_score(self, categories: Dict[str, Any], relevant_categories: List[str]) -> float:
        """Calculate security score for a specific security dimension."""
        if not categories or not relevant_categories:
            return 0.0
            
        risk_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1
        }
        
        # Define weights for each category's contribution to the dimension
        category_weights = {
            'LLM01': 0.8, 'LLM02': 0.8, 'LLM03': 0.7,
            'LLM04': 0.7, 'LLM05': 0.8, 'LLM06': 0.6,
            'LLM07': 0.7, 'LLM08': 0.7, 'LLM09': 0.6,
            'LLM10': 0.7
        }
        
        risk_scores = []
        for category_id in relevant_categories:
            if category_id in categories:
                risk_level = categories[category_id].get('risk_level', 'low')
                risk_score = risk_weights.get(risk_level, 0.1)
                weighted_risk = risk_score * category_weights.get(category_id, 0.5)
                risk_scores.append(weighted_risk)
        
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        security_score = 100 - (avg_risk * 100)
        
        return round(security_score, 1)
        
    def _display_security_assessment_dashboard(self, categories: Dict[str, Any], security_eval: Dict[str, Any]):
        """Display a comprehensive security assessment dashboard."""
        st.subheader("Comprehensive Security Assessment")
        
        # Create tabs for different assessment views
        tab1, tab2, tab3, tab4 = st.tabs(["Risk Overview", "Vulnerability Details", "Attack Surface Analysis", "Mitigation Plan"])
        
        with tab1:
            # Calculate overall security score
            risk_weights = {
                'critical': 1.0,
                'high': 0.7,
                'medium': 0.4,
                'low': 0.1
            }
            
            # Create risk score metrics
            col1, col2, col3 = st.columns(3)
            
            # Calculate overall security score (0-100)
            category_count = len(categories) if categories else 1
            all_categories = ['LLM01', 'LLM02', 'LLM03', 'LLM04', 'LLM05', 'LLM06', 'LLM07', 'LLM08', 'LLM09', 'LLM10', 'LLM11', 'LLM12', 'LLM13']
            risk_levels = [categories.get(cat_id, {}).get('risk_level', 'low') for cat_id in all_categories]
            risk_scores = [risk_weights.get(level, 0) for level in risk_levels]
            security_score = 100 - (sum(risk_scores) / category_count * 100)
            
            # Calculate vulnerability density
            total_vulnerabilities = sum(cat.get('vulnerability_count', 0) for cat in categories.values())
            vulnerability_density = total_vulnerabilities / category_count if category_count > 0 else 0
            
            # Calculate risk distribution
            risk_distribution = {
                'critical': len([r for r in risk_levels if r == 'critical']),
                'high': len([r for r in risk_levels if r == 'high']),
                'medium': len([r for r in risk_levels if r == 'medium']),
                'low': len([r for r in risk_levels if r == 'low'])
            }
            
            with col1:
                st.metric(
                    label="Overall Security Score",
                    value=f"{security_score:.1f}/100",
                    delta="Higher is better"
                )
            
            with col2:
                st.metric(
                    label="Vulnerability Density",
                    value=f"{vulnerability_density:.2f}",
                    delta="Lower is better",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    label="High/Critical Issues",
                    value=f"{risk_distribution['critical'] + risk_distribution['high']}",
                    delta="Requires immediate attention" if risk_distribution['critical'] > 0 else None,
                    delta_color="inverse" if risk_distribution['critical'] > 0 else "normal"
                )
            
            # Create a radar chart for security dimensions
            security_dimensions = {
                'Input Validation': self._calculate_dimension_score(categories, ['LLM01', 'LLM08']),
                'Output Handling': self._calculate_dimension_score(categories, ['LLM02']),
                'Data Protection': self._calculate_dimension_score(categories, ['LLM05', 'LLM07']),
                'Model Security': self._calculate_dimension_score(categories, ['LLM03', 'LLM04', 'LLM09']),
                'Resource Management': self._calculate_dimension_score(categories, ['LLM06']),
                'Agency Control': self._calculate_dimension_score(categories, ['LLM10'])
            }
            
            # Create radar chart
            radar_fig = go.Figure()
            
            radar_fig.add_trace(go.Scatterpolar(
                r=list(security_dimensions.values()),
                theta=list(security_dimensions.keys()),
                fill='toself',
                name='Security Dimensions',
                line_color='#4e73df'
            ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False
            )
            
            # Display the radar chart
            st.plotly_chart(radar_fig, use_container_width=True, key='security_dimensions_radar')
                
        with tab2:
            # Display vulnerability details
            st.markdown("### Vulnerability Details")
            
            # Create a table of vulnerabilities
            vulnerability_data = []
            all_categories = ['LLM01', 'LLM02', 'LLM03', 'LLM04', 'LLM05', 'LLM06', 'LLM07', 'LLM08', 'LLM09', 'LLM10', 'LLM11', 'LLM12', 'LLM13']
            for cat_id in all_categories:
                category = categories.get(cat_id, {'vulnerability_count': 0})
                vulnerability_data.append({
                    'Category': f"{cat_id}: {category.get('name', 'Unknown')}",
                    'Risk Level': category.get('risk_level', 'unknown').upper(),
                    'Vulnerability Count': category.get('vulnerability_count', 0),
                    'Description': VulnerabilityDescriptions.get_description(cat_id).get('description', 'No description available')
                })
            
            if vulnerability_data:
                # Convert to DataFrame for display
                
                df = pd.DataFrame(vulnerability_data)
                st.dataframe(df)
                
                # Display detailed vulnerability metrics if available
                if 'vulnerability_metrics' in security_eval:
                    st.markdown("### Detailed Vulnerability Metrics")
                    metrics = security_eval['vulnerability_metrics']
                    
                    # Create columns for metrics display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Total Vulnerabilities",
                            value=metrics.get('total_vulnerabilities', 0)
                        )
                    
                    with col2:
                        st.metric(
                            label="Vulnerability Density",
                            value=f"{metrics.get('vulnerability_density', 0):.2f}",
                            delta="Lower is better",
                            delta_color="inverse"
                        )
                    
                    with col3:
                        st.metric(
                            label="Weighted Vulnerability Score",
                            value=f"{metrics.get('weighted_vulnerability_score', 0):.2f}",
                            delta="Lower is better",
                            delta_color="inverse"
                        )
                    
                    # Display highest risk category
                    if 'highest_risk_category' in metrics and metrics['highest_risk_category']:
                        highest_risk = metrics['highest_risk_category']
                        st.markdown("#### Highest Risk Category")
                        st.markdown(f"**{highest_risk.get('category', '')}: {highest_risk.get('name', '')}**")
                        st.markdown(f"Risk Level: **{highest_risk.get('risk_level', '').upper()}**")
                        st.markdown(f"Vulnerability Count: **{highest_risk.get('vulnerability_count', 0)}**")
                        
                        # Get detailed description
                        vuln_details = VulnerabilityDescriptions.get_description(highest_risk.get('category', ''))
                        st.markdown(f"Impact: **{vuln_details.get('impact', 'Unknown')}**")
                    
                    # Display risk distribution
                    if 'risk_counts' in metrics:
                        st.markdown("#### Risk Level Distribution")
                        risk_counts = metrics['risk_counts']
                        
                        # Create a horizontal bar chart
                        fig = go.Figure()
                        
                        risk_levels = ['critical', 'high', 'medium', 'low']
                        colors = ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
                        
                        fig.add_trace(go.Bar(
                            y=[level.upper() for level in risk_levels],
                            x=[risk_counts.get(level, 0) for level in risk_levels],
                            orientation='h',
                            marker_color=colors,
                            text=[risk_counts.get(level, 0) for level in risk_levels],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title='Risk Level Distribution',
                            xaxis_title='Count',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key='security_risk_chart_' + str(uuid.uuid4()))
        with tab3:
            # Display attack surface analysis
            st.markdown("### Attack Surface Analysis")
            
            # Create a heatmap of vulnerability categories
            risk_levels = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            risk_colors = {'critical': '#dc3545', 'high': '#fd7e14', 'medium': '#ffc107', 'low': '#28a745'}
            
            # Prepare data for heatmap
            category_data = []
            all_categories = ['LLM01', 'LLM02', 'LLM03', 'LLM04', 'LLM05', 'LLM06', 'LLM07', 'LLM08', 'LLM09', 'LLM10', 'LLM11', 'LLM12', 'LLM13']
            for cat_id in all_categories:
                category = categories.get(cat_id, {'vulnerability_count': 0})
                category_data.append({
                    'Category': f"{cat_id}: {category.get('name', 'Unknown')}",
                    'Risk Level': category.get('risk_level', 'low'),
                    'Vulnerability Count': category.get('vulnerability_count', 0),
                    'Risk Score': risk_levels.get(category.get('risk_level', 'low'), 0)
                })
            
            if category_data:
                # Sort by risk score (highest first)
                category_data.sort(key=lambda x: x['Risk Score'], reverse=True)
                
                # Create a dataframe for the heatmap
                # import pandas as pd
                df = pd.DataFrame(category_data)
                
                # Display attack surface table
                st.markdown("#### Vulnerability Exposure Map")
                st.markdown("This map shows the exposure of your system to different vulnerability categories, sorted by risk level.")
                
                # Create a styled dataframe
                def color_risk_level(val):
                    color = risk_colors.get(val.lower(), '#6c757d')
                    return f'background-color: {color}; color: white; font-weight: bold'
                
                # Apply styling and display
                styled_df = df.style.applymap(color_risk_level, subset=['Risk Level'])
                st.dataframe(styled_df)
                
                # Display attack vectors
                st.markdown("#### Primary Attack Vectors")
                st.markdown("These are the most likely attack vectors based on the identified vulnerabilities:")
                
                # Identify top attack vectors based on risk levels
                high_risk_categories = [cat for cat in category_data if cat['Risk Score'] >= 3]
                if high_risk_categories:
                    for i, category in enumerate(high_risk_categories[:3]):  # Show top 3
                        cat_id = category['Category'].split(':', 1)[0].strip()
                        vuln_details = VulnerabilityDescriptions.get_description(cat_id)
                        
                        st.markdown(f"**{i+1}. {category['Category']}**")
                        st.markdown(f"*Potential attack vector:* {vuln_details['description']}")
                        st.markdown(f"*Impact:* {vuln_details['impact']}")
                        
                        # Show examples as attack scenarios
                        if vuln_details['examples']:
                            st.markdown("*Attack scenarios:*")
                            for example in vuln_details['examples'][:2]:  # Limit to 2 examples
                                st.markdown(f"- {example}")
                else:
                    st.info("No high-risk attack vectors identified.")
                
                # Display vulnerability trend analysis
                st.markdown("#### Vulnerability Distribution Analysis")
                
                # Create a pie chart of risk levels
                risk_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                for cat in category_data:
                    risk_counts[cat['Risk Level'].lower()] += 1
                
                pie_fig = go.Figure(data=[go.Pie(
                    labels=list(risk_counts.keys()),
                    values=list(risk_counts.values()),
                    hole=.4,
                    marker_colors=['#dc3545', '#fd7e14', '#ffc107', '#28a745']
                )])
                
                pie_fig.update_layout(
                    title_text="Distribution of Risk Levels",
                    annotations=[dict(text='Risk<br>Profile', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                
                st.plotly_chart(pie_fig, use_container_width=True)
            else:
                st.info("No vulnerability data available for attack surface analysis.")
                
        with tab4:
            # Display mitigation recommendations
            st.markdown("### Security Mitigation Plan")
            
            if 'recommendations' in security_eval and security_eval['recommendations']:
                # Group recommendations by priority
                critical_recs = []
                high_recs = []
                other_recs = []
                
                # Process recommendations
                for rec in security_eval['recommendations']:
                    if 'priority' in rec and rec['priority'] == 'critical':
                        critical_recs.append(rec['text'])
                    elif 'priority' in rec and rec['priority'] == 'high':
                        high_recs.append(rec['text'])
                    else:
                        other_recs.append(rec.get('text', rec) if isinstance(rec, dict) else rec)
                
                # Display recommendations by priority
                if critical_recs:
                    st.markdown("#### Critical Priority Mitigations")
                    for i, rec in enumerate(critical_recs):
                        st.markdown(f"<div class='mitigation-item' style='border-left-color: #dc3545;'>{i+1}. {rec}</div>", unsafe_allow_html=True)
                
                if high_recs:
                    st.markdown("#### High Priority Mitigations")
                    for i, rec in enumerate(high_recs):
                        st.markdown(f"<div class='mitigation-item' style='border-left-color: #fd7e14;'>{i+1}. {rec}</div>", unsafe_allow_html=True)
                
                if other_recs:
                    st.markdown("#### Other Recommended Mitigations")
                    for i, rec in enumerate(other_recs):
                        st.markdown(f"<div class='mitigation-item'>{i+1}. {rec}</div>", unsafe_allow_html=True)
                
                # Create a timeline for implementation
                st.markdown("#### Implementation Timeline")
                timeline_data = {
                    'Phase': ['Immediate', 'Short-term', 'Medium-term', 'Long-term'],
                    'Timeframe': ['Within 24 hours', 'Within 1 week', 'Within 1 month', 'Within 3 months'],
                    'Focus': ['Critical vulnerabilities', 'High-risk issues', 'Medium-risk issues', 'Low-risk issues']
                }
                
                timeline_df = pd.DataFrame(timeline_data)
                st.dataframe(timeline_df)
            else:
                st.info("No specific recommendations available.")
                
                # Display general security best practices
                st.markdown("#### General Security Best Practices")
                best_practices = [
                    "Implement input validation for all user inputs",
                    "Use content security policies to prevent injection attacks",
                    "Regularly update and patch all dependencies",
                    "Implement rate limiting to prevent abuse",
                    "Use proper authentication and authorization mechanisms"
                ]
                
                for practice in best_practices:
                    st.markdown(f"- {practice}")
            
            # Add a security improvement roadmap visualization if there are categories
            if categories:
                st.markdown("#### Security Improvement Roadmap")
                
                # Create a simple Gantt chart for security improvements
                fig = go.Figure()
                
                # Add traces for each risk level
                risk_levels = ['critical', 'high', 'medium', 'low']
                colors = ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
                
                # Create a new Figure object for the roadmap
                roadmap_fig = go.Figure()
                
                for i, (level, color) in enumerate(zip(risk_levels, colors)):
                    # Count categories with this risk level
                    count = len([cat for cat in categories.values() if cat.get('risk_level', 'low') == level])
                    if count > 0:
                        roadmap_fig.add_trace(go.Bar(
                            y=[level.capitalize()],
                            x=[count * 7],  # Estimated days to fix
                            orientation='h',
                            marker=dict(color=color),
                            name=f"{level.capitalize()} Risk",
                            text=[f"{count} issues"],
                            textposition='auto'
                        ))
                
                roadmap_fig.update_layout(
                    title="Estimated Timeline for Security Improvements",
                    xaxis_title="Days to Implement",
                    yaxis_title="Risk Level",
                    template="plotly_white"
                )
                
                st.plotly_chart(roadmap_fig, use_container_width=True)