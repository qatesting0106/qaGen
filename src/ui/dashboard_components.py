import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
import json
import os

class DashboardComponents:
    @staticmethod
    def display_control_buttons():
        """Display control buttons for test set generation, answers, and metrics."""
        st.sidebar.header("Control Panel")
        with st.sidebar.container():
            st.markdown("<style>.stButton > button {margin: 0.5rem 0;}</style>", unsafe_allow_html=True)
            generate_test = st.button("Generate Test Set", key="generate_test", help="Create test questions from the document")
            generate_answers = st.button("Generate Answers", key="generate_answers", help="Generate answers for the test questions")
            calculate_metrics = st.button("Calculate Metrics", key="calculate_metrics", help="Evaluate the performance metrics")
            evaluate_security = st.button("Evaluate Security", key="evaluate_security", help="Evaluate security aspects of the system")
            
        # Add number input for test questions in main content area
        num_questions = st.number_input("Number of test questions", min_value=1, max_value=10, value=1, step=1, help="Specify how many test questions to generate")
        return generate_test, generate_answers, calculate_metrics, evaluate_security, num_questions

    @staticmethod
    def display_config_controller():
        """Display configuration controller button and fetch configurations."""
        st.sidebar.header("Configuration")
        with st.sidebar.container():
            if st.button("Get Configurations", key="get_configs", help="Fetch and display system configurations"):
                # Placeholder for configuration fetching logic
                st.success("Configurations retrieved successfully!", icon="✅")

    @staticmethod
    def display_metrics_summary(metrics_df: pd.DataFrame) -> None:
        """Display individual metrics for each response."""
        st.markdown("""
        <style>
        .main .block-container { padding-top: 2rem; max-width: 95%; }
        .metric-container { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
        .metric-group { margin: 1rem 0; padding: 1rem; border: 1px solid #e9ecef; border-radius: 0.5rem; }
        </style>
        """, unsafe_allow_html=True)
        
        st.header("Evaluation Metrics")
        st.markdown("### Individual Response Analysis")
        
        # Define metric groups
        metric_groups = {
            'Context Metrics': ['context_precision', 'context_recall', 'context_f1'],
            'Answer Quality': ['answer_relevance', 'answer_completeness', 'answer_consistency'],
            'Similarity Metrics': ['cosine_similarity', 'faithfulness'],
            
            'OWASP Top 10': ['prompt_injection_risk', 'info_disclosure_risk', 'output_handling_risk', 'agency_risk', 'prompt_leakage_risk']
        }
        
        # Display metrics for each response
        for idx, row in metrics_df.iterrows():
            with st.container():
                st.markdown(f"## Response {idx + 1}")
                
                # Question and Answer section
                col1, col2 = st.columns([1, 1])
                with col1:
                    if 'question' in metrics_df.columns:
                        st.markdown("### Question")
                        st.markdown(f"{row['question']}")
                with col2:
                    if 'generated_answer' in metrics_df.columns:
                        st.markdown("### Answer")
                        st.markdown(f"{row['generated_answer']}")
                
                # Metrics by group
                for group_name, metrics in metric_groups.items():
                    available_metrics = [m for m in metrics if m in metrics_df.columns]
                    if not available_metrics:
                        continue
                        
                    st.markdown(f"### {group_name}")
                    cols = st.columns(len(available_metrics))
                    for col, metric in zip(cols, available_metrics):
                        try:
                            value = pd.to_numeric(row[metric], errors='coerce')
                            if not pd.isna(value):
                                display_name = ' '.join(word.capitalize() for word in metric.split('_'))
                                with col:
                                    st.metric(
                                        label=display_name,
                                        value=f"{value:.3f}",
                                        delta=None,
                                        help=f"Score for {display_name}",
                                    label_visibility="collapsed" if 'risk' in metric else "visible"
                                    )
                        except Exception:
                            continue
                
                st.markdown("---")

    @staticmethod
    def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
        """Plot comparison of different metrics."""
        st.subheader("Metrics Comparison")
        st.markdown("Visual comparison of different evaluation metrics")
        
        # Filter out non-numeric columns and ensure only numeric data is used
        non_metric_columns = ['question', 'generated_answer', 'reference_context']
        
        # First filter columns that are not in the exclusion list
        potential_numeric_columns = [col for col in metrics_df.columns if col not in non_metric_columns]
        
        # Then verify each column contains numeric data that can be averaged
        numeric_columns = []
        numeric_values = {}
        
        for col in potential_numeric_columns:
            try:
                # Check if column contains dictionary or complex objects
                if metrics_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    continue
                    
                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(metrics_df[col], errors='coerce')
                # Only include if we have valid numeric values
                if not numeric_series.isna().all():
                    numeric_columns.append(col)
                    numeric_values[col] = numeric_series.mean()
            except Exception as e:
                # Skip columns that can't be converted to numeric
                print(f"Error processing column {col}: {str(e)}")
                continue
        
        if not numeric_columns:
            st.warning("No numeric metrics columns found for visualization.")
            return
        
        # Bar chart for average metrics
        with st.container():
            bar_fig = go.Figure(data=[go.Bar(
                x=numeric_columns,
                y=[numeric_values[col] for col in numeric_columns],
                text=[f"{numeric_values[col]:.3f}" for col in numeric_columns],
                textposition='auto',
                marker_color='#2563eb',
                hovertemplate='%{x}<br>Average: %{y:.3f}<extra></extra>'
            )])
            bar_fig.update_layout(
                title={
                    'text': 'Average Metrics Performance',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                yaxis_title='Score',
                yaxis={
                    'range': [0, 1],
                    'gridcolor': '#f0f0f0'
                },
                xaxis={
                    'tickangle': -45
                },
                showlegend=False,
                template='plotly_white',
                height=500,
                margin=dict(t=100, b=100, l=50, r=50),
                plot_bgcolor='white'
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="metrics_comparison")

    @staticmethod
    def display_qa_results(question: str, answer: str, context: str) -> None:
        """Display Q&A results with context."""
        st.header("Question & Answer Analysis")
        st.markdown("Detailed breakdown of individual Q&A results")
        
        with st.expander("Question", expanded=True):
            st.markdown(f"**Question:** {question}")
        
        with st.expander("Answer", expanded=True):
            st.markdown(f"**Answer:** {answer}")
        
        with st.expander("Context", expanded=False):
            st.markdown(f"**Context:** {context}")

    @staticmethod
    def plot_metrics_heatmap(metrics_df: pd.DataFrame) -> None:
        """Plot correlation heatmap of metrics."""
        st.subheader("Metrics Correlation")
        st.markdown("Inter-metric relationships and dependencies")
        
        # Filter out non-numeric columns and complex data types
        non_metric_columns = ['question', 'generated_answer', 'reference_context']
        
        # Create a clean dataframe with only numeric values
        numeric_data = {}
        
        for col in metrics_df.columns:
            # Skip known non-numeric columns
            if col in non_metric_columns:
                continue
                
            try:
                # Check if column contains dictionary or complex objects
                if metrics_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    continue
                    
                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(metrics_df[col], errors='coerce')
                
                # Only include if we have valid numeric values
                if not numeric_series.isna().all():
                    numeric_data[col] = numeric_series
            except Exception as e:
                # Skip columns that can't be converted to numeric
                print(f"Error processing column {col}: {str(e)}")
                continue
        
        # Create a new dataframe with only numeric columns
        numeric_df = pd.DataFrame(numeric_data)
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            st.warning("Insufficient numeric data for correlation heatmap.")
            return
        
        with st.container():
            try:
                corr_matrix = numeric_df.corr()
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorscale='RdBu'
                ))
                heatmap_fig.update_layout(
                    title='Metrics Correlation Matrix',
                    template='plotly_white'
                )
                st.plotly_chart(heatmap_fig, use_container_width=True, key="metrics_heatmap")
            except Exception as e:
                st.error(f"Error generating correlation heatmap: {str(e)}")
                return

    @staticmethod
    def display_error_message(error: str) -> None:
        """Display error message in the dashboard."""
        st.error(f"Error: {error}", icon="🚨")

    @staticmethod
    def display_success_message(message: str) -> None:
        """Display success message in the dashboard."""
        st.success(message, icon="✅")

    @staticmethod
    def load_metrics_data() -> pd.DataFrame:
        """Load and format metrics from generated JSON file."""
        try:
            metrics_dir = os.path.join(os.path.dirname(__file__), '../output')
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Validate directory exists after creation attempt
            if not os.path.exists(metrics_dir):
                raise FileNotFoundError(f"Metrics directory creation failed: {metrics_dir}")

            # Get all JSON files sorted by modification time
            json_files = sorted(
                [os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir) if f.endswith('.json')],
                key=os.path.getmtime,
                reverse=True
            )

            if not json_files:
                raise ValueError("No metrics files found in directory")

            # Load most recent file
            latest_file = json_files[0]
            print("latest_file :",latest_file)
            with open(latest_file, 'r') as f:
                metrics_data = json.load(f)

            if not metrics_data:
                raise ValueError("Empty metrics data in file")

            return pd.DataFrame(metrics_data)
        except Exception as e:
            st.error(f"Error loading metrics: {str(e)}")
            return pd.DataFrame()