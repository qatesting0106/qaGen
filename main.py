import os
import json
import warnings
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import giskard
from src.core.qa_processor import QAProcessor
from src.core.metrics_evaluator import MetricsEvaluator
from src.utils.file_handler import FileHandler
from src.utils.data_processor import DataProcessor
from src.utils.report_generator import ReportGenerator
from src.ui.dashboard_components import DashboardComponents
# Load environment variables and configure settings
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Giskard settings
giskard.llm.set_llm_model("mistral/mistral-large-latest")
giskard.llm.set_embedding_model("mistral/mistral-embed")

# Constants
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "genai_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = "data"

# Initialize components
file_handler = FileHandler(OUTPUT_DIR)
qa_processor = QAProcessor(MISTRAL_API_KEY, GROQ_API_KEY)
qa_processor.initialize_embeddings()
metrics_evaluator = MetricsEvaluator(OUTPUT_DIR, qa_processor.embeddings)
report_generator = ReportGenerator(OUTPUT_DIR)
dashboard = DashboardComponents()


def main():
    # st.title("QA Dashboard")

    # Load CSS
    css_path = os.path.abspath("style.css")
    if os.path.exists(css_path):
        try:
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading CSS: {e}")

    # Add dropdown for data selection
    # Add radio button for data selection
    data_selection = st.radio('Select data source:', ('Use existing data', 'Upload new data'))

    if data_selection == 'Use existing data':
        data_dir = 'data'
        if os.path.exists(data_dir):
            existing_files = os.listdir(data_dir)
            if existing_files:
                selected_file = st.selectbox('Select a data file:', existing_files)
                uploaded_file = open(os.path.join(data_dir, selected_file), 'rb')
            else:
                st.warning('No data files found in the data directory.')
    else:
        # File upload section
        uploaded_file = st.file_uploader('Upload new test data file', type=['txt'])
        if uploaded_file is not None:
            try:
                os.makedirs('data', exist_ok=True)
                file_path = os.path.join('data', uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                selected_file = uploaded_file.name
                st.success(f'File {uploaded_file.name} uploaded successfully!')
            except Exception as e:
                st.error(f'Error saving file: {str(e)}')
                return

    # Display control buttons
    generate_test, generate_answers, calculate_metrics, evaluate_security, download_report, num_questions = dashboard.display_control_buttons()

    if uploaded_file:
        try:
            # Process document
            documents = qa_processor.load_documents(uploaded_file)
            qa_processor.initialize_embeddings()
            qa_processor.create_vector_store(documents)
            qa_processor.initialize_llm()
            qa_processor.create_qa_chain()

            # Create knowledge base and handle test generation
            if generate_test:
                knowledge_base = qa_processor.create_knowledge_base(documents)
                testset = DataProcessor.generate_test_questions(knowledge_base, num_questions=num_questions)
                test_df = testset.to_pandas()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                testset_file_path = os.path.join(OUTPUT_DIR, f"testset_{timestamp}.jsonl")
                
                # Save test set as JSONL with additional fields
                with open(testset_file_path, 'w') as f:
                    for _, row in test_df.iterrows():
                        f.write(json.dumps({
                            'question': str(row['question']),
                            'reference_context': str(row['reference_context']),
                            'reference_answer': str(row['reference_answer']),
                            'metadata': {
                                'timestamp': timestamp,
                                'question_type': row.get('question_type', 'general')
                            }
                        }) + '\n')
                st.markdown(f'<div class="alert-success">Test set generated and saved to {testset_file_path} successfully!</div>', unsafe_allow_html=True)
                st.subheader("Generated Test Set")
                # Get latest testset file
                output_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('testset_')]
                if output_files:
                    latest_file = sorted(output_files, reverse=True)[0]
                    testset_path = os.path.join(OUTPUT_DIR, latest_file)
                    
                    try:
                        with open(testset_path, 'r') as f:
                            testset_data = [json.loads(line) for line in f]
                        
                        for idx, item in enumerate(testset_data):
                            st.markdown(f"### Question {idx + 1}")
                            st.markdown(f"**Question:** {item['question']}")
                            st.markdown(f"**Reference Context:** {item['reference_context']}")
                            st.markdown(f"**Reference Answer:** {item['reference_answer']}")
                            st.markdown(f"**Metadata:**")
                            if 'metadata' in item and isinstance(item['metadata'], dict):
                                st.json(item['metadata'])
                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Error reading testset file: {str(e)}")
                else:
                    st.warning("No testset files found in the output directory.")

            # Process questions and generate answers
            if generate_answers:
                try:
                    # Get latest testset file
                    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('testset_')]
                    if not output_files:
                        dashboard.display_error_message("No testset files found. Please generate a testset first.")
                        return
                    
                    # Sort files by timestamp and get latest
                    latest_file = sorted(output_files, reverse=True)[0]
                    testset_path = os.path.join(OUTPUT_DIR, latest_file)
                    
                    testset_data = file_handler.load_jsonl(testset_path)
                    if not testset_data:
                        dashboard.display_error_message("Testset is empty. Please generate a new testset.")
                        return
                    
                    metrics_data = []
                    for idx, question in enumerate(testset_data):
                        answer = qa_processor.generate_answer(question['question'])
                        
                        st.markdown(f"### Question {idx + 1}")
                        st.markdown(f"**Question:**\n{question['question']}")
                        st.markdown(f"**Generated Answer:**\n{answer.message}")
                        st.markdown(f"**Reference Answer:**\n{question['reference_answer']}")
                        st.markdown("---")
                        
                        metrics = metrics_evaluator.calculate_metrics(
                            answer.message,
                            question['reference_answer'],
                            question['question']
                        )
                        metrics_data.append(metrics)
                    file_handler.save_json(metrics_data, "metrics_results")
                    dashboard.display_success_message("Answers generated successfully!")
                except Exception as e:
                    dashboard.display_error_message(f"Error generating answers: {str(e)}")

            # Calculate and display metrics
            if calculate_metrics:
                try:
                    # Find the latest metrics results file
                    metrics_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('metrics_results_')]
                    if not metrics_files:
                        dashboard.display_error_message("No metrics results found. Please generate answers first.")
                        return
                    
                    # Sort files by timestamp and get latest
                    latest_metrics_file = sorted(metrics_files, reverse=True)[0]
                    metrics_file_path = os.path.join(OUTPUT_DIR, latest_metrics_file)
                    metrics_data = file_handler.load_json(metrics_file_path)
                    if metrics_data:
                        metrics_df = DataProcessor.process_metrics_data(metrics_data)
                        dashboard.display_metrics_summary(metrics_df)
                        dashboard.plot_metrics_comparison(metrics_df)
                        dashboard.plot_metrics_heatmap(metrics_df)
                        dashboard.display_success_message("Metrics calculated and displayed successfully!")
                except Exception as e:
                    dashboard.display_error_message(f"Error calculating metrics: {str(e)}")
            
            # Evaluate security risks
            if evaluate_security:
                try:
                    # Get latest testset file
                    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('testset_')]
                    if not output_files:
                        dashboard.display_error_message("No testset files found. Please generate a testset first.")
                        return
                    
                    latest_file = sorted(output_files, reverse=True)[0]
                    testset_path = os.path.join(OUTPUT_DIR, latest_file)
                    testset_data = file_handler.load_jsonl(testset_path)
                    
                    if not testset_data:
                        dashboard.display_error_message("Testset is empty. Please generate a new testset.")
                        return
                    
                    st.subheader("Security Risk Assessment")
                    for idx, question_data in enumerate(testset_data):
                        security_assessment = qa_processor.security_evaluator.evaluate_security_risks(
                            question_data['question'],
                            question_data.get('reference_answer', '')
                        )
                        
                        st.markdown(f"### Question {idx + 1}")
                        st.markdown(f"**Question:** {question_data['question']}")
                        st.markdown(f"**Risk Score:** {security_assessment['risk_score']:.2f}/10.0")
                        st.markdown(f"**Severity Level:** {security_assessment['severity_level'].upper()}")
                        
                        if security_assessment['risks']:
                            st.markdown("**Detected Risks:**")
                            for risk in security_assessment['risks']:
                                st.markdown(f"- **{risk['category']}** ({risk['severity'].upper()})")
                                st.markdown(f"  - Location: {risk['location']}")
                                st.markdown(f"  - Description: {risk['description']}")
                        else:
                            st.markdown("**No security risks detected**")
                        st.markdown("---")
                        
                    dashboard.display_success_message("Security evaluation completed successfully!")
                except Exception as e:
                    dashboard.display_error_message(f"Error evaluating security: {str(e)}")
                    
                # Handle report generation after security evaluation
            if download_report:
                print("Download Report Button Clicked")
                try:
                        # Generate and display the report content
                        html_report = report_generator.generate_report()
                        print("HTML Report:", html_report)
                        
                        if html_report:
                            st.markdown("### Generated Report Preview")
                            st.components.v1.html(html_report, height=400, scrolling=True)
                            
                            # Display download button for the report
                            report_generator.display_download_button()
                            
                            # Add success message
                            st.success("Report generated successfully! Click the button above to download.")
                        else:
                            dashboard.display_error_message("Failed to generate report content")
                except Exception as e:
                    dashboard.display_error_message(f"Error generating report: {str(e)}")
        except Exception as e:
            dashboard.display_error_message(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()