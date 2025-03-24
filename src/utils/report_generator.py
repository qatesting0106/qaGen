import os
import json
import pandas as pd
import numpy as np
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
import streamlit as st

class ReportGenerator:
    def __init__(self, output_dir: str):
        """Initialize ReportGenerator with output directory."""
        self.output_dir = output_dir
        self.template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
        os.makedirs(self.template_dir, exist_ok=True)
        
    def _load_template(self) -> str:
        """Load the HTML template for the report."""
        template_path = os.path.join(self.template_dir, "report_template.html")
        try:
            with open(template_path, 'r') as f:
                return f.read()
        except Exception as e:
            st.error(f"Error loading template: {str(e)}")
            return ""
    
    def _get_metrics_data(self) -> List[Dict[str, Any]]:
        """Get the latest metrics data from the output directory."""
        metrics_files = [f for f in os.listdir(self.output_dir) if f.startswith('metrics_results_')]
        if not metrics_files:
            return []
        
        latest_file = sorted(metrics_files, reverse=True)[0]
        metrics_path = os.path.join(self.output_dir, latest_file)
        
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading metrics data: {str(e)}")
            return []
    
    def _get_testset_data(self) -> List[Dict[str, Any]]:
        """Get the latest testset data from the output directory."""
        testset_files = [f for f in os.listdir(self.output_dir) if f.startswith('testset_')]
        if not testset_files:
            return []
        
        latest_file = sorted(testset_files, reverse=True)[0]
        testset_path = os.path.join(self.output_dir, latest_file)
        
        try:
            with open(testset_path, 'r') as f:
                return [json.loads(line) for line in f]
        except Exception as e:
            st.error(f"Error loading testset data: {str(e)}")
            return []
    
    def _get_security_data(self) -> List[Dict[str, Any]]:
        """Get the latest security assessment data."""
        # This is a placeholder - in a real implementation, you would load actual security data
        # For now, we'll use mock data based on testset questions
        testset_data = self._get_testset_data()
        security_data = []
        
        for idx, question in enumerate(testset_data):
            # Mock security assessment data
            severity_level = np.random.choice(['low', 'medium', 'high', 'critical'], p=[0.6, 0.25, 0.1, 0.05])
            risks = []
            
            if severity_level != 'low':
                num_risks = np.random.randint(1, 3)
                risk_categories = ['Prompt Injection', 'Information Disclosure', 'Output Handling', 'Agency Risk']
                risk_severities = ['low', 'medium', 'high']
                
                for _ in range(num_risks):
                    risks.append({
                        'category': np.random.choice(risk_categories),
                        'severity': np.random.choice(risk_severities),
                        'location': f"Question {idx + 1}",
                        'description': f"Potential risk detected in the question or answer content."
                    })
            
            security_data.append({
                'question': question['question'],
                'severity_level': severity_level,
                'risks': risks
            })
        
        return security_data
    
    def _calculate_average_metrics(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average values for key metrics."""
        if not metrics_data:
            return {
                'avg_cosine_similarity': 0.0,
                'avg_faithfulness': 0.0,
                'avg_context_f1': 0.0,
                'avg_answer_relevance': 0.0
            }
        
        avg_metrics = {
            'avg_cosine_similarity': 0.0,
            'avg_faithfulness': 0.0,
            'avg_context_f1': 0.0,
            'avg_answer_relevance': 0.0
        }
        
        for metric in metrics_data:
            avg_metrics['avg_cosine_similarity'] += float(metric.get('cosine_similarity', 0))
            avg_metrics['avg_faithfulness'] += float(metric.get('faithfulness', 0))
            avg_metrics['avg_context_f1'] += float(metric.get('context_f1', 0))
            avg_metrics['avg_answer_relevance'] += float(metric.get('answer_relevance', 0))
        
        num_items = len(metrics_data)
        for key in avg_metrics:
            avg_metrics[key] = round(avg_metrics[key] / num_items, 3) if num_items > 0 else 0.0
        
        return avg_metrics
    
    def _get_metric_class(self, value: float) -> str:
        """Get CSS class based on metric value."""
        if value >= 0.8:
            return "high"
        elif value >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _prepare_qa_items(self, metrics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare QA items for the report template."""
        qa_items = []
        
        for idx, metric in enumerate(metrics_data):
            item = {
                'question': metric.get('question', f"Question {idx + 1}"),
                'generated_answer': metric.get('generated_answer', 'No answer generated'),
                'reference_answer': metric.get('reference_context', 'No reference available'),
                'cosine_similarity': round(float(metric.get('cosine_similarity', 0)), 3),
                'faithfulness': round(float(metric.get('faithfulness', 0)), 3),
                'context_f1': round(float(metric.get('context_f1', 0)), 3),
                'answer_relevance': round(float(metric.get('answer_relevance', 0)), 3),
            }
            
            # Add CSS classes for styling
            item['cosine_similarity_class'] = self._get_metric_class(item['cosine_similarity'])
            item['faithfulness_class'] = self._get_metric_class(item['faithfulness'])
            item['context_f1_class'] = self._get_metric_class(item['context_f1'])
            item['answer_relevance_class'] = self._get_metric_class(item['answer_relevance'])
            
            qa_items.append(item)
        
        return qa_items
    
    def _prepare_security_items(self, security_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare security items for the report template."""
        security_items = []
        
        for item in security_data:
            security_item = {
                'question': item.get('question', ''),
                'severity_level': item.get('severity_level', 'low'),
                'risks': item.get('risks', []),
                'severity_class': item.get('severity_level', 'low')
            }
            security_items.append(security_item)
        
        return security_items
    
    def _render_template(self, template: str, context: Dict[str, Any]) -> str:
        """Render the HTML template with the provided context."""
        try:
            rendered = template
            
            # Replace simple variables
            for key, value in context.items():
                if isinstance(value, (str, int, float)):
                    rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
            
            # Handle qa_items loop
            if 'qa_items' in context and context['qa_items']:
                qa_items_html = ""
                for item in context['qa_items']:
                    item_html = """
                    <div class="qa-item">
                        <div class="question">Q: {question}</div>
                        <div class="answer">A: {generated_answer}</div>
                        <div class="reference">Reference: {reference_answer}</div>
                        
                        <div class="metrics-detail">
                            <span class="metric-badge {cosine_similarity_class}">Similarity: {cosine_similarity}</span>
                            <span class="metric-badge {faithfulness_class}">Faithfulness: {faithfulness}</span>
                            <span class="metric-badge {context_f1_class}">Context F1: {context_f1}</span>
                            <span class="metric-badge {answer_relevance_class}">Relevance: {answer_relevance}</span>
                        </div>
                    </div>
                    """
                    for k, v in item.items():
                        item_html = item_html.replace(f"{{{k}}}", str(v))
                    qa_items_html += item_html
                rendered = rendered.replace("{{#each qa_items}}\n            <div class=\"qa-item\">\n                <div class=\"question\">Q: {{question}}</div>\n                <div class=\"answer\">A: {{generated_answer}}</div>\n                <div class=\"reference\">Reference: {{reference_answer}}</div>\n                \n                <div class=\"metrics-detail\">\n                    <span class=\"metric-badge {{cosine_similarity_class}}\">Similarity: {{cosine_similarity}}</span>\n                    <span class=\"metric-badge {{faithfulness_class}}\">Faithfulness: {{faithfulness}}</span>\n                    <span class=\"metric-badge {{context_f1_class}}\">Context F1: {{context_f1}}</span>\n                    <span class=\"metric-badge {{answer_relevance_class}}\">Relevance: {{answer_relevance}}</span>\n                </div>\n            </div>\n            {{/each}}", qa_items_html)
            
            # Handle security_items loop
            if 'security_items' in context and context['security_items']:
                security_items_html = ""
                for item in context['security_items']:
                    risks_html = ""
                    if item.get('risks'):
                        risks_html = "<h4>Detected Risks:</h4>\n"
                        for risk in item['risks']:
                            risks_html += f"""
                            <div class="risk-item">
                                <strong>{risk['category']}</strong> ({risk['severity']})<br>
                                <small>Location: {risk['location']}</small><br>
                                {risk['description']}
                            </div>
                            """
                    else:
                        risks_html = "<p>No security risks detected.</p>"
                    
                    item_html = """
                    <div class="security-item">
                        <div class="question">Q: {question}</div>
                        <div class="risk-level {severity_class}">Risk Level: {severity_level}</div>
                        {risks}
                    </div>
                    """
                    item_html = item_html.replace("{question}", item.get('question', ''))
                    item_html = item_html.replace("{severity_level}", item.get('severity_level', 'low'))
                    item_html = item_html.replace("{severity_class}", item.get('severity_class', 'low'))
                    item_html = item_html.replace("{risks}", risks_html)
                    
                    security_items_html += item_html
                
                rendered = rendered.replace("{{#each security_items}}\n            <div class=\"security-item\">\n                <div class=\"question\">Q: {{question}}</div>\n                <div class=\"risk-level {{severity_class}}\">Risk Level: {{severity_level}}</div>\n                \n                {{#if risks.length}}\n                <h4>Detected Risks:</h4>\n                {{#each risks}}\n                <div class=\"risk-item\">\n                    <strong>{{category}}</strong> ({{severity}})<br>\n                    <small>Location: {{location}}</small><br>\n                    {{description}}\n                </div>\n                {{/each}}\n                {{else}}\n                <p>No security risks detected.</p>\n                {{/if}}\n            </div>\n            {{/each}}", security_items_html)
            
            return rendered
        except Exception as e:
            st.error(f"Error rendering template: {str(e)}")
            return ""
            
            rendered = rendered.replace("{{#each qa_items}}\n            <div class=\"qa-item\">\n                <div class=\"question\">Q: {{question}}</div>\n                <div class=\"answer\">A: {{generated_answer}}</div>\n                <div class=\"reference\">Reference: {{reference_answer}}</div>\n                \n                <div class=\"metrics-detail\">\n                    <span class=\"metric-badge {{cosine_similarity_class}}\">Similarity: {{cosine_similarity}}</span>\n                    <span class=\"metric-badge {{faithfulness_class}}\">Faithfulness: {{faithfulness}}</span>\n                    <span class=\"metric-badge {{context_f1_class}}\">Context F1: {{context_f1}}</span>\n                    <span class=\"metric-badge {{answer_relevance_class}}\">Relevance: {{answer_relevance}}</span>\n                </div>\n            </div>\n            {{/each}}", qa_items_html)
        
        # Handle security_items loop
        if 'security_items' in context and context['security_items']:
            security_items_html = ""
            for item in context['security_items']:
                risks_html = ""
                if item.get('risks'):
                    risks_html += "<h4>Detected Risks:</h4>\n"
                    for risk in item['risks']:
                        risk_html = """
                        <div class="risk-item">
                            <strong>{category}</strong> ({severity})<br>
                            <small>Location: {location}</small><br>
                            {description}
                        </div>
                        """
                        for k, v in risk.items():
                            risk_html = risk_html.replace(f"{{{k}}}", str(v))
                        risks_html += risk_html
                else:
                    risks_html = "<p>No security risks detected.</p>"
                
                item_html = """
                <div class="security-item">
                    <div class="question">Q: {question}</div>
                    <div class="risk-level {severity_class}">Risk Level: {severity_level}</div>
                    
                    {risks}
                </div>
                """
                item_html = item_html.replace("{question}", item.get('question', ''))
                item_html = item_html.replace("{severity_level}", item.get('severity_level', 'low'))
                item_html = item_html.replace("{severity_class}", item.get('severity_class', 'low'))
                item_html = item_html.replace("{risks}", risks_html)
                
                security_items_html += item_html
            
            rendered = rendered.replace("{{#each security_items}}\n            <div class=\"security-item\">\n                <div class=\"question\">Q: {{question}}</div>\n                <div class=\"risk-level {{severity_class}}\">Risk Level: {{severity_level}}</div>\n                \n                {{#if risks.length}}\n                <h4>Detected Risks:</h4>\n                {{#each risks}}\n                <div class=\"risk-item\">\n                    <strong>{{category}}</strong> ({{severity}})<br>\n                    <small>Location: {{location}}</small><br>\n                    {{description}}\n                </div>\n                {{/each}}\n                {{else}}\n                <p>No security risks detected.</p>\n                {{/if}}\n            </div>\n            {{/each}}", security_items_html)
        
        return rendered
    
    def generate_report(self, document_name: Optional[str] = None) -> str:
        """Generate the HTML report."""
        try:
            # Load template
            template = self._load_template()
            if not template:
                st.error("Failed to load report template")
                return ""
            
            # Get data
            metrics_data = self._get_metrics_data()
            testset_data = self._get_testset_data()
            security_data = self._get_security_data()
            
            if not metrics_data:
                st.warning("No metrics data available for the report")
                return ""
            
            # Calculate average metrics
            avg_metrics = self._calculate_average_metrics(metrics_data)
            
            # Format document name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_doc_name = f"{document_name}_{timestamp}" if document_name else f"Uploaded_Document_{timestamp}"
            
            # Prepare context for template
            context = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'document_name': formatted_doc_name,
                'num_questions': len(testset_data),
                'current_year': datetime.now().year,
                'qa_items': self._prepare_qa_items(metrics_data),
                'security_items': self._prepare_security_items(security_data),
                **avg_metrics
            }
            
            # Render template
            rendered_report = self._render_template(template, context)
            if not rendered_report:
                st.error("Failed to render report template")
                return ""
                
            return rendered_report
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            return ""
    
    def save_report_to_file(self, html_content: str, filename: str = "qa_report.html") -> str:
        """Save the HTML report to a temporary file and return the path."""
        import tempfile
        import os
        
        try:
            # Create a temporary directory specifically for QA reports
            temp_base_dir = tempfile.gettempdir()
            qa_report_dir = os.path.join(temp_base_dir, "qa_reports")
            os.makedirs(qa_report_dir, exist_ok=True)
            
            # Create the file path
            file_path = os.path.join(qa_report_dir, filename)
            
            # Save the report
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return file_path
        except Exception as e:
            st.error(f"Error saving report to file: {str(e)}")
            return ""
    
    def display_download_button(self, document_name: Optional[str] = None) -> None:
        """Display a download button for the report."""
        try:
            # Generate report content
            report_content = self.generate_report(document_name)
            if not report_content:
                st.error("Failed to generate report content")
                return
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qa_report_{timestamp}.html"
            if document_name:
                filename = f"qa_report_{document_name}_{timestamp}.html"
            
            # Create download button
            st.markdown("### Download Report")
            st.markdown("Click the button below to download a comprehensive HTML report of the evaluation results.")
            
            # Save report to temporary file
            temp_file_path = self.save_report_to_file(report_content, filename)
            if not temp_file_path:
                st.error("Failed to save report to temporary file")
                return
            
            # Read the file content for download
            try:
                with open(temp_file_path, 'rb') as f:
                    file_content = f.read()
                
                st.download_button(
                    label="Download HTML Report",
                    data=file_content,
                    file_name=filename,
                    mime="text/html",
                    key=f"download_report_{timestamp}",
                    help="Download a comprehensive HTML report of the evaluation results"
                )
                
                # Add a direct download link as backup
                encoded_content = base64.b64encode(file_content).decode()
                st.markdown(
                    f"""<div style='margin-top: 10px; font-size: 0.8em;'>
                        Alternative download method: 
                        <a href='data:text/html;base64,{encoded_content}' 
                           download='{filename}' 
                           style='text-decoration: underline; color: #4CAF50;'>
                            Click here
                        </a>
                    </div>""",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error reading temporary file: {str(e)}")
            finally:
                # Clean up temporary file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
        except Exception as e:
            st.error(f"Error displaying download button: {str(e)}")