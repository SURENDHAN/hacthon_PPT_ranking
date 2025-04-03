import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
st.set_page_config(page_title="Enhanced PDF Evaluator", layout="wide")
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
import matplotlib.pyplot as plt
import numpy as np
import re
import json
from io import BytesIO
import time
import pandas as pd
import concurrent.futures
from functools import partial
import PyPDF2

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

@st.cache_resource(ttl=3600) 
def load_llm_model():
    try:
        
        return ChatOllama(
            model="deepseek-r1:7b", 
            base_url="http://localhost:11434",  
            temperature=0.1,
            top_p=0.9,
            num_ctx=4096,  
            system="You are an expert evaluator of technical presentations...",  
            timeout=60  
        )
    except Exception as e:
        st.error(f"Failed to load Ollama model: {e}")
        st.info("Please ensure Ollama is running locally and the DeepSeek model is downloaded.")
        st.info("Run these commands in your terminal:")
        st.code("""
        ollama pull deepseek-llm:7b
        ollama serve
        """)
        return None

chat_model = load_llm_model()

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = []
        total_pages = len(pdf_reader.pages)
        
        progress_bar = st.progress(0)
        
        for i in range(total_pages):
            page = pdf_reader.pages[i]
            page_text = page.extract_text()
            if page_text:
                text.append(f"[PAGE {i+1}/{total_pages}]\n{page_text}")
            
            progress_bar.progress((i + 1) / total_pages)
        
        progress_bar.empty()
        
        return "\n\n".join(text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_scores(response_text):
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return result.get('scores', {}), result.get('justifications', {})
            except json.JSONDecodeError:
                pass
        
        categories = [
            "Innovation", "Technical Feasibility", "Impact", 
            "Technical Depth", "Scalability", "Market Potential"
        ]
        scores = {}
        justifications = {}
        
        score_patterns = [
            re.compile(rf"{category}.*?\(Score\s*(\d+(?:\.\d+)?)/10\)", re.IGNORECASE | re.DOTALL),
            re.compile(rf"{category}.*?:\s*(\d+(?:\.\d+)?)/10", re.IGNORECASE | re.DOTALL),
            re.compile(rf"{category}.*?(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE | re.DOTALL)
        ]
        
        justification_patterns = [
            re.compile(rf"{category}.*?\(Score\s*\d+(?:\.\d+)?/10\)\s*(.*?)(?=\n\n|\n\*|\Z)", re.IGNORECASE | re.DOTALL),
            re.compile(rf"{category}.*?:\s*(?:\d+(?:\.\d+)?/10)\s*(.*?)(?=\n\n|\n\*|\Z)", re.IGNORECASE | re.DOTALL),
            re.compile(rf"Justification for {category}:\s*(.*?)(?=\n\n|\n\*|\Z)", re.IGNORECASE | re.DOTALL)
        ]
        
        for category in categories:
            for i, pattern in enumerate(score_patterns):
                score_match = pattern.search(response_text)
                if score_match:
                    try:
                        scores[category] = float(score_match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
            
            for i, pattern in enumerate(justification_patterns):
                just_match = pattern.search(response_text)
                if just_match:
                    justifications[category] = just_match.group(1).strip()
                    break
            
            if category not in justifications:
                justifications[category] = "No justification found."
        
        return scores, justifications
    except Exception as e:
        st.error(f"⚠ Error in extracting evaluation results: {e}")
        return {}, {}

def calculate_weighted_average(scores, weights):
    valid_scores = {k: v for k, v in scores.items() if v is not None}
    if not valid_scores:
        return 0
    
    total_weighted_score = sum(valid_scores[category] * weights[category] for category in valid_scores)
    total_weight = sum(weights[category] for category in valid_scores)
    
    return total_weighted_score / total_weight if total_weight > 0 else 0

def calculate_average(scores):
    valid_scores = [v for v in scores.values() if v is not None]
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0

def plot_evaluation(scores, justifications, weights=None):
    valid_scores = {k: v for k, v in scores.items() if v is not None}
    
    if not valid_scores:
        st.error("❌ No valid scores available for visualization.")
        return
    
    if weights is None:
        weights = {category: 1.0 for category in valid_scores}
    
    weighted_avg = calculate_weighted_average(valid_scores, weights)
    regular_avg = calculate_average(valid_scores)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background-color:#f0f9ff; padding:15px; border-radius:10px; 
             text-align:center; margin:20px 0;'>
            <h3 style='margin:0; color:#0c4a6e;'>Weighted Score</h3>
            <div class='weighted-score'>
                {weighted_avg:.1f}<span style='font-size:28px;'>/10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color:#f0f2f5; padding:15px; border-radius:10px; 
             text-align:center; margin:20px 0;'>
            <h3 style='margin:0; color:#4b5563;'>Unweighted Score</h3>
            <div class='weighted-score' style='color:#4b5563;'>
                {regular_avg:.1f}<span style='font-size:28px;'>/10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(valid_scores.keys())
    values = list(valid_scores.values())
    
    colors = ['#ff6b6b' if v < 4 else '#ffd166' if v < 7 else '#06d6a0' for v in values]
    
    bars = ax.barh(categories, values, color=colors)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel("Scores (out of 10)", fontsize=12)
    ax.set_xlim(0, 10.5)
    ax.set_title("Evaluation Scores", fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.axvline(x=regular_avg, color='#4a4e69', linestyle='--', linewidth=2)
    ax.text(regular_avg, -0.5, f'Avg: {regular_avg:.1f}', ha='center', va='top', 
            fontweight='bold', color='#4a4e69')
            
    ax.axvline(x=weighted_avg, color='#0284c7', linestyle='-', linewidth=2)
    ax.text(weighted_avg, -0.8, f'Weighted: {weighted_avg:.1f}', ha='center', va='top', 
            fontweight='bold', color='#0284c7')
    
    st.pyplot(fig)
    
    st.markdown("### 📋 Detailed Evaluation")
    
    for category in scores:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            score_val = scores[category] if scores[category] is not None else 0
            score_color = '#ff6b6b' if score_val < 4 else '#ffd166' if score_val < 7 else '#06d6a0'
            st.markdown(f"""
            <div class="score-box">
                <div class="score-label">{category}</div>
                <div class="score-value" style="background-color: {score_color};">
                    {score_val}/10
                </div>
                <div style="font-size: 0.9em; margin-top: 5px;">
                    Weight: {weights[category]}x
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Justification:**")
            st.write(justifications[category])
        
        with col3:
            weighted_contribution = score_val * weights[category]
            st.markdown(f"""
            <div style="text-align: center; margin-top: 25px;">
                <div style="font-size: 0.9em;">Weighted Score:</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #0284c7;">
                    {score_val} × {weights[category]} = {weighted_contribution:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

def get_evaluation_prompt(ppt_text):
    return  f""" 
    ## Ask
    Evaluate a hackathon/ideathon presentation for college students, focusing exclusively on the core technical content and provide a detailed assessment in a specific JSON format.

    ## Content
    Extract and analyze only the technical and substantive content from the presentation based on six key criteria, each scored on a 1-10 scale:
    - Innovation: Originality, use of emerging technologies, fresh perspectives
    - Technical Feasibility: Implementation practicality, problem-solution alignment, architectural logic
    - Impact: Effectiveness in solving real-world problems, measurable benefits, deployment potential
    - Technical Depth: Sophistication of approach, justified technology choices, best practices
    - Scalability: Expansion potential, adaptability, performance considerations
    - Market Potential: Demand evidence, commercial viability, adoption prospects

    ## Context
    You are an expert judge evaluating presentations from college students. Ignore introductory slides, team details, slide titles, college/university names, and non-substantive content. Apply appropriate penalties for ideas that already exist without improvement, solutions lacking implementation details, minimal impact, insufficient technical depth, non-scalable concepts, or unclear market need. Handle blank PDFs, templates, and weak content appropriately.

    ## Presentation Content to Evaluate
    {ppt_text}

    ## Example Output Format
    ```json
    {{
        "scores": {{
            "Innovation": X.X,
            "Technical Feasibility": X.X,
            "Impact": X.X,
            "Technical Depth": X.X,
            "Scalability": X.X,
            "Market Potential": X.X
        }},
        "justifications": {{
            "Innovation": "Detailed explanation...",
            "Technical Feasibility": "Detailed explanation...",
            "Impact": "Detailed explanation...",
            "Technical Depth": "Detailed explanation...",
            "Scalability": "Detailed explanation...",
            "Market Potential": "Detailed explanation..."
        }},
        "overall_average": X.X,
        "summary": "Brief overall assessment..."
    }}
    ```
    """
def get_default_weights():
    return {
        "Innovation": 4.00,
        "Technical Feasibility": 2.00,
        "Impact": 3.00,
        "Technical Depth": 1.50,
        "Scalability": 2.00,
        "Market Potential": 2.00
    }

def process_single_presentation(uploaded_file):
    file_bytes = BytesIO(uploaded_file.getvalue())
    file_name = uploaded_file.name
    
    pdf_text = extract_text_from_pdf(file_bytes)
    if not pdf_text:
        return None
        
    result = {
        'file_name': file_name,
        'pdf_text': pdf_text
    }
    
    return result

def process_presentations_in_parallel(uploaded_files):
    extract_func = partial(process_single_presentation)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(extract_func, uploaded_file): uploaded_file 
                         for uploaded_file in uploaded_files}
        
        results = []
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    st.success(f"✅ Successfully processed {file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")
                
    return results

def batch_evaluate_presentations(presentation_data_list, chat_model, weights):
    results = []
    
    total = len(presentation_data_list)
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, data in enumerate(presentation_data_list):
        file_name = data['file_name']
        pdf_text = data['pdf_text']
        
        progress_text.text(f"Evaluating {i+1}/{total}: {file_name}")
        progress_bar.progress((i) / total)
        
        try:
            prompt = get_evaluation_prompt(pdf_text)
            
            response = chat_model.invoke(prompt)
            
            if not response or not hasattr(response, "content"):
                st.error(f"❌ Received empty response from AI model for {file_name}.")
                continue
            
            response_text = response.content
            scores, justifications = extract_scores(response_text)
            
            if not scores or all(v is None for v in scores.values()):
                st.warning(f"⚠️ Could not extract proper scores from AI response for {file_name}.")
                continue
            
            weighted_score = calculate_weighted_average(scores, weights)
            unweighted_score = calculate_average(scores)
            
            result = {
                'file_name': file_name,
                'scores': scores,
                'justifications': justifications,
                'weighted_score': weighted_score,
                'unweighted_score': unweighted_score,
                'response_text': response_text,
                'pdf_text': pdf_text
            }
            
            results.append(result)
            
            progress_bar.progress((i + 1) / total)
            
        except Exception as e:
            st.error(f"❌ Error during evaluation of {file_name}: {e}")
            continue
    
    progress_text.empty()
    progress_bar.empty()
    
    return results

def display_rankings(results_list):
    if not results_list:
        st.warning("No presentations have been evaluated yet.")
        return
    
   
    for result in results_list:
        
        weighted_score = result['weighted_score']
        unweighted_score = result['unweighted_score']
        
        
        innovation_score = result['scores'].get('Innovation', 0)
        adjusted_score = (weighted_score * 0.6) + (unweighted_score * 0.3) + (innovation_score * 0.1)
        result['adjusted_score'] = adjusted_score
    
    
    sorted_results = sorted(
        results_list,
        key=lambda x: (-x['adjusted_score'], -x['weighted_score'], -x['scores'].get('Innovation', 0))
    )
    
    
    for i, result in enumerate(sorted_results):
        if i > 0 and result['adjusted_score'] == sorted_results[i-1]['adjusted_score']:
            
            result['rank'] = sorted_results[i-1]['rank']
        else:
            result['rank'] = i + 1
    
    df = pd.DataFrame([
        {
            'Rank': result['rank'],
            'File Name': result['file_name'],
            'Adjusted Score': f"{result['adjusted_score']:.1f}",
            'Weighted Score': f"{result['weighted_score']:.1f}",
            'Unweighted Score': f"{result['unweighted_score']:.1f}",
            'Innovation': f"{result['scores'].get('Innovation', 0):.1f}",
            'Technical Feasibility': f"{result['scores'].get('Technical Feasibility', 0):.1f}",
            'Impact': f"{result['scores'].get('Impact', 0):.1f}",
            'Technical Depth': f"{result['scores'].get('Technical Depth', 0):.1f}",
            'Scalability': f"{result['scores'].get('Scalability', 0):.1f}",
            'Market Potential': f"{result['scores'].get('Market Potential', 0):.1f}"
        } for result in sorted_results
    ])
    
    st.markdown("### 🏆 Presentation Rankings")
    
    st.dataframe(
        df,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "🏆 Rank",
                help="Position in the ranking (ties share the same rank)",
                format="%d"
            ),
            "Adjusted Score": st.column_config.NumberColumn(
                "Combined Score",
                help="60% weighted + 30% unweighted + 10% innovation",
                format="%.1f"
            ),
            "Weighted Score": st.column_config.NumberColumn(
                "Weighted Score",
                help="Score with criteria weights applied",
                format="%.1f"
            ),
            "Unweighted Score": st.column_config.NumberColumn(
                "Raw Score",
                help="Average score without weights",
                format="%.1f"
            ),
        },
        hide_index=True,
        use_container_width=True
    )
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(sorted_results) * 0.4)))
    
    names = [f"{r['rank']}. {r['file_name']}" for r in sorted_results]
    scores = [r['adjusted_score'] for r in sorted_results]
    
   
    colors = []
    for score in scores:
        if score >= 8:
            colors.append('gold') 
        elif score >= 6:
            colors.append('silver')  
        elif score >= 4:
            colors.append('#cd7f32') 
        else:
            colors.append('#3498db')  
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Adjusted Score')
    ax.set_title('Presentation Rankings by Combined Score')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                ha='left', va='center', fontweight='bold')
    
    ax.set_xlim(0, 10.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
    
    return sorted_results

def generate_summary_report(results):
    report = "# Evaluation Summary Report\n\n"
    
    sorted_results = sorted(results, key=lambda x: x.get('rank', float('inf')))
    
    report += "## Overall Rankings\n\n"
    report += "| Rank | Presentation | Weighted Score | Unweighted Score | Combined Score |\n"
    report += "|------|-------------|----------------|------------------|---------------|\n"
    
    for result in sorted_results:
        rank = result.get('rank', 'N/A')
        file_name = result['file_name']
        weighted_score = result['weighted_score']
        unweighted_score = result['unweighted_score']
        adjusted_score = result.get('adjusted_score', weighted_score)
        
        report += f"| {rank} | {file_name} | {weighted_score:.1f}/10 | {unweighted_score:.1f}/10 | {adjusted_score:.1f}/10 |\n"
    
    report += "\n\n"
    
    for result in sorted_results:
        report += f"## {result['file_name']}\n"
        report += f"**Rank:** {result.get('rank', 'N/A')}\n"
        report += f"**Weighted Score:** {result['weighted_score']:.1f}/10\n"
        report += f"**Unweighted Score:** {result['unweighted_score']:.1f}/10\n"
        if 'adjusted_score' in result:
            report += f"**Combined Score:** {result['adjusted_score']:.1f}/10\n"
        report += "\n"
        
        report += "### Detailed Scores\n\n"
        report += "| Category | Score | Justification |\n"
        report += "|----------|-------|---------------|\n"
        
        categories = ["Innovation", "Technical Feasibility", "Impact", 
                      "Technical Depth", "Scalability", "Market Potential"]
        
        for category in categories:
            score = result['scores'].get(category, 'N/A')
            if score != 'N/A':
                score = f"{score:.1f}/10"
            
            justification = result['justifications'].get(category, 'N/A')
            justification = justification.replace('\n', ' ').strip()
            
            report += f"| {category} | {score} | {justification} |\n"
        
        report += "\n"
        
        report += "### Full AI Evaluation\n\n"
        report += "```\n"
        report += result['response_text']
        report += "\n```\n\n"
        
        pdf_excerpt = result.get('pdf_text', '')
        if pdf_excerpt:
            pdf_excerpt = pdf_excerpt[:500] + "..." if len(pdf_excerpt) > 500 else pdf_excerpt
            report += "### PDF Content Preview\n\n"
            report += "```\n"
            report += pdf_excerpt
            report += "\n```\n\n"
        
        report += "---\n\n"
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report += f"\n\nReport generated on: {timestamp}\n"
    
    return report

def main():
    st.title("📊 Multi-PDF Evaluator & Ranker")
    st.markdown("""
    Upload multiple PDF presentations to receive AI-powered evaluations 
    and see them ranked based on weighted scores.
    """)
    
    if 'weights' not in st.session_state:
        st.session_state.weights = get_default_weights()
    
    if 'evaluated_files' not in st.session_state:
        st.session_state.evaluated_files = []
    
    if st.session_state.evaluated_files:
        ranked_results = display_rankings(st.session_state.evaluated_files)
        
        if st.button("Clear All Evaluations", type="primary", help="Remove all current evaluations"):
            st.session_state.evaluated_files = []
            st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.download_button(
                label="📥 Download Full Report (Markdown)",
                data=generate_summary_report(st.session_state.evaluated_files),
                file_name="evaluation_summary_report.md",
                mime="text/markdown"
            ):
                st.success("Full report downloaded successfully!")
                
        with col2:
            if st.button("📊 Export Rankings as CSV"):
                csv_data = generate_csv_rankings(st.session_state.evaluated_files)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="presentation_rankings.csv",
                    mime="text/csv"
                )

    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple files to evaluate and rank them."
    )

    if uploaded_files:
        existing_filenames = [result['file_name'] for result in st.session_state.evaluated_files]
        new_files = [f for f in uploaded_files if f.name not in existing_filenames]
        
        if new_files:
            st.markdown(f"### 🔍 Evaluating {len(new_files)} new presentations")
            
            if not chat_model:
                st.error("❌ LLM model is not available. Please check your configuration.")
            else:
                with st.status("Extracting text from PDFs...") as status:
                    presentation_data = process_presentations_in_parallel(new_files)
                    status.update(label=f"Text extraction complete for {len(presentation_data)} files", state="complete")
                
                if presentation_data:
                    with st.status("Running AI evaluations...") as status:
                        results = batch_evaluate_presentations(presentation_data, chat_model, st.session_state.weights)
                        status.update(label=f"Evaluation complete for {len(results)} files", state="complete")
                    
                    st.session_state.evaluated_files.extend(results)
                    
                    st.success(f"✅ Successfully evaluated {len(results)} presentations")
                    
                    if st.session_state.evaluated_files:
                        ranked_results = display_rankings(st.session_state.evaluated_files)
                        st.markdown("### 🎯 All Evaluations Complete")
                else:
                    st.warning("No valid presentation data was extracted. Please check your files.")
        else:
            st.info("All uploaded files have already been evaluated. View the rankings above.")
    else:
        if not st.session_state.evaluated_files:
            st.info("👆 Upload PDF files to get started")
            
    with st.expander("🔍 View Individual Presentation Details"):
        if st.session_state.evaluated_files:
            file_options = [result['file_name'] for result in st.session_state.evaluated_files]
            selected_file = st.selectbox("Select a presentation to view detailed evaluation:", file_options)
            
            selected_data = next((result for result in st.session_state.evaluated_files 
                                if result['file_name'] == selected_file), None)
            
            if selected_data:
                st.markdown(f"""
                <div style='background-color:#f8fafc; padding:15px; border-radius:10px; margin-bottom:20px;'>
                    <h3 style='margin:0;'>🗂️ {selected_data['file_name']}</h3>
                    <p class='file-info'>Rank: {selected_data.get('rank', 'Unranked')} | 
                    Weighted Score: {selected_data['weighted_score']:.1f}/10</p>
                </div>
                """, unsafe_allow_html=True)
                
                plot_evaluation(selected_data['scores'], selected_data['justifications'], 
                               st.session_state.weights)
                
                if st.checkbox("Show Raw AI Evaluation"):
                    st.markdown("### 🤖 Raw AI Response")
                    st.text_area("Full evaluation text:", selected_data['response_text'], 
                                height=300, key="raw_response")
                
                if st.download_button("📥 Download This Evaluation", 
                                    data=generate_summary_report([selected_data]),
                                    file_name=f"evaluation_{selected_data['file_name']}.md",
                                    mime="text/markdown"):
                    st.success("Download started!")
        else:
            st.info("No presentations have been evaluated yet.")

def generate_csv_rankings(results):
    import io
    import csv
    
    sorted_results = sorted(results, key=lambda x: x.get('rank', float('inf')))
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    header = [
        "Rank", "File Name", "Combined Score", "Weighted Score", "Unweighted Score",
        "Innovation", "Technical Feasibility", "Impact", "Technical Depth", 
        "Scalability", "Market Potential", "Total Users"
    ]
    writer.writerow(header)
    
    for result in sorted_results:
        rank = result.get('rank', 'N/A')
        file_name = result['file_name']
        weighted_score = result['weighted_score']
        unweighted_score = result['unweighted_score']
        adjusted_score = result.get('adjusted_score', weighted_score)
        
        scores = result['scores']
        row = [
            rank,
            file_name,
            f"{adjusted_score:.1f}",
            f"{weighted_score:.1f}",
            f"{unweighted_score:.1f}",
            f"{scores.get('Innovation', 'N/A')}" if isinstance(scores.get('Innovation'), (int, float)) else 'N/A',
            f"{scores.get('Technical Feasibility', 'N/A')}" if isinstance(scores.get('Technical Feasibility'), (int, float)) else 'N/A',
            f"{scores.get('Impact', 'N/A')}" if isinstance(scores.get('Impact'), (int, float)) else 'N/A',
            f"{scores.get('Technical Depth', 'N/A')}" if isinstance(scores.get('Technical Depth'), (int, float)) else 'N/A',
            f"{scores.get('Scalability', 'N/A')}" if isinstance(scores.get('Scalability'), (int, float)) else 'N/A',
            f"{scores.get('Market Potential', 'N/A')}" if isinstance(scores.get('Market Potential'), (int, float)) else 'N/A',
            "N/A"
        ]
        writer.writerow(row)
    
    return output.getvalue()

if __name__ == "__main__":
    main()