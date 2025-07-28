# -*- coding: utf-8 -*-
# app.py

import streamlit as st
import pandas as pd
import re
from dotenv import load_dotenv
import os
import json
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="NeuroAI Hub",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    /* Custom purple spinner color */
    .stSpinner > div > div {
        border-top-color: #9c27b0;
    }
</style>""", unsafe_allow_html=True)

st.title("🧠 NeuroAI Hub")
st.caption("Your conversational assistant for neuroradiology datasets. I can find datasets, create summaries, and generate plots.")


# --- API KEY & LLM SETUP ---
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.warning("⚠️ API key not found. Please set OPENAI_API_KEY in Streamlit secrets or environment.")
    st.stop()

# Set up the LLM using the provided API key from secrets
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4.1-mini',
    base_url="https://api.avalai.ir/v1",
    temperature=0
)


# --- DATA LOADING (Cached to run only once) ---
@st.cache_resource
def load_data():
    """Loads and prepares data from the Excel file."""
    file_path = 'neuroradiology_datasets_S_L.xlsx'
    if not os.path.exists(file_path):
        st.error(f"FATAL: The data file '{file_path}' was not found. Please make sure it's in your repository.", icon="️⚠️")
        st.stop()

    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    dataframes = {sheet: pd.read_excel(xls, sheet_name=sheet).assign(category=sheet) for sheet in sheet_names}
    combined_df = pd.concat(dataframes.values(), ignore_index=True)
    
    return dataframes, combined_df, sheet_names

dataframes, combined_df, sheet_names = load_data()


# --- AGENT AND TOOLS SETUP (Cached to run only once) ---
@st.cache_resource
def setup_agent(_llm, _combined_df, _dataframes, _sheet_names):
    """Defines all tools and initializes the LangChain agent."""

    ALL_DISPLAY_COLUMNS = [
        "dataset_name", "category", "doi", "url", "year", "access_type", "institution", "country",
        "modality", "resolution", "subject_no_f", "slice_scan_no", "age_range", "disease",
        "segmentation_mask", "healthy_control", "staging_information", "clinical_data_score",
        "histopathology", "lab_data", "notes"
    ]

    # --- Tool 1: Hybrid Dataset Finder ---
    def hybrid_dataset_finder(user_query: str) -> str:
        parser_prompt_template = """
        You are an expert query parser. Your job is to deconstruct the user's query and map it to a structured JSON filter based on the available options.
        User Query: "{query}"
        Available options for filtering:
        - category: {category_options}
        - disease: {disease_options}
        - access_type: {access_type_options}
        - modality: {modality_options}
        - segmentation_mask: {segmentation_mask_options}
        - institution: {institution_options}
        - country: {country_options}
        - format: {format_options}
        - healthy_control: {healthy_control_options}
        - staging_information: {staging_information_options}
        - clinical_data_score: {clinical_data_score_options}
        - histopathology: {histopathology_options}
        - lab_data: {lab_data_options}
        Analyze the user's query and translate it into a JSON object with a 'filters' key. For numerical fields like 'year' or 'subject_no_f', create a sub-object with 'operator' (e.g., '>', '<=', '==') and 'value'. If a filter is not mentioned, omit it. Respond with ONLY the JSON object.
        """
        parser_prompt = PromptTemplate.from_template(parser_prompt_template)
        parser_chain = parser_prompt | _llm
        
        all_options = {
            'category': _sheet_names,
            'disease': _combined_df['disease'].dropna().unique().tolist(),
            'access_type': _combined_df['access_type'].dropna().unique().tolist(),
            'modality': _combined_df['modality'].dropna().unique().tolist(),
            'segmentation_mask': _combined_df['segmentation_mask'].dropna().unique().tolist(),
            'institution': _combined_df['institution'].dropna().unique().tolist(),
            'country': _combined_df['country'].dropna().unique().tolist(),
            'format': _combined_df['format'].dropna().unique().tolist(),
            'healthy_control': _combined_df['healthy_control'].dropna().unique().tolist(),
            'staging_information': _combined_df['staging_information'].dropna().unique().tolist(),
            'clinical_data_score': _combined_df['clinical_data_score'].dropna().unique().tolist(),
            'histopathology': _combined_df['histopathology'].dropna().unique().tolist(),
            'lab_data': _combined_df['lab_data'].dropna().unique().tolist(),
        }

        # Dynamically create the input dictionary for the prompt
        prompt_input = {"query": user_query}
        for key, value in all_options.items():
            prompt_input[f"{key}_options"] = value

        filter_json_str = parser_chain.invoke(prompt_input).content
        
        filtered_df = _combined_df.copy()
        try:
            clean_json_str = re.sub(r"```json\n?|```", "", filter_json_str).strip()
            filters = json.loads(clean_json_str).get("filters", {})
            if not filters:
                return json.dumps({"result": "I couldn't identify any specific search criteria."})

            for key, value in filters.items():
                if isinstance(value, dict) and 'operator' in value and 'value' in value:
                    op, val = value['operator'], value['value']
                    col_to_filter = 'subject_no_f' if 'subjects' in key else key
                    if col_to_filter in filtered_df.columns:
                        filtered_df[col_to_filter] = pd.to_numeric(filtered_df[col_to_filter], errors='coerce')
                        if op == '>': filtered_df = filtered_df[filtered_df[col_to_filter] > val]
                        elif op == '>=': filtered_df = filtered_df[filtered_df[col_to_filter] >= val]
                        elif op == '<': filtered_df = filtered_df[filtered_df[col_to_filter] < val]
                        elif op == '<=': filtered_df = filtered_df[filtered_df[col_to_filter] <= val]
                        elif op == '==': filtered_df = filtered_df[filtered_df[col_to_filter] == val]
                elif key in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[key].astype(str).str.contains(str(value), case=False, na=False)]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            return json.dumps({"result": f"I had trouble parsing your request. Error: {e}"})

        if filtered_df.empty:
            return json.dumps({"result": "No datasets found that match your specific criteria."})

        data_as_dict = filtered_df.to_dict(orient='records')
        return json.dumps({"count": len(filtered_df), "data": data_as_dict})

    # --- Tool 2: Category Summarizer ---
    def get_category_summary(user_query: str) -> str:
        category_finder_prompt = PromptTemplate.from_template(
            "You are a classification assistant. From the list {categories}, find the single best match for the user query: '{query}'. Respond with only the category name or 'None'."
        )
        finder_chain = category_finder_prompt | _llm
        target_category = finder_chain.invoke({"categories": _sheet_names, "query": user_query}).content.strip()

        if target_category not in _dataframes:
            return json.dumps({"summary": f"I couldn't find the category you asked for. Available categories are: {_sheet_names}", "category_name": None})
        
        df = _dataframes[target_category]
        summary_prompt_template = """
        You are a data summarization expert. Based on the provided data, create a concise, one-paragraph summary.
        - From the list of diseases, identify and list the top 4 primary conditions, summarizing them cleanly.
        - From the list of modalities, identify and list the top 2 unique modalities.
        DATA:
        - Category Name: {category}
        - Total Datasets: {count}
        - Year Range: {min_year} - {max_year}
        - List of Diseases: {diseases}
        - List of Modalities: {modalities}
        Generate the summary in this exact format:
        "The {category} category contains {count} datasets. They primarily focus on conditions like [Top 4 summarized diseases], using modalities such as [Top 2 unique modalities], with data published between {min_year} and {max_year}."
        """
        summary_prompt = PromptTemplate.from_template(summary_prompt_template)
        summary_chain = summary_prompt | _llm
        summary = summary_chain.invoke({
            "category": target_category, "count": len(df),
            "min_year": int(df['year'].min()) if not df['year'].empty else 'N/A',
            "max_year": int(df['year'].max()) if not df['year'].empty else 'N/A',
            "diseases": df['disease'].dropna().unique().tolist(), 
            "modalities": df['modality'].dropna().unique().tolist()
        }).content
        return json.dumps({"summary": summary, "category_name": target_category})

    # --- Tool 3: Plotting Tool (Modified for Streamlit) ---
    def create_chart(data_query: str, chart_type: str = 'bar') -> str:
        """Generates a chart based on a pandas query. Does not save or show."""
        try:
            data_series = eval(data_query, {"combined_df": _combined_df, "pd": pd})
            fig, ax = plt.subplots(figsize=(10, 6))
            chart_type = chart_type.lower().strip()

            if chart_type == 'bar':
                sns.barplot(x=data_series.index, y=data_series.values, palette="viridis", ax=ax)
                ax.set_title(f'Bar Chart: {data_query}', fontsize=14)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            elif chart_type == 'pie':
                ax.pie(data_series.values, labels=data_series.index, autopct='%1.1f%%', startangle=140)
                ax.set_title(f'Pie Chart: {data_query}', fontsize=14)
                ax.set_ylabel('')
            elif chart_type == 'line':
                sns.lineplot(x=data_series.index, y=data_series.values, marker='o', ax=ax)
                ax.set_title(f'Line Chart: {data_query}', fontsize=14)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                return f"Error: Unsupported chart type '{chart_type}'. Please use 'bar', 'line', or 'pie'."
            
            plt.tight_layout()
            # IMPORTANT: We don't save or show; Streamlit will handle the figure object.
            return f"Chart '{chart_type}' successfully generated for display."
        except Exception as e:
            # We don't close the plot, so Streamlit doesn't get an empty figure
            return f"Error creating chart: {e}. The input must be a valid pandas command."

    def charting_wrapper(query_string: str) -> str:
        """Wrapper for create_chart. Input: 'chart_type|pandas_query'."""
        try:
            parts = query_string.split('|', 1)
            if len(parts) != 2: return "Error: Invalid input format. Expected 'chart_type|pandas_query'."
            chart_type, data_query = parts[0].strip(), parts[1].strip()
            if not data_query: return "Error: 'pandas_query' part of the input is empty."
            return create_chart(data_query=data_query, chart_type=chart_type)
        except Exception as e:
            return f"An unexpected error occurred in the plotting wrapper: {e}"

    # --- Tool Definitions ---
    hybrid_finder_tool = Tool(name="hybrid_dataset_finder", func=hybrid_dataset_finder, description="Use this as the primary tool to find specific datasets based on criteria like category, disease, access type, modality, institution, country, format, segmentation_mask, etc. The input should be the user's full query.")
    category_summary_tool = Tool(name="category_summarizer", func=get_category_summary, description="Use this tool ONLY when the user asks for a general overview, summary, or a full list of datasets for a broad category.")
    python_repl_tool = Tool(name="python_code_interpreter", func=PythonAstREPLTool(locals={"pd": pd, "combined_df": _combined_df}).run, description="CRITICAL: Use this tool for any questions that involve ranking, comparison, or calculation (e.g., 'most', 'least', 'highest', 'compare', 'how many'). This tool is for performing Python-based analysis on the 'combined_df' pandas DataFrame to answer a question. The input MUST be a valid Python command.")
    plotting_tool = Tool(name="chart_generator", func=charting_wrapper, description="Use this to create and display a chart from data. The input MUST be a single string separated by a pipe `|` in the format: 'chart_type|pandas_query'. Example: \"pie|combined_df['access_type'].value_counts()\"")

    tools = [hybrid_finder_tool, category_summary_tool, python_repl_tool, plotting_tool]

    # --- Agent Prompt ---
    prompt_template = """
    You are NeuroAI, a helpful and friendly assistant for exploring neuroradiology datasets. Your goal is to answer user questions accurately by using the tools provided.
    You have access to the following tools: {tools}
    To use a tool, please use the following format:
    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```
    When you have a response to say to the user, or if you do not need to use a tool, you MUST use the format:
    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```
    --- IMPORTANT RULES ---
    1. For "summary" or "overview" of a category, use `category_summarizer`.
    2. To *find* or *list* specific datasets, use `hybrid_dataset_finder`.
    3. To "plot", "chart", "graph", or "visualize", use `chart_generator`. The input MUST be a single string 'chart_type|pandas_query'. Example: "pie|combined_df['access_type'].value_counts()"
    4. For ranking/comparison ("most", "highest", "compare"), FIRST use `hybrid_dataset_finder` to get relevant data, THEN use `python_code_interpreter` to analyze it.
    
    Begin!

    Previous conversation history (last 5 turns):
    {chat_history}

    New input: {input}
    {agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # --- Agent Executor ---
    agent = create_react_agent(_llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True,
        handle_parsing_errors="I'm sorry, I had trouble processing that request. Please try rephrasing.",
        return_intermediate_steps=True,
        max_iterations=7
    )
    return agent_executor, ALL_DISPLAY_COLUMNS

agent_executor, ALL_DISPLAY_COLUMNS = setup_agent(llm, combined_df, dataframes, sheet_names)


# --- STREAMLIT CHAT UI ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you explore the neuroradiology datasets today?"}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "table" in msg and msg["table"] is not None:
            st.dataframe(msg["table"])
        if "chart" in msg and msg["chart"] is not None:
            st.pyplot(msg["chart"])

# Handle user input
if user_query := st.chat_input("Ask about datasets, request a summary, or ask for a plot..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                inputs = {"input": user_query, "chat_history": st.session_state.memory.load_memory_variables({})['chat_history']}
                response = agent_executor.invoke(inputs)

                assistant_message = {"role": "assistant"}
                final_answer = response.get('output', "I'm sorry, I encountered an issue.")
                assistant_message["content"] = final_answer
                st.markdown(final_answer)

                if 'intermediate_steps' in response and response['intermediate_steps']:
                    last_action, last_observation = response['intermediate_steps'][-1]
                    
                    if last_action.tool == 'hybrid_dataset_finder':
                        tool_output = json.loads(last_observation)
                        if 'data' in tool_output:
                            df = pd.DataFrame(tool_output['data'])
                            df_display = df[[col for col in ALL_DISPLAY_COLUMNS if col in df.columns]]
                            st.dataframe(df_display)
                            assistant_message["table"] = df_display

                    elif last_action.tool == 'category_summarizer':
                        tool_output = json.loads(last_observation)
                        target_category = tool_output.get("category_name")
                        if target_category and target_category in dataframes:
                            df = dataframes[target_category]
                            df_display = df[[col for col in ALL_DISPLAY_COLUMNS if col in df.columns]]
                            st.dataframe(df_display)
                            assistant_message["table"] = df_display
                    
                    elif last_action.tool == 'chart_generator' and "Error" not in last_observation:
                        fig = plt.gcf()
                        st.pyplot(fig)
                        assistant_message["chart"] = fig

                st.session_state.memory.save_context(inputs, {"output": final_answer})
                st.session_state.messages.append(assistant_message)

            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
