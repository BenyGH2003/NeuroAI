# -*- coding: utf-8 -*-
"""NeuroAIHub_Streamlit_App_No_Update.py

A Streamlit web app for the NeuroAIHub agent, allowing users to explore neuroradiology datasets through a conversational interface.
Supports open-ended queries and displays datasets in tabulated format. Uses an existing database without updating.
"""

import streamlit as st
import pandas as pd
import os
from typing import TypedDict, Optional
import json
from tabulate import tabulate
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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" 
)

# Define the dataset state structure
class DatasetState(TypedDict):
    paper_text: str
    dataset_name: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    year: Optional[str]
    access_type: Optional[str]
    institution: Optional[str]
    country: Optional[str]
    modality: Optional[str]
    resolution: Optional[str]
    subject_no_f: Optional[str]
    slice_scan_no: Optional[str]
    age_range: Optional[str]
    acquisition_protocol: Optional[str]
    format: Optional[str]
    segmentation_mask: Optional[str]
    preprocessing: Optional[str]
    disease: Optional[str]
    healthy_control: Optional[str]
    staging_information: Optional[str]
    clinical_data_score: Optional[str]
    histopathology: Optional[str]
    lab_data: Optional[str]

# General-purpose query processing prompt
QUERY_PROMPT = """
You are NeuroAIHub, a friendly and expert assistant for exploring a neuroradiology database. Your goal is to provide clear, accurate, and engaging answers to user queries, grounded in the database's structure, while avoiding assumptions or fabricated details. The database contains categorized datasets with the following structure:

- Categories: Neurodegenerative, Neoplasm, Cerebrovascular, Psychiatric, Spinal, Neurodevelopmental
- Columns: dataset_name, doi, url, year, access_type, institution, country, modality, resolution, subject_no_f (e.g., "80(30)"), slice_scan_no, age_range, acquisition_protocol, format, segmentation_mask, preprocessing, disease, healthy_control, staging_information, clinical_data_score, histopathology, lab_data

Disease-to-category mapping:
- Neoplasm: brain tumor, glioma, glioblastoma, astrocytoma, meningioma, tumor, malignancy, metastasis, metastatic disease
- Neurodegenerative: Alzheimer's, Parkinson's, Multiple Sclerosis
- Cerebrovascular: stroke, aneurysm
- Psychiatric: schizophrenia, ADHD, depression
- Spinal: scoliosis, herniated disc, spinal tumor, metastatic spine disease, spinal malignancy, spinal metastasis
- Neurodevelopmental: autism, ADHD

MRI modality clarification:
- The "modality" column may include specific MRI sequences: T1, T1ce (T1 contrast-enhanced), T1w (T1-weighted), T2, T2w (T2-weighted), FLAIR, DWI. Treat these as subtypes of MRI when users query for "MRI" or "MRI scans."
- Example: If the user asks for "MRI datasets," include datasets with T1, T1ce, T1w, T2, T2w, FLAIR, or DWI in the modality column.

Query: {user_query}

Instructions:
- Provide a warm, concise, natural language response rooted in the database's structure and categories.
- Do not invent dataset names, numbers, or details.
- Recognize synonyms like "tumor," "malignancy," "metastasis," and "metastatic disease" as equivalent to specific diseases (e.g., "spinal tumor" includes "metastatic spine disease") and map to the appropriate category (e.g., Spinal or Neoplasm).
- For dataset queries (e.g., "find spinal tumor datasets"), generate filter conditions (e.g., [{"column": "disease", "value": "spinal tumor|metastatic spine disease|spinal malignancy|spinal metastasis"}, {"column": "category", "value": "Spinal"}]) and include a general explanation of retrieved datasets.
- For specific sequence queries (e.g., "find glioma datasets with FLAIR"), filter by the exact sequence in the modality column.
- For dataset-specific queries (e.g., "what’s the DOI for ADNI?"), filter by "dataset_name" and describe key details (e.g., modality, disease).
- For column value queries (e.g., "what’s the modality of BraTS?"), filter by dataset_name and return the specific column value with context.
- For comparison queries (e.g., "compare ADNI and BraTS patient numbers"), note that direct comparison requires database access but suggest filters and describe typical attributes.
- For descriptive queries (e.g., "tell me about brain tumors"), provide a category/column-based overview with a friendly tone.
- For unclear queries, gently ask for clarification with 1-3 relevant follow-up suggestions.
- Always return a JSON object:
  {
    "response": "friendly natural language response",
    "filters": [{"column": "column_name", "value": "filter_value"}] or [],
    "follow_up_suggestions": ["suggestion 1", "suggestion 2"],
    "dataset_explanation": "general description of retrieved datasets" or ""
  }
- If no dataset matches, state: "I couldn’t find specific datasets for this query, but I can filter the database with these conditions to explore further."
- Ensure filters are accurate, using regex-style patterns (e.g., "T1|T1ce", "spinal tumor|metastatic spine disease") for synonyms and multi-condition queries.

Examples:
- Query: "Is there any spinal tumor dataset?"
  {
    "response": "I’m happy to help you find datasets for spinal tumors in the Spinal category!",
    "filters": [{"column": "disease", "value": "spinal tumor|metastatic spine disease|spinal malignancy|spinal metastasis"}, {"column": "category", "value": "Spinal"}],
    "follow_up_suggestions": ["Want to filter by MRI sequences like T1?", "Interested in datasets with clinical data?"],
    "dataset_explanation": "These datasets cover spinal tumors, including metastatic spine diseases, often with MRI or CT scans, subject demographics, and clinical details."
  }

- Query: "Find glioma datasets with MRI scans"
  {
    "response": "Let’s find glioma datasets with MRI scans in the Neoplasm category for you!",
    "filters": [{"column": "disease", "value": "glioma|tumor|malignancy|metastasis"}, {"column": "modality", "value": "T1|T1ce|T1w|T2|T2w|FLAIR|DWI"}, {"column": "category", "value": "Neoplasm"}],
    "follow_up_suggestions": ["Want to filter by specific sequences like FLAIR?", "Curious about subject counts?"],
    "dataset_explanation": "These datasets use MRI sequences like T1, T2, or FLAIR to study gliomas or related malignancies, including details on subjects and protocols."
  }

- Query: "What’s the modality of BraTS?"
  {
    "response": "Let’s check the modality for the BraTS dataset!",
    "filters": [{"column": "dataset_name", "value": "BraTS"}],
    "follow_up_suggestions": ["Want to know BraTS subject counts?", "Interested in other glioma datasets?"],
    "dataset_explanation": "BraTS typically includes MRI sequences like T1ce, T2, and FLAIR for glioma and malignancy studies."
  }

- Query: "Compare ADNI and BraTS patient numbers"
  {
    "response": "I’d love to compare ADNI and BraTS, but I need database access for exact numbers. Let’s filter these datasets instead!",
    "filters": [{"column": "dataset_name", "value": "ADNI"}, {"column": "dataset_name", "value": "BraTS"}],
    "follow_up_suggestions": ["Want to see subject counts?", "Curious about their modalities?"],
    "dataset_explanation": "ADNI (Alzheimer’s) uses MRI like T1 for longitudinal studies, while BraTS (gliomas/malignancies) uses T1ce and FLAIR for tumor imaging."
  }

- Query: "Tell me about brain tumors"
  {
    "response": "Brain tumors are fascinating! They’re in the Neoplasm category, covering gliomas, malignancies, and more.",
    "filters": [],
    "follow_up_suggestions": ["Want to explore glioma datasets with MRI?", "Curious about metastatic tumor studies?"],
    "dataset_explanation": ""
  }

Note: For unmapped diseases, filter "disease" across all categories unless specified. Use regex-style filters (e.g., "T1|T1ce", "spinal tumor|metastatic spine disease") for MRI subtypes and disease synonyms. Keep responses engaging, accurate, and inclusive of synonym mappings.
"""

def apply_filters(df: pd.DataFrame, filters: list, category: str = None) -> pd.DataFrame:
    filtered_df = df
    for f in filters:
        column = f["column"]
        value = f["value"]
        if column == "category":
            continue  # Category is handled at the DataFrame selection level
        if column in df.columns:
            filtered_df = filtered_df[filtered_df[column].str.contains(value, case=False, na=False)]
    return filtered_df

def process_query(query: str, excel_book: dict, categories: list) -> dict:
    prompt = QUERY_PROMPT.replace("{user_query}", query)
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-distill-llama-70b:free",
            messages=[
                {"role": "system", "content": "You are a precise JSON-generating assistant. Always return a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        result = result.replace("```json", "").replace("```", "").strip()
        parsed_result = json.loads(result)
        return parsed_result
    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error: {str(e)}")
        st.error(f"Raw API Response: {response.choices[0].message.content}")
        return {
            "response": "I didn't quite understand your request. Could you clarify what you're looking for?",
            "filters": [],
            "follow_up_suggestions": ["Are you interested in a specific category like Neoplasm?", "Do you want to search for datasets?"],
            "dataset_explanation": ""
        }
    except Exception as e:
        st.error(f"Query processing failed: {str(e)}")
        return {
            "response": "Something went wrong. Could you try again or clarify your request?",
            "filters": [],
            "follow_up_suggestions": ["Try asking about a category or dataset feature."],
            "dataset_explanation": ""
        }

def format_response(response: str, df: pd.DataFrame = None, follow_up_suggestions: list = [], dataset_explanation: str = "") -> str:
    output = f"{response}\n"
    if dataset_explanation:
        output += f"\n{dataset_explanation}\n"
    if df is not None and not df.empty:
        column_map = {
            'dataset_name': 'Name',
            'doi': 'DOI',
            'url': 'URL',
            'year': 'Year',
            'access_type': 'Access',
            'institution': 'Institution',
            'country': 'Country',
            'modality': 'Modality',
            'resolution': 'Resolution',
            'subject_no_f': 'Subjects (F)',
            'slice_scan_no': 'Slices/Scans',
            'age_range': 'Age Range',
            'acquisition_protocol': 'Protocol',
            'format': 'Format',
            'segmentation_mask': 'Segmentation',
            'preprocessing': 'Preprocessing',
            'disease': 'Disease',
            'healthy_control': 'Healthy?',
            'staging_information': 'Staging',
            'clinical_data_score': 'Clinical Data',
            'histopathology': 'Histopath?',
            'lab_data': 'Lab Data?'
        }
        relevant_columns = list(column_map.keys())
        display_df = df[relevant_columns].rename(columns=column_map)
        output += "\n**Appendix: Dataset Details**\n"
        output += f"```\n{tabulate(display_df, headers='keys', tablefmt='fancy_grid', showindex=False)}\n```\n"
    if follow_up_suggestions:
        output += "\n**What else can I help with?**\n"
        for i, suggestion in enumerate(follow_up_suggestions, 1):
            output += f"- {suggestion}\n"
    return output

# Streamlit App
def main():
    st.title("NeuroAIHub: Neuroradiology Imaging Dataset Finder")
    st.markdown("Explore a rich database of neuroradiology datasets. Ask anything about categories, datasets, or trends!")

    # Remove the sidebar update controls section
    # Sidebar is no longer needed
    # with st.sidebar:
    #     st.header("Database Controls")
    categories = ['Neurodegenerative', 'Neoplasm', 'Cerebrovascular', 'Psychiatric', 'Spinal', 'Neurodevelopmental']
    #     selected_category = st.selectbox("Select Category to Update", categories)
    #     if st.button("Update Database"):
    #         with st.spinner(f"Updating the {selected_category} Database..."):
    #             # Database update code removed...

    # Load the dataset
    dataset_file = 'dataset.xlsx'
    if 'excel_book' not in st.session_state:
        if os.path.exists(dataset_file):
            st.session_state['excel_book'] = pd.read_excel(dataset_file, sheet_name=None)
        else:
            st.session_state['excel_book'] = {cat: pd.DataFrame(columns=[ 
                "dataset_name", "doi", "url", "year", "access_type", "institution", "country",
                "modality", "resolution", "subject_no_f", "slice_scan_no", "age_range",
                "acquisition_protocol", "format", "segmentation_mask", "preprocessing", "disease",
                "healthy_control", "staging_information", "clinical_data_score",
                "histopathology", "lab_data"
            ]) for cat in categories}

    # Chat Interface
    st.header("Chat with NeuroAIHub Agent")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [{"role": "assistant", "content": "Hello! I'm NeuroAIHub, your assistant for exploring neuroradiology datasets. Ask me anything about datasets, categories, or trends!"}]

    # Display chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Ask about neuroradiology datasets:")
    if prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Processing your query..."):
            query_result = process_query(prompt, st.session_state['excel_book'], categories)
            response = query_result["response"]
            filters = query_result["filters"]
            follow_up_suggestions = query_result["follow_up_suggestions"]

            # Apply filters to retrieve datasets if specified
            df = None
            if filters:
                category = next((f["value"] for f in filters if f["column"] == "category"), None)
                if category and category.capitalize() in categories:
                    df = st.session_state['excel_book'][category.capitalize()]
                else:
                    all_dfs = [df for df in st.session_state['excel_book'].values() if not df.empty]
                    df = pd.concat(all_dfs) if all_dfs else pd.DataFrame()
                df = apply_filters(df, filters)

            response_text = format_response(response, df, follow_up_suggestions)
            st.session_state['chat_history'].append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)

if __name__ == "__main__":
    main()
