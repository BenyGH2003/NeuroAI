import streamlit as st
import pandas as pd
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain.prompts import PromptTemplate
from groq import Groq
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time
import regex as re
import json
import io
import PyPDF2
from tavily import TavilyClient
from tabulate import tabulate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
SERPER_API_KEY = os.getenv('SERPER_API_KEY', 'edf28dbbb85930e14c617ad0eb0479799de050c1')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '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')
client = Groq(api_key=GROQ_API_KEY)

# Define the updated state structure
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

# Updated extraction prompt
EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Extract the following information from the provided neuroradiology dataset paper text.
Your response should be concise and structured. If information is not found, return 'Not specified'.

Now, extract the following details from the given text:

1. Dataset Name - The official name of the dataset.
2. DOI - The Digital Object Identifier (DOI) if available.
3. URL - Link to access the dataset (if provided).
4. Year of Release - The year the dataset was published.
5. Access Type - Whether the dataset is open-access, restricted, etc.
6. Institution - The university, research lab, or company that created the dataset.
7. Country - The country of the institution.
8. Modality - The imaging type (e.g., MRI, CT, X-ray).
9. Resolution - The imaging resolution (e.g., voxel size).
10. Number of Subjects (Female) - The number of total subjects with female subjects in parenthesis e.g 80(30).
11. Number of Slices/Scans - How many images/scans are available.
12. Age Range - The age range of subjects.
13. Acquisition Protocol - Details of how images were acquired.
14. Format - The file format (e.g., DICOM, NIfTI, MHA, PNG, JPG etc.).
15. Segmentation Mask - Whether segmentation masks are included (Yes/No) with additional information about if the masks are derived automatically or manual.
16. Preprocessing - Explain preprocessing steps on the dataset.
17. Disease - The main disease(s) studied in the dataset.
18. Healthy Control - Whether healthy controls are included (Yes/No).
19. Staging Information - Disease staging details if available.
20. Clinical Data, Score - Whether clinical data or scores are included (if yes, specify what).
21. Histopathology - Whether histopathology data is included (Yes/No).
22. Lab Data - Whether lab data (e.g., blood tests) is included (Yes/No).

Text to Analyze:
{text}

Output Format:
- Dataset Name: [answer]
- DOI: [answer]
- URL: [answer]
...

Final Instructions for AI:
- Extract information only from the provided text (do not assume details).
- If a field is missing, explain why instead of just saying `"Not specified"`.
- Verify that numerical values (year, subject count) are accurate.
"""
)

def fetch_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        if url.lower().endswith(".pdf"):
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        else:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            text = " ".join(soup.stripped_strings)
        return text
    except Exception as e:
        return f"Error fetching text from {url}: {str(e)}"

def split_text_smart(text, max_tokens=5000):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def extract_info(state: DatasetState) -> DatasetState:
    prompt = EXTRACTION_PROMPT.format(text=state["paper_text"])
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an expert in extracting structured data from scientific texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5000
        )
        result = response.choices[0].message.content
    except Exception as e:
        result = "\n".join([f"{k}: Not specified" for k in DatasetState.__annotations__ if k != "paper_text"])

    lines = result.split("\n")
    field_mapping = {
        "dataset_name": "dataset_name",
        "doi": "doi",
        "url": "url",
        "year_of_release": "year",
        "access_type": "access_type",
        "institution": "institution",
        "country": "country",
        "modality": "modality",
        "resolution": "resolution",
        "number_of_subjects_(female)": "subject_no_f",
        "number_of_slices_scans": "slice_scan_no",
        "age_range": "age_range",
        "acquisition_protocol": "acquisition_protocol",
        "format": "format",
        "segmentation_mask": "segmentation_mask",
        "preprocessing": "preprocessing",
        "disease": "disease",
        "healthy_control": "healthy_control",
        "staging_information": "staging_information",
        "clinical_data,_score": "clinical_data_score",
        "histopathology": "histopathology",
        "lab_data": "lab_data"
    }
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip("- ").lower().replace(" ", "_").replace("/", "_")
            if key in field_mapping:
                state[field_mapping[key]] = value.strip()
            elif key in DatasetState.__annotations__:
                state[key] = value.strip()
    return state

def format_output(state: DatasetState) -> DatasetState:
    fields = [
        "dataset_name", "doi", "url", "year", "access_type", "institution", "country",
        "modality", "resolution", "subject_no_f", "slice_scan_no", "age_range",
        "acquisition_protocol", "format", "segmentation_mask", "preprocessing", "disease",
        "healthy_control", "staging_information", "clinical_data_score",
        "histopathology", "lab_data"
    ]
    for field in fields:
        if field not in state or state[field] is None:
            state[field] = "Not specified"
    return state

workflow = StateGraph(DatasetState)
workflow.add_node("extract_info", extract_info)
workflow.add_node("format_output", format_output)
workflow.add_edge("extract_info", "format_output")
workflow.add_edge("format_output", END)
workflow.set_entry_point("extract_info")
app = workflow.compile()

def ensemble_results(chunk_results):
    ensembled = {}
    fields = [k for k in DatasetState.__annotations__ if k != "paper_text"]
    for field in fields:
        values = [result.get(field, "Not specified") for result in chunk_results]
        for value in values:
            if value != "Not specified":
                ensembled[field] = value
                break
        else:
            ensembled[field] = "Not specified"
    return ensembled

def analyze_paper(paper_text: str) -> dict:
    chunks = split_text_smart(paper_text, max_tokens=5000)
    chunk_results = []
    for chunk in chunks:
        initial_state = {"paper_text": chunk}
        result = app.invoke(initial_state)
        chunk_results.append({k: v for k, v in result.items() if k != "paper_text"})
    return ensemble_results(chunk_results)

def serper_search(query: str, api_key: str, num_results: int = 10, page: int = 1) -> Optional[pd.DataFrame]:
    url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    payload = {"q": query, "num": num_results, "page": page}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
        organic_results = results.get('organic', [])
        processed = [
            {'Position': item.get('position'), 'Title': item.get('title'), 'URL': item.get('link')}
            for item in organic_results
        ]
        return pd.DataFrame(processed)
    except Exception as e:
        st.error(f"Serper search failed: {str(e)}")
        return pd.DataFrame()

def paginated_search(query: str, api_key: str, max_pages: int = 3) -> pd.DataFrame:
    all_results = pd.DataFrame()
    page = 1
    num_per_page = 10
    while page <= max_pages:
        st.write(f"Searching Serper page {page}...")
        results_df = serper_search(query, api_key, num_per_page, page)
        if results_df is None or results_df.empty:
            st.write("No more Serper results found or error occurred")
            break
        all_results = pd.concat([all_results, results_df], ignore_index=True)
        page += 1
        time.sleep(1)
    return all_results

def get_tavily_results(query: str, api_key: str, max_results: int = 30) -> pd.DataFrame:
    try:
        tavily_client = TavilyClient(api_key=api_key)
        response = tavily_client.search(query, max_results=max_results)
        results = response['results']
        df = pd.DataFrame([{'Position': i+1, 'Title': r['title'], 'URL': r['url']} 
                           for i, r in enumerate(results)])
        st.write(f"Tavily search returned {len(df)} results")
        return df
    except Exception as e:
        st.error(f"Tavily search failed: {str(e)}")
        return pd.DataFrame()

def get_combined_results(query: str, serper_api_key: str = None, tavily_api_key: str = None, 
                        max_pages: int = 3, max_tavily_results: int = 30) -> pd.DataFrame:
    dfs = []
    if serper_api_key:
        serper_df = paginated_search(query, serper_api_key, max_pages=max_pages)
        dfs.append(serper_df)
    if tavily_api_key:
        tavily_df = get_tavily_results(query, tavily_api_key, max_results=max_tavily_results)
        dfs.append(tavily_df)
    if dfs:
        combined_df = pd.concat(dfs).drop_duplicates(subset=['URL']).reset_index(drop=True)
        st.write(f"Combined unique results from Serper and Tavily: {len(combined_df)} URLs")
        return combined_df
    st.write("No results from either Serper or Tavily")
    return pd.DataFrame()

def process_search_results(search_results: pd.DataFrame, max_urls: int = 5) -> pd.DataFrame:
    urls = search_results['URL'].head(max_urls).tolist()
    texts = []
    for idx, url in enumerate(urls):
        st.write(f"Processing URL {idx+1}/{len(urls)}: {url}")
        text = fetch_text_from_url(url)
        texts.append(text)
        time.sleep(2)
    results = [analyze_paper(text) for text in texts if text and not text.startswith("Error fetching text from")]
    df = pd.DataFrame(results)
    df['url'] = [url for url, text in zip(urls, texts) if text and not text.startswith("Error fetching text from")]
    return df

def update_dataset(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_name", "doi", "url", "year", "access_type", "institution", "country",
        "modality", "resolution", "subject_no_f", "slice_scan_no", "age_range",
        "acquisition_protocol", "format", "segmentation_mask", "preprocessing", "disease",
        "healthy_control", "staging_information", "clinical_data_score",
        "histopathology", "lab_data"
    ]
    existing_df = existing_df.reindex(columns=columns, fill_value="Not specified")
    new_df = new_df.reindex(columns=columns, fill_value="Not specified")

    def is_duplicate_within(df):
        seen_keys = set()
        duplicates = []
        for idx, row in df.iterrows():
            primary_key = (row['dataset_name'], row['url'], row['doi'])
            is_duplicate = any(k in seen_keys for k in primary_key if k != "Not specified")
            duplicates.append(is_duplicate)
            if not is_duplicate:
                seen_keys.update(k for k in primary_key if k != "Not specified")
        return pd.Series(duplicates)

    new_duplicates = is_duplicate_within(new_df)
    new_df_cleaned = new_df[~new_duplicates].reset_index(drop=True)
    st.write(f"New data after internal deduplication: {len(new_df_cleaned)} rows (from {len(new_df)})")

    combined_df = pd.concat([existing_df, new_df_cleaned], ignore_index=True)
    st.write(f"Combined dataframe size: {len(combined_df)} rows")

    existing_keys = set()
    for _, row in existing_df.iterrows():
        primary_key = (row['dataset_name'], row['url'], row['doi'])
        existing_keys.update(k for k in primary_key if k != "Not specified")

    def is_duplicate_against_existing(row):
        primary_key = (row['dataset_name'], row['url'], row['doi'])
        return any(k in existing_keys for k in primary_key if k != "Not specified")

    combined_duplicates = []
    for idx, row in combined_df.iterrows():
        if idx < len(existing_df):
            combined_duplicates.append(False)
        else:
            combined_duplicates.append(is_duplicate_against_existing(row))

    cleaned_df = combined_df[~pd.Series(combined_duplicates)].reset_index(drop=True)
    st.write(f"Final cleaned dataframe size: {len(cleaned_df)} rows")
    return cleaned_df

def search_dataset(df: pd.DataFrame, modality: str, disease: str, segmentation: str, access_type: str) -> dict:
    relevant_columns = [
        'dataset_name', 'doi', 'url', 'year', 'access_type', 'institution', 'country',
        'modality', 'resolution', 'subject_no_f', 'slice_scan_no', 'age_range',
        'acquisition_protocol', 'format', 'segmentation_mask', 'preprocessing', 'disease',
        'healthy_control', 'staging_information', 'clinical_data_score', 'histopathology', 'lab_data'
    ]
    filtered_df = df[relevant_columns].head(20)
    dataset_entries = [
        f"Dataset: {row['dataset_name']} | Modality: {row['modality']} | Disease: {row['disease']} | "
        f"Segmentation: {row['segmentation_mask']} | Access: {row['access_type']} | URL: {row['url']}"
        for _, row in filtered_df.iterrows()
    ]
    dataset_context = "\n".join(dataset_entries)

    search_prompt = f"""
    You are a medical imaging dataset expert. Your task is to find datasets matching specific criteria.

    USER CRITERIA:
    - Modality: {modality}
    - Disease: {disease}
    - Segmentation Required: {segmentation}
    - Access Type: {access_type}

    AVAILABLE DATASETS:
    {dataset_context}

    INSTRUCTIONS:
    1. Find exact and partial matches based on the criteria
    2. Return ONLY a valid JSON object with this exact structure:
    {{
        "matches": [
            {{
                "dataset_name": "exact name from list",
                "url": "exact url from list",
                "relevance": "one sentence explanation"
            }}
        ],
        "alternatives": [
            {{
                "dataset_name": "exact name from list",
                "url": "exact url from list",
                "explanation": "one sentence explanation"
            }}
        ]
    }}
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a precise JSON-generating assistant."},
                {"role": "user", "content": search_prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        parsed_results = json.loads(result)
        return parsed_results
    except Exception as e:
        st.error(f"Error querying LLM: {str(e)}")
        return perform_basic_matching(filtered_df, modality, disease, segmentation, access_type)

def perform_basic_matching(df, modality, disease, segmentation, access_type):
    matches = []
    alternatives = []
    for _, row in df.iterrows():
        if (modality.lower() in str(row['modality']).lower() and
            disease.lower() in str(row['disease']).lower() and
            segmentation.lower() in str(row['segmentation_mask']).lower() and
            access_type.lower() in str(row['access_type']).lower()):
            matches.append({
                "dataset_name": row['dataset_name'],
                "url": row['url'],
                "relevance": "Matches all criteria"
            })
        elif (modality.lower() in str(row['modality']).lower() or
              disease.lower() in str(row['disease']).lower()):
            alternatives.append({
                "dataset_name": row['dataset_name'],
                "url": row['url'],
                "explanation": "Partial match on modality or disease"
            })
    return {"matches": matches, "alternatives": alternatives}

# Streamlit App
def main():
    st.title("NeuroAI: Neuroradiology Imaging Dataset Finder")

    # Sidebar for Update Database
    with st.sidebar:
        st.header("Database Controls")
        categories = ['Neurodegenerative', 'Neoplasm', 'Cerebrovascular', 'Psychiatric', 'Spinal', 'Neurodevelopmental']
        selected_category = st.selectbox("Select Category to Update", categories)
        if st.button("Update Database"):
            with st.spinner(f"Updating the {selected_category} Database..."):
                dataset_file = 'dataset.xlsx'
                if os.path.exists(dataset_file):
                    excel_book = pd.read_excel(dataset_file, sheet_name=None)
                    st.write(f"Loaded existing dataset with sheets: {list(excel_book.keys())}")
                    existing_df = excel_book.get(selected_category, pd.DataFrame(columns=[
                        "dataset_name", "doi", "url", "year", "access_type", "institution", "country",
                        "modality", "resolution", "subject_no_f", "slice_scan_no", "age_range",
                        "acquisition_protocol", "format", "segmentation_mask", "preprocessing", "disease",
                        "healthy_control", "staging_information", "clinical_data_score",
                        "histopathology", "lab_data"
                    ]))
                    st.write(f"Loaded {len(existing_df)} rows from {selected_category}")
                else:
                    excel_book = {cat: pd.DataFrame(columns=[
                        "dataset_name", "doi", "url", "year", "access_type", "institution", "country",
                        "modality", "resolution", "subject_no_f", "slice_scan_no", "age_range",
                        "acquisition_protocol", "format", "segmentation_mask", "preprocessing", "disease",
                        "healthy_control", "staging_information", "clinical_data_score",
                        "histopathology", "lab_data"
                    ]) for cat in categories}
                    existing_df = excel_book[selected_category]
                    st.write("No existing dataset found, initializing empty dataframe")

                SEARCH_QUERY = f"{selected_category.lower()} imaging dataset"
                search_results = get_combined_results(SEARCH_QUERY, SERPER_API_KEY, TAVILY_API_KEY)
                if not search_results.empty:
                    new_df = process_search_results(search_results, max_urls=5)
                    st.write(f"Found {len(new_df)} new rows from search")
                    updated_df = update_dataset(existing_df, new_df)
                    excel_book[selected_category] = updated_df
                    with pd.ExcelWriter(dataset_file, engine='openpyxl', mode='w') as writer:
                        for sheet, df in excel_book.items():
                            df.to_excel(writer, sheet_name=sheet, index=False)
                    st.success(f"Dataset updated and saved to {dataset_file}, sheet: {selected_category}")
                    st.session_state['excel_book'] = excel_book
                else:
                    st.warning(f"No new datasets found for {selected_category}, using existing dataset.")
                    st.session_state['excel_book'] = excel_book

    # Load dataset
    dataset_file = 'dataset.xlsx'
    categories = ['Neurodegenerative', 'Neoplasm', 'Cerebrovascular', 'Psychiatric', 'Spinal', 'Neurodevelopmental']
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
    st.header("Chat with NeuroAI Agent")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [{"role": "assistant", "content": f"Here is NeuroAI agent, I can help you find your neuroradiology imaging dataset.\n\nI have datasets in multiple categories: {', '.join(categories)}\nWhich category are you interested in?"}]
    if 'step' not in st.session_state:
        st.session_state['step'] = "category"
    if 'criteria' not in st.session_state:
        st.session_state['criteria'] = {}

    # Display chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Your response:")
    if prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process based on current step
        if st.session_state['step'] == "category":
            if prompt.capitalize() not in categories:
                response = f"Invalid category. Please choose from: {', '.join(categories)}\nWhich category are you interested in?"
                st.session_state['step'] = "category"
            else:
                response = f"Great, let's find a {prompt.capitalize()} imaging dataset for you!\nWhat modality do you want? (e.g., MRI, CT)"
                st.session_state['step'] = "modality"
            st.session_state['criteria']['category'] = prompt.capitalize()

        elif st.session_state['step'] == "modality":
            st.session_state['criteria']['modality'] = prompt
            response = f"What {st.session_state['criteria']['category'].lower()} disease are you interested in?"
            st.session_state['step'] = "disease"

        elif st.session_state['step'] == "disease":
            st.session_state['criteria']['disease'] = prompt
            response = "Do you need segmentation masks? (Yes/No)"
            st.session_state['step'] = "segmentation"

        elif st.session_state['step'] == "segmentation":
            st.session_state['criteria']['segmentation'] = prompt
            response = "What type of access do you prefer? (e.g., open, restricted)"
            st.session_state['step'] = "access_type"

        elif st.session_state['step'] == "access_type":
            st.session_state['criteria']['access_type'] = prompt
            with st.spinner("Searching for datasets..."):
                df = st.session_state['excel_book'][st.session_state['criteria']['category']]
                results = search_dataset(
                    df,
                    st.session_state['criteria']['modality'],
                    st.session_state['criteria']['disease'],
                    st.session_state['criteria']['segmentation'],
                    st.session_state['criteria']['access_type']
                )
                response = ""
                if results["matches"]:
                    response += "**Best Matching Datasets:**\n\n"
                    match_names = [match["dataset_name"] for match in results["matches"]]
                    matches_df = df[df['dataset_name'].isin(match_names)][[
                        'dataset_name', 'doi', 'url', 'year', 'access_type', 'institution', 'country',
                        'modality', 'resolution', 'subject_no_f', 'slice_scan_no', 'age_range',
                        'acquisition_protocol', 'format', 'segmentation_mask', 'preprocessing', 'disease',
                        'healthy_control', 'staging_information', 'clinical_data_score', 'histopathology', 'lab_data'
                    ]].rename(columns={
                        'dataset_name': 'Name', 'doi': 'DOI', 'url': 'URL', 'year': 'Year',
                        'access_type': 'Access', 'institution': 'Institution', 'country': 'Country',
                        'modality': 'Modality', 'resolution': 'Resolution', 'subject_no_f': 'Subjects (F)',
                        'slice_scan_no': 'Slices/Scans', 'age_range': 'Age Range', 'acquisition_protocol': 'Protocol',
                        'format': 'Format', 'segmentation_mask': 'Segmentation', 'preprocessing': 'Preprocessing',
                        'disease': 'Disease', 'healthy_control': 'Healthy?', 'staging_information': 'Staging',
                        'clinical_data_score': 'Clinical Data', 'histopathology': 'Histopath?', 'lab_data': 'Lab Data?'
                    })
                    response += f"```\n{tabulate(matches_df, headers='keys', tablefmt='fancy_grid', showindex=False)}\n```\n"
                else:
                    response += "No exact matches found.\n"
                if results["alternatives"]:
                    response += "**Alternative Suggestions:**\n\n"
                    alt_names = [alt["dataset_name"] for alt in results["alternatives"]]
                    alternatives_df = df[df['dataset_name'].isin(alt_names)][[
                        'dataset_name', 'doi', 'url', 'year', 'access_type', 'institution', 'country',
                        'modality', 'resolution', 'subject_no_f', 'slice_scan_no', 'age_range',
                        'acquisition_protocol', 'format', 'segmentation_mask', 'preprocessing', 'disease',
                        'healthy_control', 'staging_information', 'clinical_data_score', 'histopathology', 'lab_data'
                    ]].rename(columns={
                        'dataset_name': 'Name', 'doi': 'DOI', 'url': 'URL', 'year': 'Year',
                        'access_type': 'Access', 'institution': 'Institution', 'country': 'Country',
                        'modality': 'Modality', 'resolution': 'Resolution', 'subject_no_f': 'Subjects (F)',
                        'slice_scan_no': 'Slices/Scans', 'age_range': 'Age Range', 'acquisition_protocol': 'Protocol',
                        'format': 'Format', 'segmentation_mask': 'Segmentation', 'preprocessing': 'Preprocessing',
                        'disease': 'Disease', 'healthy_control': 'Healthy?', 'staging_information': 'Staging',
                        'clinical_data_score': 'Clinical Data', 'histopathology': 'Histopath?', 'lab_data': 'Lab Data?'
                    })
                    response += f"```\n{tabulate(alternatives_df, headers='keys', tablefmt='fancy_grid', showindex=False)}\n```\n"
                if not results["matches"] and not results["alternatives"]:
                    response += "No datasets found matching your criteria. Try broadening your search parameters.\n"
                response += "\nWould you like to search again? (y/n)"
            st.session_state['step'] = "repeat"

        elif st.session_state['step'] == "repeat":
            if prompt.lower() == 'y':
                response = f"I have datasets in multiple categories: {', '.join(categories)}\nWhich category are you interested in?"
                st.session_state['step'] = "category"
                st.session_state['criteria'] = {}
            else:
                response = "Goodbye!"
                st.session_state['step'] = "done"

        if st.session_state['step'] != "done":
            st.session_state['chat_history'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            st.session_state['chat_history'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state['step'] = "category"
            st.session_state['chat_history'] = [{"role": "assistant", "content": f"Here is NeuroAI agent, I can help you find your neuroradiology imaging dataset.\n\nI have datasets in multiple categories: {', '.join(categories)}\nWhich category are you interested in?"}]

if __name__ == "__main__":
    main()
