import streamlit as st
import pandas as pd
import requests
import os
from groq import Groq
from bs4 import BeautifulSoup
import time

# Hardcoded API keys (replace with your actual keys or use secrets in Colab)
GROQ_API_KEY = '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'
SERPER_API_KEY = 'edf28dbbb85930e14c617ad0eb0479799de050c1'
client = Groq(api_key=GROQ_API_KEY)

# --- Helper Functions for Database Update ---
def serper_search(query, api_key, page=1):
    """Perform a search using the Serper API."""
    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "page": page,
        "num": 10
    }
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def paginated_search(query, api_key, max_pages=3):
    """Search for datasets across multiple pages."""
    all_results = []
    for page in range(1, max_pages + 1):
        results = serper_search(query, api_key, page)
        organic = results.get("organic", [])
        if not organic:
            break
        all_results.extend(organic)
        time.sleep(1)  # Avoid hitting rate limits
    return pd.DataFrame(all_results)

def fetch_text_from_url(url):
    """Fetch text content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))
    except Exception:
        return ""

def process_search_results(search_df, max_urls=5):
    """Process search results to extract dataset info."""
    columns = ["dataset_name", "doi", "url", "year", "access_type", "institution", 
               "country", "modality", "subject", "slice_scan_no", "format", 
               "segmentation_mask", "disease"]
    new_df = pd.DataFrame(columns=columns)
    
    for _, row in search_df.head(max_urls).iterrows():
        url = row["link"]
        text = fetch_text_from_url(url)
        # Simplified extraction (in practice, use Grok or regex for accuracy)
        entry = {
            "dataset_name": row["title"],
            "doi": "",  # Placeholder
            "url": url,
            "year": "",  # Placeholder
            "access_type": "open",  # Default
            "institution": "",
            "country": "",
            "modality": "MRI" if "MRI" in text else "CT" if "CT" in text else "",
            "subject": "spinal",
            "slice_scan_no": "",
            "format": "",
            "segmentation_mask": "Yes" if "segmentation" in text.lower() else "No",
            "disease": "stenosis" if "stenosis" in text.lower() else "metastatic disease" if "metastatic" in text.lower() else ""
        }
        new_df = pd.concat([new_df, pd.DataFrame([entry])], ignore_index=True)
    return new_df

def update_dataset(existing_df, new_df):
    """Update the existing dataset with new entries, avoiding duplicates."""
    if existing_df.empty:
        return new_df
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    return combined_df.drop_duplicates(subset=["url"], keep="first")

# --- Search Function for Chat Interface ---
def search_dataset(df, modality, disease, segmentation, access_type):
    """Search the dataset based on user criteria."""
    matches = df.copy()
    if modality:
        matches = matches[matches["modality"].str.lower() == modality.lower()]
    if disease:
        matches = matches[matches["disease"].str.lower() == disease.lower()]
    if segmentation.lower() in ["yes", "no"]:
        matches = matches[matches["segmentation_mask"].str.lower() == segmentation.lower()]
    if access_type:
        matches = matches[matches["access_type"].str.lower() == access_type.lower()]
    return matches.to_dict("records")

# --- Streamlit App Layout ---
st.title("NeuroAI - Spinal Imaging Dataset Search")

# Update Database Button at the Top
st.subheader("Database Management")
if st.button("Update Database"):
    with st.spinner("Updating database... Please wait."):
        search_results = paginated_search("spinal imaging dataset", SERPER_API_KEY, max_pages=3)
        if not search_results.empty:
            new_df = process_search_results(search_results, max_urls=5)
            if os.path.exists("dataset.xlsx"):
                existing_df = pd.read_excel("dataset.xlsx", sheet_name="spinal")
            else:
                existing_df = pd.DataFrame(columns=["dataset_name", "doi", "url", "year", "access_type", 
                                                    "institution", "country", "modality", "subject", 
                                                    "slice_scan_no", "format", "segmentation_mask", "disease"])
            updated_df = update_dataset(existing_df, new_df)
            with pd.ExcelWriter("dataset.xlsx", engine="openpyxl", mode="w") as writer:
                updated_df.to_excel(writer, sheet_name="spinal", index=False)
            st.success("Database updated successfully!")
        else:
            st.info("No new datasets found.")

# Load the Database
if os.path.exists("dataset.xlsx"):
    df = pd.read_excel("dataset.xlsx", sheet_name="spinal")
else:
    df = pd.DataFrame(columns=["dataset_name", "doi", "url", "year", "access_type", 
                               "institution", "country", "modality", "subject", 
                               "slice_scan_no", "format", "segmentation_mask", "disease"])
    st.warning("No database found. Please update the database first.")
st.write(f"Current database contains {len(df)} datasets.")

# Chat Interface in the Middle
st.subheader("Chat with NeuroAI Agent")
if "step" not in st.session_state:
    st.session_state.step = 0
if "inputs" not in st.session_state:
    st.session_state.inputs = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

questions = [
    "What modality do you want? (e.g., MRI, CT)",
    "What spine disease are you interested in? (e.g., metastatic disease, stenosis)",
    "Do you need segmentation masks? (Yes/No)",
    "What type of access do you prefer? (e.g., open, restricted)"
]

# Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle Conversation Steps
if st.session_state.step < 4:
    question = questions[st.session_state.step]
    with st.chat_message("assistant"):
        st.write(question)

user_input = st.chat_input("Type your response here")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    if st.session_state.step < 4:
        key = ["modality", "disease", "segmentation", "access_type"][st.session_state.step]
        st.session_state.inputs[key] = user_input
        st.session_state.step += 1
    elif st.session_state.step == 4:
        matches = search_dataset(df, st.session_state.inputs["modality"], 
                                st.session_state.inputs["disease"], 
                                st.session_state.inputs["segmentation"], 
                                st.session_state.inputs["access_type"])
        if not matches:
            response = "No datasets match your requirements."
        else:
            response = "Here are the matching datasets:\n" + "\n".join(
                [f"- {match['dataset_name']}: {match['url']}" for match in matches]
            )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": "Would you like to search again? (yes/no)"})
        st.session_state.step = 5
    elif st.session_state.step == 5:
        if user_input.lower() == "yes":
            st.session_state.step = 0
            st.session_state.inputs = {}
            st.session_state.messages.append({"role": "assistant", "content": "Let's start a new search."})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Goodbye!"})
            st.session_state.step = 6  # End conversation
