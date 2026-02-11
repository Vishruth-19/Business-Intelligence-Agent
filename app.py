import streamlit as st
import pandas as pd
import requests
from openai import OpenAI
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Founder BI Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #e8f4f8;
        border-left: 4px solid #2ecc71;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #666;
        margin-left: auto;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df_deals' not in st.session_state:
    st.session_state.df_deals = pd.DataFrame()
if 'df_orders' not in st.session_state:
    st.session_state.df_orders = pd.DataFrame()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- MONDAY.COM FUNCTIONS ---
def fetch_board_data(board_id, api_key):
    if not board_id or not api_key:
        return pd.DataFrame()
    
    url = "https://api.monday.com/v2"
    headers = {"Authorization": api_key, "API-Version": "2023-10"}
    query = f"""query {{ boards (ids: {board_id}) {{ items_page (limit: 500) {{ items {{ name column_values {{ column {{ title }} text }} }} }} }} }}"""
    
    try:
        response = requests.post(url, json={'query': query}, headers=headers)
        data = response.json()
        items = data['data']['boards'][0]['items_page']['items']
        
        processed_data = []
        for item in items:
            row = {"Item Name": item['name']}
            for col in item['column_values']:
                row[col['column']['title']] = col['text']
            processed_data.append(row)
        
        return pd.DataFrame(processed_data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def clean_data(df):
    if df.empty:
        return df
    
    df.columns = [c.strip() for c in df.columns]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().astype(str).iloc[0] if not df[col].dropna().empty else ""
            if any(x in sample for x in ['$', '€', '£']) or sample.replace('.', '', 1).isdigit():
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                    errors='coerce'
                ).fillna(0)
    
    return df

# --- AI AGENT FUNCTION ---
def ask_agent(query, df_deals, df_orders, openrouter_key):
    if not openrouter_key:
        return "⚠️ Please enter your OpenRouter API Key in the sidebar."
    
    # Initialize OpenRouter Client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Founder BI Agent"
        }
    )
    
    # Prepare data context
    deals_head = df_deals.head(3).to_markdown(index=False) if not df_deals.empty else "No data"
    orders_head = df_orders.head(3).to_markdown(index=False) if not df_orders.empty else "No data"
    
    prompt = f"""
I have two pandas DataFrames: `df_deals` and `df_orders`.

Sample df_deals:
{deals_head}

Sample df_orders:
{orders_head}

User Query: "{query}"

Task: Write Python code to answer this question.
- Use variable `result` for the final answer
- Output ONLY the code inside ```python ``` blocks
- Make the result clear and concise
- If creating a table, convert to markdown or dict
"""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = completion.choices[0].message.content
        
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0].strip()
            local_vars = {
                'df_deals': df_deals,
                'df_orders': df_orders,
                'pd': pd
            }
            
            exec(code, {}, local_vars)
            result = local_vars.get('result')
            
            # Format the result nicely
            if isinstance(result, pd.DataFrame):
                return f"**Answer:**\n\n{result.to_markdown(index=False)}\n\n*Analysis by gpt-oss-120b*"
            else:
                return f"**Answer:** {result}\n\n*Analysis by gpt-oss-120b*"
        
        return text
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease try rephrasing your question or check the data."

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔌 System Status")
    
    # Check secrets
    if "OPENROUTER_API_KEY" in st.secrets and "MONDAY_API_KEY" in st.secrets:
        OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
        MONDAY_API_KEY = st.secrets["MONDAY_API_KEY"]
        DEALS_ID = st.secrets["DEALS_ID"]
        WORK_ORDERS_ID = st.secrets["WORK_ORDERS_ID"]
        
        st.success("✅ Connected to Monday.com")
        st.success("✅ AI Intelligence Active")
    else:
        OPENROUTER_API_KEY = st.text_input("OpenRouter API Key", type="password")
        MONDAY_API_KEY = st.text_input("Monday.com API Key", type="password")
        DEALS_ID = st.text_input("Deals Board ID")
        WORK_ORDERS_ID = st.text_input("Work Orders Board ID")
    
    st.divider()
    
    # Data sync section
    st.header("📊 Data Management")
    
    if st.button("🔄 Sync with Monday.com", use_container_width=True):
        with st.spinner("Syncing data..."):
            st.session_state.df_deals = clean_data(fetch_board_data(DEALS_ID, MONDAY_API_KEY))
            st.session_state.df_orders = clean_data(fetch_board_data(WORK_ORDERS_ID, MONDAY_API_KEY))
            st.session_state.data_loaded = True
            st.rerun()
    
    if st.session_state.data_loaded:
        st.info(f"📈 Deals: {len(st.session_state.df_deals)} records")
        st.info(f"📋 Orders: {len(st.session_state.df_orders)} records")
        
        # Show last sync time
        if 'last_sync' not in st.session_state:
            st.session_state.last_sync = datetime.now()
        st.caption(f"Last synced: {st.session_state.last_sync.strftime('%I:%M %p')}")
    
    st.divider()
    
    # Chat controls
    st.header("💬 Chat Controls")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📥 Export Chat", use_container_width=True):
        if st.session_state.messages:
            chat_export = "\n\n".join([
                f"{'USER' if msg['role'] == 'user' else 'ASSISTANT'} [{msg['timestamp']}]:\n{msg['content']}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                label="Download Chat",
                data=chat_export,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    st.divider()
    
    # Quick insights
    if st.session_state.data_loaded:
        st.header("💡 Quick Insights")
        with st.expander("Sample Questions"):
            st.markdown("""
            - What's the total value of all deals?
            - How many work orders are pending?
            - Show me the top 5 deals by value
            - What's the average deal size?
            - How many deals closed this month?
            - Compare deals vs work orders
            """)

# --- MAIN UI ---
st.title("🚀 Founder's BI Companion")
st.caption("Ask questions about your deals and work orders in natural language")

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">
                    👤 You
                    <span class="timestamp">{message['timestamp']}</span>
                </div>
                <div>{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">
                    🤖 BI Assistant
                    <span class="timestamp">{message['timestamp']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(message['content'])

# Input area (fixed at bottom)
st.divider()

# Check if data is loaded
if not st.session_state.data_loaded:
    st.warning("⚠️ Please sync with Monday.com first using the sidebar button.")
    st.stop()

# Chat input
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Ask a question about your data:",
        placeholder="e.g., What's the total value of all deals?",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_button = st.button("Send 📤", use_container_width=True)

# Process input
if send_button and user_input:
    # Add user message
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Get AI response
    with st.spinner("🤔 Analyzing your data..."):
        response = ask_agent(
            user_input,
            st.session_state.df_deals,
            st.session_state.df_orders,
            OPENROUTER_API_KEY
        )
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })
    
    # Update last sync time
    st.session_state.last_sync = datetime.now()
    
    # Rerun to show new messages
    st.rerun()

# Show data preview at bottom (collapsible)
with st.expander("📊 View Raw Data"):
    tab1, tab2 = st.tabs(["Deals", "Work Orders"])
    
    with tab1:
        if not st.session_state.df_deals.empty:
            st.dataframe(st.session_state.df_deals, use_container_width=True)
        else:
            st.info("No deals data loaded")
    
    with tab2:
        if not st.session_state.df_orders.empty:
            st.dataframe(st.session_state.df_orders, use_container_width=True)
        else:
            st.info("No work orders data loaded")
