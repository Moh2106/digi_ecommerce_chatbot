import streamlit as st
from llama_index.core import Settings
from Settings import AppSettings
from ProductsRAG import RagTools

#
AppSettings = AppSettings.AppSettings() # Object for the AppSettings class
Settings.llm = AppSettings.get_llm() # To initialise llm
Settings.embed_model = AppSettings.get_embeddings() # for embeddings

image_path = "data/digishare.png"


st.set_page_config(
    page_title="Digishare chatbot",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Title
import base64

with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Créer l'URL en base64
image_base64 = f"data:image/png;base64,{encoded_image}"

st.markdown(
       f"""
       <div style="text-align: center;">
           <img src="{image_base64}" style="
               box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
               width: 50%;
               object-fit : contain;
               height: auto;
           " alt="Example Image">
       </div>
       """,
       unsafe_allow_html=True
   )



if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello how can I help you ?",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the product docs - hang tight! This should take 1-2 minutes."):
        tools = RagTools.RagTools()
        data_path = "data/marba_shop_new.pdf"

        # Put the data in the vectorstore
        index = tools.get_router_query_engine(file_path=data_path,embed_model=Settings.embed_model)
        return index



index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(llm=Settings.llm, chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Writing ..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history