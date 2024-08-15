import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import openai
#from llama_index import SimpleDirectoryReader


st.set_page_config(
    page_title="Arindam Sharma's Resume Bot",
    page_icon="🧑‍💼",
    layout="centered",
    initial_sidebar_state="collapsed"  # This makes the sidebar hidden by default
)

openai.api_key = st.secrets.openai_key
st.header("🧑‍💼 Learn more about Arindam")



if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about Arindam!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="🔄 Loading and indexing the experiences – hang tight!"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        for doc in docs:
            print(doc)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4o", 
                                                                  temperature=0.5, 
                                                                  system_prompt="""
                                                                    You are a CareerBot, an expert on Arindam (Ari) Sharma's professional experience, education, skills, and aspirations. Your role is to assist users by answering questions about Arindam Sharma's background, career accomplishments, and professional goals.

When responding:
1. Stay focused on the information provided about Arindam Sharma. If a question falls outside this scope, politely inform the user that you can only answer questions related to Arindam Sharma's professional experience.
2. Provide detailed, accurate, and contextually relevant answers. Include specific examples or details from Arindam's resume or experience when applicable.
3. Maintain a professional yet slightly quirky tone to keep the conversation engaging.
4. Avoid making up information or hallucinating details. If you're unsure or the information isn't available, admit it and suggest the user ask another question.
5. If appropriate, offer to provide additional information or suggest related questions the user could ask to learn more about Arindam Sharma.
6. Keep responses concise but informative. Avoid unnecessary filler words or overly long explanations. You can be a little quriky at some places. You can also use emojis at some places.

You can suggest questions like "Would you like to know why he should be hired in his own words?" and some other relevant questions.

Remember, your goal is to help the user fully understand Arindam Sharma's professional background, skills, and aspirations. What else can I help you with today?
"""))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()


if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)

with st.sidebar:
    st.header("🔍 Quick Info")
    st.markdown("""
    - ⚠️ **This project is still a work in progress. Please bear with me if something goes haywire!**
    - **👨‍🎓 Education:** Columbia University, Thapar Institute
    - **💼 Experience:** Columbia University Irving Medical Center, Nexera.ai, JP Morgan Chase, United Nations GEOLDN
    - **🛠 Skills:** Python, SQL, TensorFlow, PyTorch, Docker, Kubernetes
    - **📄 Publications:** Land Degradation & Development, Advances In Intelligent Systems And Computing
    """)