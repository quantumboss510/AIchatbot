import os
import json
import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import random

os.environ["GOOGLE_API_KEY"] = "API_KEY"

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")


####  GENERAL QUERRRY

def general(input=''):
    try:
        results = model.invoke(input)
        print(results.content)
        return results.content
    except Exception as e:
        return "ERROR"


general_tool = Tool(
    name="General Tool",
    func=general,
    description="Handles general querries using google's gemini model"

)


###  AGENT 2 math ops

def math_func(input_text):
    try:
        result = eval(input_text)
        return f"the result of {input_text} is {result}"
    except Exception as e:
        return "ERROR"


math_tool = Tool(
    name="Math Agent",
    func=math_func,
    description="handles math operations such  as solving expressions"
)


### AGENT 3: FUN FACT


def facts_func(input_text):
    facts_list = {
        "Honey is sweet",
        "she is wet",
        "manvith is garu",
        "Girish is playBoy"
    }
    return random.choice(list((facts_list)))


facts_tool = Tool(
    name="Facts Agent",
    func=facts_func,
    description="handles facts and provide random facts"
)

# CONVERSATION MEMORY

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=3,
    return_messages=True
)

# Converational agent with alll tools and memory to bridge

conversational_agent = initialize_agent(
    agent="chat-conversational-react-description",  # Removed the extra space before 'chat'
    tools=[general_tool, math_tool, facts_tool],
    llm=model,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory
)

st.title(" MULTI AGENT AI Chatbot")
st.markdown("Ask me about:")

st.markdown("""
-general querry
-math problem
-fun facts
""")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("you:", "")

if user_input:
    st.session_state.messages.append(f"YOu: {user_input}")
    response = conversational_agent.run(user_input)
    st.session_state.messages.append(f"Bot: {response}")

    for message in st.session_state.messages:
        st.write(message)
