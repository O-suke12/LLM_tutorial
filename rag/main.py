import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage  # ChatGPTの返答
from langchain.schema import HumanMessage  # 人間の質問
from langchain.schema import SystemMessage  # システムメッセージ


def main():
    llm = ChatOpenAI(temperature=0.5, max_tokens=100)

    st.set_page_config(page_title="Hello Streamlit", page_icon="😇")
    st.header("myGPT")

    if "message_history" not in st.session_state:
        st.session_state.message_history = [SystemMessage(content="Welcome to myGPT!")]

    if user_input := st.text_input("Enter some text"):
        st.session_state.message_history.append(HumanMessage(text=user_input))
        with st.spinner("Wait for it..."):
            responese = llm.predict(st.session_state.input)
        st.session_state.message_history.append(AIMessage(text=response))

    container = st.container()
    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area("Massage", key="input", height=100)
            submit_button = st.form_submit_button(label="Submit")

    messages = st.session_state.get("message_history", [])
    for message in messages:
        if isinstance(message, HumanMessage):
            st.write(f"👩: {message.text}")
        elif isinstance(message, AIMessage):
            st.write(f"🤖: {message.text}")
        elif isinstance(message, SystemMessage):
            st.write(f"💬: {message.text}")
        else:
            raise ValueError("Unknown message type")


if __name__ == "__main__":
    main()
