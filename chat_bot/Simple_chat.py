from typing import Any, Dict, List

import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


def init_page(header: str):
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header(header)


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    temperture = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)

    return ChatOpenAI(
        temperature=temperture,
        model_name=model_name,
        streaming=True,
    )


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = 0


def update_cost(new_cost):
    global total_cost
    st.session_state.costs += new_cost


class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """Copied only streaming part from StreamlitCallbackHandler"""

    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)


def main():
    init_page(header="My ChatGPT 🤗")

    llm = select_model()
    init_messages()

    # チャット履歴の表示
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("user").markdown(user_input)
        with get_openai_callback() as cb:
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = llm(messages, callbacks=[st_callback])
                new_cost = cb.total_cost
                update_cost(new_cost)
        st.session_state.messages.append(AIMessage(content=response.content))

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${st.session_state.costs:.4f}**")


if __name__ == "__main__":
    main()
