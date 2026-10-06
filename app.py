"""
Streamlit chat UI for the IPL WAR analytics agent.

Run with:
    streamlit run app.py
"""

import streamlit as st

import config

st.set_page_config(page_title="IPL WAR Analyst", page_icon="🏏", layout="centered")

SAMPLE_QUESTIONS = [
    "What is Virat Kohli's career WAR?",
    "How does MS Dhoni perform against spin vs pace?",
    "Compare Jasprit Bumrah and Rashid Khan in 2023.",
    "Who had the highest total WAR in the 2016 season?",
    "How does Yuzvendra Chahal do against left-handed batters?",
    "What are Rishabh Pant's keeping numbers?",
]

# ---------------------------------------------------------------------------
# Sidebar — local LLM status + data status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")
    st.subheader("Local LLM (Ollama)")

    import agent as _agent_status  # cheap import, just for is_available()/MODEL
    ollama_ok, ollama_msg = _agent_status.is_available()
    if ollama_ok:
        st.success(f"Connected — model '{_agent_status.MODEL}' ready.")
    else:
        st.error(ollama_msg)
        st.caption(
            "No API key needed — this runs fully offline on your machine. "
            "Install Ollama (ollama.com), then `ollama pull "
            f"{_agent_status.MODEL}`."
        )
    if st.button("Recheck connection"):
        st.rerun()

    st.divider()
    st.subheader("Data layer")

    data_ready = (
        config.BATTING_WAR_CSV.exists()
        and config.BOWLING_WAR_CSV.exists()
        and config.FIELDING_WAR_CSV.exists()
        and config.ENRICHED_DELIVERIES_PARQUET.exists()
    )
    if data_ready:
        st.success("WAR CSVs + enriched deliveries found.")
    else:
        st.warning("Data not built yet.")
    if st.button("Rebuild data layer" if data_ready else "Build data layer"):
        import data_layer
        with st.spinner("Running WAR pipelines and building enriched deliveries..."):
            data_layer.build_all(force=False)
        st.rerun()

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.pop("display_history", None)
        st.session_state.pop("agent_messages", None)
        st.rerun()

st.title("🏏 IPL WAR Analyst")
st.caption(
    "Ask about player WAR, batting/bowling splits, matchups, comparisons, "
    "leaderboards, or keeper stats — built on CricSheet ball-by-ball data. "
    "Runs on a local LLM (Ollama) — no API key, nothing leaves this machine."
)

if not data_ready:
    st.info("Build the data layer from the sidebar before asking questions.")
    st.stop()

if not ollama_ok:
    st.info("Set up Ollama (see sidebar) before asking questions.")
    st.stop()

agent = _agent_status  # already imported above for the status check

if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

with st.expander("Sample questions"):
    for q in SAMPLE_QUESTIONS:
        st.markdown(f"- {q}")

for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("tool_calls"):
            with st.expander("Data used to answer this"):
                for call in turn["tool_calls"]:
                    st.markdown(f"**{call['tool']}**`({call['input']})`")
                    st.json(call["output"])

question = st.chat_input("Ask a question about IPL WAR...")

if question:
    st.session_state.display_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking up the data..."):
            try:
                result = agent.ask(question, history=st.session_state.agent_messages)
            except Exception as e:
                result = {"answer": f"Something went wrong calling the local model: {e}",
                          "messages": st.session_state.agent_messages, "tool_calls": []}
        st.markdown(result["answer"])
        if result["tool_calls"]:
            with st.expander("Data used to answer this"):
                for call in result["tool_calls"]:
                    st.markdown(f"**{call['tool']}**`({call['input']})`")
                    st.json(call["output"])

    st.session_state.agent_messages = result["messages"]
    st.session_state.display_history.append({
        "role": "assistant",
        "content": result["answer"],
        "tool_calls": result["tool_calls"],
    })
