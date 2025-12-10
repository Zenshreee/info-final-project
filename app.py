import streamlit as st
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("API_KEY", ""),
    base_url="https://api.ai.it.cornell.edu",
)

CHAT_MODEL = "openai.gpt-4o-mini"

st.set_page_config(
    page_title="Fantasy Football Injury Assistant",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 Fantasy Football Injury Assistant")
st.markdown(
    """
    Get real-time NFL injury updates to help with your fantasy football decisions.  
    Ask about player injuries, game statuses, and lineup advice.
    """
)


@st.cache_data
def load_injuries():
    """Load injury data from CSV."""
    try:
        df = pd.read_csv("injuries.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


injuries_df = load_injuries()

st.sidebar.header("📋 Injury Report")

if not injuries_df.empty:
    st.sidebar.metric("Total Players Listed", len(injuries_df))

    if "status" in injuries_df.columns:
        out_count = len(injuries_df[injuries_df["status"] == "Out"])
        questionable_count = len(injuries_df[injuries_df["status"] == "Questionable"])
        doubtful_count = len(injuries_df[injuries_df["status"] == "Doubtful"])

        st.sidebar.metric("Out", out_count)
        st.sidebar.metric("Questionable", questionable_count)
        st.sidebar.metric("Doubtful", doubtful_count)

    st.sidebar.subheader("Filter by Team")
    if "team" in injuries_df.columns:
        teams = ["All"] + sorted(injuries_df["team"].dropna().unique().tolist())
        selected_team = st.sidebar.selectbox("Select Team", teams)

        if selected_team != "All":
            filtered_df = injuries_df[injuries_df["team"] == selected_team]
            st.sidebar.dataframe(
                filtered_df[["player", "injury", "status"]].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.sidebar.warning("No injury data loaded. Run the scraper first.")


def build_injury_context(df: pd.DataFrame) -> str:
    """Build context string from injury data for the LLM."""
    if df.empty:
        return "No injury data available."

    context_parts = []

    out_players = df[df["status"] == "Out"]
    if not out_players.empty:
        out_list = [
            f"- {row['player']} ({row['team']}, {row['position']}): {row['injury']}"
            for _, row in out_players.iterrows()
        ]
        context_parts.append("PLAYERS OUT:\n" + "\n".join(out_list[:20]))

    doubtful = df[df["status"] == "Doubtful"]
    if not doubtful.empty:
        doubtful_list = [
            f"- {row['player']} ({row['team']}, {row['position']}): {row['injury']}"
            for _, row in doubtful.iterrows()
        ]
        context_parts.append("PLAYERS DOUBTFUL:\n" + "\n".join(doubtful_list[:20]))

    questionable = df[df["status"] == "Questionable"]
    if not questionable.empty:
        q_list = [
            f"- {row['player']} ({row['team']}, {row['position']}): {row['injury']}"
            for _, row in questionable.iterrows()
        ]
        context_parts.append("PLAYERS QUESTIONABLE:\n" + "\n".join(q_list[:20]))

    return "\n\n".join(context_parts)


def get_player_info(player_name: str, df: pd.DataFrame) -> str:
    """Get specific player injury info."""
    if df.empty:
        return ""

    mask = df["player"].str.lower().str.contains(player_name.lower(), na=False)
    matches = df[mask]

    if matches.empty:
        return f"No injury information found for '{player_name}'."

    info = []
    for _, row in matches.iterrows():
        info.append(
            f"{row['player']} ({row['team']}, {row['position']}): "
            f"{row['injury']} - Status: {row['status']}"
        )
    return "\n".join(info)


def answer_question(question: str, history: list, injuries_df: pd.DataFrame) -> str:
    """Generate answer using injury data context."""

    injury_context = build_injury_context(injuries_df)

    system_prompt = f"""You are a helpful Fantasy Football Assistant specializing in NFL injuries.
Use the provided injury report data to answer questions about player injuries, game statuses, 
and provide fantasy football advice based on injury information.

CURRENT NFL INJURY REPORT (Week 15):
{injury_context}

Guidelines:
- Be concise and helpful for fantasy football decisions
- If a player isn't in the injury report, they're likely healthy
- Mention injury type and game status (Out, Doubtful, Questionable)
- For fantasy advice, consider the player's injury severity and matchup
- If you don't have information, say so clearly
"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}\n\nMake sure API_KEY environment variable is set."


st.subheader("💬 Ask about injuries")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I'm your Fantasy Football Injury Assistant. Ask me about player injuries, "
            "who's out this week, or get advice on your lineup decisions!",
        }
    ]


def needs_response():
    if len(st.session_state.messages) == 0:
        return False
    return st.session_state.messages[-1]["role"] == "user"


if needs_response():
    pending_question = st.session_state.messages[-1]["content"]
    with st.spinner("Checking injury reports..."):
        bot_reply = answer_question(
            question=pending_question,
            history=st.session_state.messages[:-1],
            injuries_df=injuries_df,
        )
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input("Ask about injuries (e.g., 'Is CeeDee Lamb playing?')")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Checking injury reports..."):
        bot_reply = answer_question(
            question=user_question,
            history=st.session_state.messages[:-1],
            injuries_df=injuries_df,
        )
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()

st.markdown("---")
st.subheader("🔍 Quick Lookups")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Who's OUT this week?"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": "Who are the notable players ruled OUT this week?",
            }
        )
        st.rerun()

with col2:
    if st.button("Questionable WRs"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": "Which wide receivers are questionable this week?",
            }
        )
        st.rerun()

with col3:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")
# st.caption(
#     "Data sources: NFL.com Official Injury Report, @UnderdogNFL Twitter. "
#     "Run `python run_scraper.py` to update injury data."
# )
