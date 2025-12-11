import streamlit as st
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Import LangChain components directly
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("API_KEY", ""),
    base_url="https://api.ai.it.cornell.edu",
)

CHAT_MODEL = "openai.gpt-4o-mini"
EMBEDDING_MODEL = "openai.text-embedding-3-large"
OPENAI_BASE_URL = "https://api.ai.it.cornell.edu"

st.set_page_config(
    page_title="Fantasy Football AI Assistant",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 Fantasy Football AI Assistant")
st.markdown(
    """
    Get comprehensive fantasy football insights combining:
    - 📊 **Player Statistics** - Season-long performance and advanced metrics
    - 🏥 **Injury Reports** - Real-time NFL injury updates
    - 🎯 **Smart Analysis** - Data-driven lineup recommendations
    """
)


# Initialize Vector DB Search
@st.cache_resource
def init_vector_search():
    """Initialize vector database search (cached)"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        chroma_dir = Path("./chroma_db")
        if not chroma_dir.exists():
            return None
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.environ.get("API_KEY"),
            openai_api_base=OPENAI_BASE_URL
        )
        
        # Connect to existing vector store
        vectorstore = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
            collection_name="fantasy_players"
        )
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None


vector_search = init_vector_search()


@st.cache_data
def load_injuries():
    """Load injury data from CSV."""
    try:
        df = pd.read_csv("data/injuries.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


injuries_df = load_injuries()

# Sidebar
st.sidebar.header("📋 System Status")

# Vector DB status
if vector_search:
    try:
        # Get count from vectorstore
        collection = vector_search._collection
        player_count = collection.count()
        st.sidebar.success(f"✅ Stats DB: {player_count:,} players")
    except:
        st.sidebar.success("✅ Stats DB: Ready")
else:
    st.sidebar.warning("⚠️ Stats DB: Not loaded")
    with st.sidebar.expander("How to enable"):
        st.markdown("""
        1. Run: `python build_vector_db_langchain.py`
        2. Wait 5-10 minutes for completion
        3. Refresh this page
        """)

# Injury Report status
st.sidebar.markdown("---")
st.sidebar.header("🏥 Injury Report")

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

# Quick player lookup in sidebar
if vector_search:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Quick Player Lookup")
    
    player_name = st.sidebar.text_input("Search player:")
    if player_name:
        with st.sidebar:
            with st.spinner("Searching..."):
                # Use vectorstore similarity_search
                results = vector_search.similarity_search(player_name, k=3)
                if results:
                    st.success(f"Found {len(results)} matches:")
                    for doc in results[:3]:
                        meta = doc.metadata
                        with st.expander(f"{meta.get('player_name', 'Unknown')} ({meta.get('position', '?')}, {meta.get('team', '?')})"):
                            # Show key stats
                            if 'total_fantasy_points_ppr' in meta:
                                st.metric("Fantasy Pts (PPR)", f"{meta['total_fantasy_points_ppr']:.1f}")
                            
                            st.text(doc.page_content)
                else:
                    st.info("No players found")


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


def search_player_stats(query: str, n_results: int = 5) -> str:
    """Search player stats using vector database"""
    if not vector_search:
        return "Player statistics database not available. Enable it by running 'python build_vector_db_langchain.py'"

    try:
        # Use vectorstore similarity_search
        results = vector_search.similarity_search(query, k=n_results)

        if not results:
            return f"No players found matching: {query}"

        # Format results for LLM context
        context_parts = [f"PLAYER STATISTICS (matching '{query}'):"]

        for i, doc in enumerate(results, 1):
            context_parts.append(f"\n{i}. {doc.page_content}")

        return "\n".join(context_parts)

    except Exception as e:
        return f"Error searching player stats: {e}"


def answer_question(question: str, history: list, injuries_df: pd.DataFrame) -> str:
    """Generate answer using both injury data and player stats context."""

    # Build injury context
    injury_context = build_injury_context(injuries_df)

    # Search for relevant player stats using semantic search
    stats_context = ""
    if vector_search:
        # Extract key terms from question for better search
        stats_context = search_player_stats(question, n_results=5)
    else:
        stats_context = "Player statistics database not available."

    # Combine contexts
    full_context = f"""INJURY REPORT:
{injury_context}

---

{stats_context}
"""

    system_prompt = f"""You are an expert Fantasy Football Assistant with access to comprehensive NFL data.

You have access to:
1. **Current injury reports** - Real-time injury status (Out, Doubtful, Questionable)
2. **Player statistics database** - Season performance, advanced metrics, NextGen Stats

CURRENT DATA:
{full_context}

Your role:
- Provide data-driven fantasy football advice
- Combine injury information with player performance metrics
- Be specific with numbers and statistics when available
- Explain your reasoning clearly
- If a player is injured, consider their backup or replacement
- Use advanced metrics like EPA, efficiency, yards over expected when relevant

Guidelines:
- **Prioritize player health** - Check injury status first
- **Support with stats** - Use performance data to back recommendations  
- **Consider context** - Matchups, recent trends, usage patterns
- **Be concise** - Clear and actionable advice
- **Cite sources** - Reference specific stats when making claims
- **Acknowledge gaps** - If you don't have info, say so clearly
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
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}\n\nMake sure API_KEY environment variable is set."


# Quick Questions Section
st.markdown("---")
st.subheader("🎯 Quick Questions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Top RBs"):
        st.session_state.setdefault("messages", []).append(
            {
                "role": "user",
                "content": "Who are the top 5 running backs by fantasy points this season?",
            }
        )
        st.rerun()

with col2:
    if st.button("Injury Impact"):
        st.session_state.setdefault("messages", []).append(
            {
                "role": "user",
                "content": "Which injured players will have the biggest fantasy impact this week?",
            }
        )
        st.rerun()

with col3:
    if st.button("Sleeper Picks"):
        st.session_state.setdefault("messages", []).append(
            {
                "role": "user",
                "content": "Suggest undervalued players based on advanced metrics like efficiency and yards over expected",
            }
        )
        st.rerun()

with col4:
    if st.button("Compare QBs"):
        st.session_state.setdefault("messages", []).append(
            {
                "role": "user",
                "content": "Compare the top 3 quarterbacks by passing yards and EPA",
            }
        )
        st.rerun()

# Advanced Stats Explorer (only if vector DB is available)
if vector_search:
    st.markdown("---")
    st.subheader("📊 Advanced Stats Explorer")

    col1, col2 = st.columns([2, 1])

    with col1:
        semantic_query = st.text_input(
            "Search:",
            placeholder="e.g., explosive playmakers, red zone threats, efficient passers",
        )

    with col2:
        position_filter = st.selectbox(
            "Filter by position:", ["All", "QB", "RB", "WR", "TE", "K"]
        )

    if semantic_query:
        pos_filter = None if position_filter == "All" else position_filter
        
        with st.spinner("Searching player database..."):
            # Use vectorstore with optional filter
            if pos_filter:
                results = vector_search.similarity_search(
                    semantic_query, 
                    k=10, 
                    filter={"position": pos_filter}
                )
            else:
                results = vector_search.similarity_search(semantic_query, k=10)

        if results:
            st.write(f"**Found {len(results)} players:**")

            for i, doc in enumerate(results, 1):
                meta = doc.metadata
                with st.expander(
                    f"{i}. {meta.get('player_name', 'Unknown')} ({meta.get('position', '?')}, {meta.get('team', '?')})"
                ):
                    # Show full player description
                    st.text(doc.page_content)
                    
                    # Add separator
                    st.markdown("---")
                    
                    # Show key metadata metrics
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        if 'total_fantasy_points_ppr' in meta:
                            st.metric("Fantasy Pts (PPR)", f"{meta['total_fantasy_points_ppr']:.1f}")
                    
                    with col_b:
                        if 'games_played' in meta:
                            st.metric("Games Played", int(meta['games_played']))
                    
                    with col_c:
                        if 'avg_fantasy_points_ppr_per_game' in meta:
                            st.metric("Pts Per Game", f"{meta['avg_fantasy_points_ppr_per_game']:.1f}")
        else:
            st.info("No results found. Try a different search.")

# Chat Interface
st.markdown("---")
st.subheader("💬 Ask about injuries and player performance")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I'm your Fantasy Football AI Assistant. I have access to:\n\n"
            "📊 **Player statistics** (season performance, advanced metrics)\n"
            "🏥 **Injury reports** (real-time status updates)\n\n"
            "Ask me about player injuries, performance analysis, lineup decisions, or matchup advice!",
        }
    ]


def needs_response():
    if len(st.session_state.messages) == 0:
        return False
    return st.session_state.messages[-1]["role"] == "user"


if needs_response():
    pending_question = st.session_state.messages[-1]["content"]
    with st.spinner("Analyzing player data..."):
        bot_reply = answer_question(
            question=pending_question,
            history=st.session_state.messages[:-1],
            injuries_df=injuries_df,
        )
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input("Ask about players, injuries, stats, or lineup advice...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Analyzing player data..."):
        bot_reply = answer_question(
            question=user_question,
            history=st.session_state.messages[:-1],
            injuries_df=injuries_df,
        )
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()

st.markdown("---")
# st.caption(
#     "💡 Powered by LangChain RAG with OpenAI embeddings (text-embedding-3-large) "
#     "and real-time injury data. Stats updated for 2025 season."
# )