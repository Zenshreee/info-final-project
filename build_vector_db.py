"""
Build Vector Database for Fantasy Football Player Stats using LangChain
Aggregates player data and creates embeddings with OpenAI
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import os
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Configuration
DATA_DIR = Path("data")
CHROMA_DIR = Path("./chroma_db")
EMBEDDING_MODEL = "openai.text-embedding-3-large"
OPENAI_BASE_URL = "https://api.ai.it.cornell.edu"


class PlayerDataAggregator:
    """Aggregate player statistics into comprehensive player profiles"""
    
    def __init__(self, stats_file: str, nextgen_rush_file: str = None, 
                 nextgen_pass_file: str = None, nextgen_rec_file: str = None):
        self.stats_df = pd.read_csv(stats_file, low_memory=False)
        self.nextgen_rush_df = pd.read_csv(nextgen_rush_file) if nextgen_rush_file else None
        self.nextgen_pass_df = pd.read_csv(nextgen_pass_file) if nextgen_pass_file else None
        self.nextgen_rec_df = pd.read_csv(nextgen_rec_file) if nextgen_rec_file else None
        
    def aggregate_player_stats(self) -> pd.DataFrame:
        """Aggregate stats by player across all weeks"""
        
        print("📊 Aggregating player statistics...")
        
        # Group by player and aggregate
        player_groups = self.stats_df.groupby(['player_id', 'player_display_name', 'position', 'team'])
        
        aggregated_players = []
        
        for (player_id, name, position, team), player_data in tqdm(player_groups, desc="Processing players"):
            player_profile = {
                'player_id': player_id,
                'player_name': name,
                'position': position,
                'team': team,
                'games_played': len(player_data),
                'weeks': player_data['week'].tolist(),
            }
            
            # Aggregate offensive stats
            if position in ['QB', 'RB', 'WR', 'TE']:
                player_profile.update(self._aggregate_offensive_stats(player_data, position))
            
            # Aggregate defensive stats
            if position in ['DT', 'DE', 'LB', 'OLB', 'MLB', 'CB', 'S']:
                player_profile.update(self._aggregate_defensive_stats(player_data))
            
            # Aggregate special teams stats
            if position in ['K', 'P']:
                player_profile.update(self._aggregate_kicker_stats(player_data))
            
            # Add NextGen stats
            nextgen_stats = self._get_all_nextgen_stats(name, position)
            if nextgen_stats:
                player_profile.update(nextgen_stats)
            
            # Create weekly breakdown
            player_profile['weekly_breakdown'] = self._create_weekly_breakdown(player_data)
            
            aggregated_players.append(player_profile)
        
        print(f"✅ Aggregated {len(aggregated_players)} players")
        return pd.DataFrame(aggregated_players)
    
    def _aggregate_offensive_stats(self, player_data: pd.DataFrame, position: str) -> Dict:
        """Aggregate offensive statistics"""
        stats = {}
        
        # Passing stats (QB)
        if position == 'QB':
            stats.update({
                'total_passing_yards': player_data['passing_yards'].sum(),
                'total_passing_tds': player_data['passing_tds'].sum(),
                'total_interceptions': player_data['passing_interceptions'].sum(),
                'avg_completions': player_data['completions'].mean(),
                'avg_attempts': player_data['attempts'].mean(),
                'completion_pct': (player_data['completions'].sum() / player_data['attempts'].sum() * 100) if player_data['attempts'].sum() > 0 else 0,
                'avg_passing_epa': player_data['passing_epa'].mean(),
                'total_sacks': player_data['sacks_suffered'].sum(),
                'passing_yards_per_game': player_data['passing_yards'].mean(),
            })
        
        # Rushing stats (RB, QB)
        if position in ['RB', 'QB', 'WR']:
            stats.update({
                'total_rushing_yards': player_data['rushing_yards'].sum(),
                'total_rushing_tds': player_data['rushing_tds'].sum(),
                'total_carries': player_data['carries'].sum(),
                'avg_yards_per_carry': player_data['rushing_yards'].sum() / player_data['carries'].sum() if player_data['carries'].sum() > 0 else 0,
                'avg_rushing_epa': player_data['rushing_epa'].mean(),
                'rushing_yards_per_game': player_data['rushing_yards'].mean(),
            })
        
        # Receiving stats (WR, TE, RB)
        if position in ['WR', 'TE', 'RB']:
            stats.update({
                'total_receptions': player_data['receptions'].sum(),
                'total_targets': player_data['targets'].sum(),
                'total_receiving_yards': player_data['receiving_yards'].sum(),
                'total_receiving_tds': player_data['receiving_tds'].sum(),
                'catch_rate': (player_data['receptions'].sum() / player_data['targets'].sum() * 100) if player_data['targets'].sum() > 0 else 0,
                'avg_yards_per_reception': player_data['receiving_yards'].sum() / player_data['receptions'].sum() if player_data['receptions'].sum() > 0 else 0,
                'avg_receiving_epa': player_data['receiving_epa'].mean(),
                'avg_target_share': player_data['target_share'].mean(),
                'receiving_yards_per_game': player_data['receiving_yards'].mean(),
            })
        
        # Fantasy points
        stats.update({
            'total_fantasy_points': player_data['fantasy_points'].sum(),
            'total_fantasy_points_ppr': player_data['fantasy_points_ppr'].sum(),
            'avg_fantasy_points_per_game': player_data['fantasy_points'].mean(),
            'avg_fantasy_points_ppr_per_game': player_data['fantasy_points_ppr'].mean(),
        })
        
        return stats
    
    def _aggregate_defensive_stats(self, player_data: pd.DataFrame) -> Dict:
        """Aggregate defensive statistics"""
        return {
            'total_tackles': player_data['def_tackles_solo'].sum() + player_data['def_tackles_with_assist'].sum(),
            'total_solo_tackles': player_data['def_tackles_solo'].sum(),
            'total_sacks': player_data['def_sacks'].sum(),
            'total_interceptions': player_data['def_interceptions'].sum(),
            'total_forced_fumbles': player_data['def_fumbles_forced'].sum(),
            'total_tds': player_data['def_tds'].sum(),
            'avg_tackles_per_game': (player_data['def_tackles_solo'].sum() + player_data['def_tackles_with_assist'].sum()) / len(player_data),
        }
    
    def _aggregate_kicker_stats(self, player_data: pd.DataFrame) -> Dict:
        """Aggregate kicker statistics"""
        return {
            'fg_made': player_data['fg_made'].sum(),
            'fg_att': player_data['fg_att'].sum(),
            'fg_pct': (player_data['fg_made'].sum() / player_data['fg_att'].sum() * 100) if player_data['fg_att'].sum() > 0 else 0,
            'longest_fg': player_data['fg_long'].max(),
            'pat_made': player_data['pat_made'].sum(),
            'pat_att': player_data['pat_att'].sum(),
            'total_fantasy_points': player_data['fantasy_points'].sum(),
        }
    
    def _get_all_nextgen_stats(self, player_name: str, position: str) -> Dict:
        """Get all relevant NextGen stats for a player"""
        stats = {}
        
        # Rushing stats for RBs and QBs
        if position in ['RB', 'QB'] and self.nextgen_rush_df is not None:
            rush_stats = self._get_nextgen_rushing_stats(player_name)
            stats.update(rush_stats)
        
        # Passing stats for QBs
        if position == 'QB' and self.nextgen_pass_df is not None:
            pass_stats = self._get_nextgen_passing_stats(player_name)
            stats.update(pass_stats)
        
        # Receiving stats for WRs, TEs, RBs
        if position in ['WR', 'TE', 'RB'] and self.nextgen_rec_df is not None:
            rec_stats = self._get_nextgen_receiving_stats(player_name)
            stats.update(rec_stats)
        
        return stats
    
    def _get_nextgen_rushing_stats(self, player_name: str) -> Dict:
        """Get NextGen rushing stats for a player"""
        player_nextgen = self.nextgen_rush_df[self.nextgen_rush_df['player_display_name'] == player_name]
        
        if player_nextgen.empty:
            return {}
        
        player_nextgen = player_nextgen.iloc[0]  # Use season totals (week 0)
        
        return {
            'ng_rush_efficiency': player_nextgen['efficiency'],
            'ng_rush_yards_over_expected': player_nextgen['rush_yards_over_expected'],
            'ng_rush_yards_over_expected_per_att': player_nextgen['rush_yards_over_expected_per_att'],
            'ng_avg_time_to_los': player_nextgen['avg_time_to_los'],
            'ng_rush_pct_over_expected': player_nextgen['rush_pct_over_expected'],
        }
    
    def _get_nextgen_passing_stats(self, player_name: str) -> Dict:
        """Get NextGen passing stats for a player"""
        player_nextgen = self.nextgen_pass_df[self.nextgen_pass_df['player_display_name'] == player_name]
        
        if player_nextgen.empty:
            return {}
        
        player_nextgen = player_nextgen.iloc[0]  # Use season totals
        
        return {
            'ng_avg_time_to_throw': player_nextgen['avg_time_to_throw'],
            'ng_avg_completed_air_yards': player_nextgen['avg_completed_air_yards'],
            'ng_completion_pct_above_expectation': player_nextgen['completion_percentage_above_expectation'],
            'ng_aggressiveness': player_nextgen['aggressiveness'],
            'ng_max_completed_air_distance': player_nextgen['max_completed_air_distance'],
        }
    
    def _get_nextgen_receiving_stats(self, player_name: str) -> Dict:
        """Get NextGen receiving stats for a player"""
        player_nextgen = self.nextgen_rec_df[self.nextgen_rec_df['player_display_name'] == player_name]
        
        if player_nextgen.empty:
            return {}
        
        player_nextgen = player_nextgen.iloc[0]  # Use season totals
        
        return {
            'ng_avg_cushion': player_nextgen['avg_cushion'],
            'ng_avg_separation': player_nextgen['avg_separation'],
            'ng_avg_yac_above_expectation': player_nextgen['avg_yac_above_expectation'],
            'ng_catch_percentage': player_nextgen['catch_percentage'],
        }
    
    def _create_weekly_breakdown(self, player_data: pd.DataFrame) -> str:
        """Create a text summary of weekly performance"""
        weekly_summary = []
        
        for _, week_data in player_data.iterrows():
            week = week_data['week']
            opponent = week_data.get('opponent_team', 'N/A')
            fantasy_pts = week_data['fantasy_points_ppr']
            
            weekly_summary.append(f"Week {week} vs {opponent}: {fantasy_pts:.1f} pts")
        
        return " | ".join(weekly_summary[:10])  # Limit to first 10 weeks


class LangChainVectorDBBuilder:
    """Build ChromaDB vector database using LangChain with OpenAI embeddings"""
    
    def __init__(self, chroma_dir: Path):
        self.chroma_dir = chroma_dir
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.environ.get("API_KEY"),
            openai_api_base=OPENAI_BASE_URL
        )
        
    def create_player_embeddings(self, players_df: pd.DataFrame) -> None:
        """Create embeddings and store in ChromaDB using LangChain"""
        
        print("\n🔨 Building vector database with LangChain...")
        
        # Create documents for each player
        documents = []
        
        for idx, player in tqdm(players_df.iterrows(), total=len(players_df), desc="Creating documents"):
            # Create rich text description for embedding
            doc_text = self._create_player_document(player)
            
            # Create metadata (all values must be JSON-serializable)
            metadata = self._create_metadata(player)
            
            # Create LangChain Document
            doc = Document(
                page_content=doc_text,
                metadata=metadata
            )
            documents.append(doc)
        
        # Create or recreate the vector store
        if self.chroma_dir.exists():
            import shutil
            shutil.rmtree(self.chroma_dir)
        
        print("💾 Storing in ChromaDB with OpenAI embeddings...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.chroma_dir),
            collection_name="fantasy_players"
        )
        
        print(f"✅ Stored {len(documents)} player embeddings in ChromaDB")
    
    def _create_player_document(self, player: pd.Series) -> str:
        """Create a rich text document for embedding"""
        
        doc_parts = [
            f"Player: {player['player_name']}",
            f"Position: {player['position']}",
            f"Team: {player['team']}",
            f"Games Played: {player['games_played']}",
        ]
        
        # Add position-specific stats
        position = player['position']
        
        if position == 'QB':
            doc_parts.extend([
                f"Passing: {player.get('total_passing_yards', 0):.0f} yards, {player.get('total_passing_tds', 0):.0f} TDs, {player.get('total_interceptions', 0):.0f} INTs",
                f"Completion: {player.get('completion_pct', 0):.1f}%",
                f"EPA: {player.get('avg_passing_epa', 0):.2f}",
                f"Yards per game: {player.get('passing_yards_per_game', 0):.1f}",
            ])
            
            # Add NextGen passing stats
            if not pd.isna(player.get('ng_avg_time_to_throw')):
                doc_parts.append(f"Time to throw: {player['ng_avg_time_to_throw']:.2f}s, Completion % above expected: {player['ng_completion_pct_above_expectation']:.1f}%")
            
            if player.get('total_carries', 0) > 0:
                doc_parts.append(f"Rushing: {player.get('total_rushing_yards', 0):.0f} yards, {player.get('total_rushing_tds', 0):.0f} TDs")
        
        elif position == 'RB':
            doc_parts.extend([
                f"Rushing: {player.get('total_rushing_yards', 0):.0f} yards, {player.get('total_rushing_tds', 0):.0f} TDs on {player.get('total_carries', 0):.0f} carries",
                f"Average: {player.get('avg_yards_per_carry', 0):.2f} yards per carry",
                f"Receiving: {player.get('total_receptions', 0):.0f} catches on {player.get('total_targets', 0):.0f} targets for {player.get('total_receiving_yards', 0):.0f} yards",
            ])
            
            # Add NextGen rushing stats
            if not pd.isna(player.get('ng_rush_efficiency')):
                doc_parts.append(f"Efficiency: {player['ng_rush_efficiency']:.2f}, Rush yards over expected: {player['ng_rush_yards_over_expected']:.1f}, Time to LOS: {player['ng_avg_time_to_los']:.2f}s")
        
        elif position in ['WR', 'TE']:
            doc_parts.extend([
                f"Receiving: {player.get('total_receptions', 0):.0f} catches on {player.get('total_targets', 0):.0f} targets",
                f"Yards: {player.get('total_receiving_yards', 0):.0f}, TDs: {player.get('total_receiving_tds', 0):.0f}",
                f"Catch Rate: {player.get('catch_rate', 0):.1f}%",
                f"Yards per Reception: {player.get('avg_yards_per_reception', 0):.1f}",
                f"Target Share: {player.get('avg_target_share', 0):.3f}",
            ])
            
            # Add NextGen receiving stats
            if not pd.isna(player.get('ng_avg_separation')):
                doc_parts.append(f"Separation: {player['ng_avg_separation']:.2f} yards, YAC above expected: {player['ng_avg_yac_above_expectation']:.2f}")
        
        elif position == 'K':
            doc_parts.extend([
                f"Field Goals: {player.get('fg_made', 0):.0f}/{player.get('fg_att', 0):.0f} ({player.get('fg_pct', 0):.1f}%)",
                f"Longest: {player.get('longest_fg', 0):.0f} yards",
            ])
        
        elif position in ['DT', 'DE', 'LB', 'OLB', 'MLB', 'CB', 'S']:
            doc_parts.extend([
                f"Tackles: {player.get('total_tackles', 0):.0f} total, {player.get('total_solo_tackles', 0):.0f} solo",
                f"Sacks: {player.get('total_sacks', 0):.1f}",
                f"Interceptions: {player.get('total_interceptions', 0):.0f}",
            ])
        
        # Add fantasy points
        if not pd.isna(player.get('total_fantasy_points_ppr')):
            doc_parts.extend([
                f"Fantasy Points (PPR): {player.get('total_fantasy_points_ppr', 0):.1f} total, {player.get('avg_fantasy_points_ppr_per_game', 0):.1f} per game",
            ])
        
        return " | ".join(doc_parts)
    
    def _create_metadata(self, player: pd.Series) -> Dict:
        """Create metadata dictionary (must be JSON-serializable)"""
        
        metadata = {
            'player_id': str(player['player_id']),
            'player_name': str(player['player_name']),
            'position': str(player['position']),
            'team': str(player['team']),
            'games_played': int(player['games_played']),
        }
        
        # Add numeric stats (convert to float to ensure JSON serialization)
        numeric_fields = [
            'total_fantasy_points', 'total_fantasy_points_ppr', 
            'avg_fantasy_points_per_game', 'avg_fantasy_points_ppr_per_game',
            'total_passing_yards', 'total_passing_tds', 'total_rushing_yards',
            'total_rushing_tds', 'total_receptions', 'total_receiving_yards',
            'total_receiving_tds', 'total_tackles', 'total_sacks'
        ]
        
        for field in numeric_fields:
            if field in player and not pd.isna(player[field]):
                metadata[field] = float(player[field])
        
        return metadata


def main():
    """Main execution function"""
    
    print("🏈 Fantasy Football Vector Database Builder (LangChain + OpenAI)\n")
    
    # Check API key
    if not os.environ.get("API_KEY"):
        print("❌ Error: API_KEY environment variable not set!")
        print("   Please set it in your .env file")
        return
    
    # File paths
    stats_file = DATA_DIR / "player_stats_2025.csv"
    nextgen_rush_file = DATA_DIR / "nexgen_stats_2025_rush.csv"
    nextgen_pass_file = DATA_DIR / "nexgen_stats_2025_pass.csv"
    nextgen_rec_file = DATA_DIR / "nexgen_stats_2025_rec.csv"
    
    # Check files exist
    if not stats_file.exists():
        print(f"❌ Error: {stats_file} not found!")
        return
    
    # Step 1: Aggregate player data
    aggregator = PlayerDataAggregator(
        stats_file=str(stats_file),
        nextgen_rush_file=str(nextgen_rush_file) if nextgen_rush_file.exists() else None,
        nextgen_pass_file=str(nextgen_pass_file) if nextgen_pass_file.exists() else None,
        nextgen_rec_file=str(nextgen_rec_file) if nextgen_rec_file.exists() else None
    )
    
    players_df = aggregator.aggregate_player_stats()
    
    # Save aggregated data
    output_file = DATA_DIR / "aggregated_players.csv"
    players_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved aggregated data to {output_file}")
    
    # Step 2: Build vector database
    builder = LangChainVectorDBBuilder(chroma_dir=CHROMA_DIR)
    builder.create_player_embeddings(players_df)
    
    print(f"\n✅ Vector database built successfully!")
    print(f"📂 Database location: {CHROMA_DIR}")
    print(f"\n💡 Next steps:")
    print(f"   1. Run 'python query_vector_db.py' to test queries")
    print(f"   2. Update your Streamlit app to use the vector database")


if __name__ == "__main__":
    main()