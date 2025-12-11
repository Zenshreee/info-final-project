import nflreadpy as nfl

# player stats
# player_stats = nfl.load_player_stats([2025])
# player_stats.write_csv('player_stats_2025.csv')

# next gen stats
nexgen_stats = nfl.load_nextgen_stats([2025], stat_type='rushing')
nexgen_stats.write_csv('nexgen_stats_2025_rush.csv')
