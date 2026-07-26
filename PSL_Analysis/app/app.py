"""
PSL Data Analytics Dashboard — Streamlit App (2016–2024)
=========================================================
Run:  streamlit run app/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import re
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSL Analytics 2016–2024",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0D1117; color: #C9D1D9; }
  .metric-card {
      background: #161B22; border: 1px solid #30363D;
      border-radius: 10px; padding: 20px; text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #58A6FF; }
  .metric-label { font-size: 0.85rem; color: #8B949E; margin-top: 4px; }
  h1, h2, h3 { color: #E6EDF3 !important; }
  .stSelectbox label, .stMultiSelect label { color: #C9D1D9 !important; }
  div[data-testid="stSidebar"] { background-color: #161B22; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DARK_BG = '#0D1117'
CARD_BG = '#161B22'
FONT_CLR = '#C9D1D9'

TEAM_COLORS = {
    'Islamabad United' : '#FACC15',
    'Karachi Kings'    : '#3B82F6',
    'Lahore Qalandars' : '#D85A30',
    'Multan Sultans'   : '#993556',
    'Peshawar Zalmi'   : '#10B981',
    'Quetta Gladiators': '#7C3AED',
}

LAYOUT = dict(
    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
    font=dict(color=FONT_CLR, size=12),
    title_font=dict(size=15, color='white'),
    xaxis=dict(gridcolor='#21262D', zerolinecolor='#21262D'),
    yaxis=dict(gridcolor='#21262D', zerolinecolor='#21262D'),
    margin=dict(l=50, r=30, t=55, b=45),
)

# ── Data root: supports running from any working directory ────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'preprocessed') + os.sep

# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all peprocessed / feature-enriched datasets."""

    def safe_csv(fname, fallback=None):
        path = PROC + fname
        if os.path.exists(path):
            return pd.read_csv(path)
        # try clean version
        if fallback and os.path.exists(PROC + fallback):
            return pd.read_csv(PROC + fallback)
        return pd.DataFrame()

    d = {}
    d['runs']        = safe_csv('most_runs_feat.csv', 'most_runs_clean.csv')
    d['wickets']     = safe_csv('most_wickets_feat.csv', 'most_wickets_clean.csv')
    d['sixes']       = safe_csv('most_sixes_history_clean.csv')
    d['catches']     = safe_csv('most_catches_clean.csv')
    d['dismissals']  = safe_csv('most_dismissals_clean.csv')
    d['hs']          = safe_csv('highest_score_clean.csv')
    d['sixes_inn']   = safe_csv('most_sixes_innings_clean.csv')
    d['best_bowl']   = safe_csv('best_bowling_clean.csv')
    d['high_tot']    = safe_csv('highest_totals_feat.csv', 'highest_totals_clean.csv')
    d['low_tot']     = safe_csv('lowest_totals_feat.csv', 'lowest_totals_clean.csv')
    d['team_res']    = safe_csv('result_summary_feat.csv', 'result_summary_clean.csv')
    d['timeline']    = safe_csv('match_wins_feat.csv', 'match_wins_clean.csv')
    return d

data = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏏 PSL Analytics")
st.sidebar.markdown("**Pakistan Super League**  \n2016 – 2024")
st.sidebar.divider()

PAGE = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🏆 Team Analysis", "🏏 Player Analysis",
     "🎯 Match Analysis", "📈 Season Timeline"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Global Filters")

ALL_TEAMS = list(TEAM_COLORS.keys())
sel_teams = st.sidebar.multiselect("Select Teams", ALL_TEAMS, default=ALL_TEAMS)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if PAGE == "🏠 Home":
    st.markdown("<h1 style='text-align:center;'>🏏 Pakistan Super League Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8B949E; font-size:1.1rem;'>End-to-end data analysis of PSL performance (2016–2024)</p>", unsafe_allow_html=True)
    st.divider()

    # ── KPI Cards ──────────────────────────────────────────────────────────
    runs_df  = data['runs']
    bowl_df  = data['wickets']
    team_df  = data['team_res']
    tl_df    = data['timeline']

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Players (Batting)", f"{len(runs_df):,}", "Players in records"),
        (c2, "Most Runs", f"{int(runs_df['runs'].max()):,}" if not runs_df.empty else "–",
              runs_df.nlargest(1,'runs')['player_name'].values[0] if not runs_df.empty else ""),
        (c3, "Most Wickets", f"{int(bowl_df['wickets'].max())}" if not bowl_df.empty else "–",
              bowl_df.nlargest(1,'wickets')['player_name'].values[0] if not bowl_df.empty else ""),
        (c4, "Best Win %", f"{team_df['win_pct'].max():.1f}%" if not team_df.empty else "–",
              team_df.nlargest(1,'win_pct')['team'].values[0] if not team_df.empty else ""),
        (c5, "Total Matches", f"{int(tl_df['match_number'].max())}" if not tl_df.empty else "–",
              "PSL matches analysed"),
    ]
    for col, label, val, sub in kpis:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
          <div style="color:#58A6FF; font-size:0.8rem; margin-top:4px;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Team Win % overview ────────────────────────────────────────────────
    st.subheader("Team Win Percentage Overview")
    if not team_df.empty:
        filt = team_df[team_df['team'].isin(sel_teams)].sort_values('win_pct', ascending=True)
        fig = go.Figure()
        for _, row in filt.iterrows():
            fig.add_trace(go.Bar(
                x=[row['win_pct']], y=[row['team']], orientation='h',
                marker_color=TEAM_COLORS.get(row['team'], '#888'),
                text=f"{row['win_pct']:.1f}%", textposition='outside',
                name=row['team'],
            ))
        fig.update_layout(**LAYOUT, showlegend=False,
                          xaxis_title="Win %", xaxis_range=[0, 70], height=320)
        st.plotly_chart(fig, use_container_width=True)

    # ── Project info ───────────────────────────────────────────────────────
    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📂 Datasets Used")
        ds_info = [
            ("Batting","Most Runs, Highest Score, Most Sixes (History), Most Sixes (Innings)"),
            ("Bowling","Most Wickets, Best Bowling Figures"),
            ("Fielding","Most Catches, WK Dismissals"),
            ("Team","Highest Totals, Lowest Totals, Result Summary"),
            ("Timeline","Cumulative Match Wins"),
        ]
        for cat, desc in ds_info:
            st.markdown(f"**{cat}**: {desc}")
    with col_r:
        st.subheader("🎯 Project Objective")
        st.markdown("""
        Analyse PSL performance from 2016–2024 to identify:
        - What factors make teams win (venue, toss, batting order)
        - Player performance patterns (batting, bowling, fielding)
        - What Karachi Kings need to improve to compete at the top
        - Era trends — is PSL scoring getting higher over time?
        """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TEAM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif PAGE == "🏆 Team Analysis":
    st.title("🏆 Team Analysis")

    team_df = data['team_res']
    high_df = data['high_tot']
    low_df  = data['low_tot']

    if team_df.empty:
        st.warning("Team data not found. Run notebooks 01-03 first.")
    else:
        filt_teams = team_df[team_df['team'].isin(sel_teams)]

        tab1, tab2, tab3 = st.tabs(["📊 Win Statistics", "🏟️ Ground Performance", "📉 Low Scores"])

        with tab1:
            st.subheader("Win / Loss / Draw Records")
            c1, c2 = st.columns(2)

            with c1:
                fig = px.bar(
                    filt_teams.sort_values('won', ascending=True),
                    x='won', y='team', orientation='h', color='team',
                    color_discrete_map=TEAM_COLORS,
                    title='Total Wins by Team', labels={'won': 'Wins', 'team': ''},
                    text='won',
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(**LAYOUT, showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.bar(
                    filt_teams.sort_values('win_pct', ascending=True),
                    x='win_pct', y='team', orientation='h', color='team',
                    color_discrete_map=TEAM_COLORS,
                    title='Win Percentage by Team', labels={'win_pct': 'Win %', 'team': ''},
                    text=filt_teams.sort_values('win_pct')['win_pct'].apply(lambda x: f"{x:.1f}%"),
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(**LAYOUT, showlegend=False, height=350, xaxis_range=[0,70])
                st.plotly_chart(fig, use_container_width=True)

            # Match outcomes stacked bar
            st.subheader("Match Outcome Composition")
            outcome_df = filt_teams[['team', 'won', 'lost', 'tied', 'no_result']].melt(
                id_vars='team', var_name='Outcome', value_name='Count')
            fig = px.bar(outcome_df, x='team', y='Count', color='Outcome',
                         barmode='stack',
                         color_discrete_map={'won': '#3FB950', 'lost': '#FF6E6E',
                                             'tied': '#FACC15', 'no_result': '#8B949E'},
                         title='Match Outcome Breakdown by Team')
            fig.update_layout(**LAYOUT, height=380)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Full Team Stats Table")
            show_cols = ['team', 'matches', 'won', 'lost', 'tied', 'no_result',
                         'win_pct', 'loss_pct', 'win_loss_ratio']
            show_cols = [c for c in show_cols if c in filt_teams.columns]
            st.dataframe(filt_teams[show_cols].sort_values('win_pct', ascending=False)
                         .reset_index(drop=True), use_container_width=True)

        with tab2:
            if not high_df.empty:
                st.subheader("Highest Totals by Team and Ground")
                sel_team_ht = st.selectbox("Select Team", sel_teams, key='ht_team')
                team_ht = high_df[high_df['team'] == sel_team_ht] if not high_df.empty else pd.DataFrame()

                if not team_ht.empty and 'ground' in team_ht.columns:
                    g_avg = team_ht.groupby('ground')['runs'].agg(['mean','max','count']).reset_index()
                    g_avg.columns = ['ground','avg_runs','max_runs','matches']
                    fig = px.bar(g_avg.sort_values('avg_runs', ascending=True),
                                 x='avg_runs', y='ground', orientation='h',
                                 color='max_runs', color_continuous_scale='Blues',
                                 hover_data={'max_runs': True, 'matches': True},
                                 title=f'{sel_team_ht} — Avg Highest Score by Ground')
                    fig.update_layout(**LAYOUT, coloraxis_showscale=True, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Cross-team ground heat
                st.subheader("All Teams — Average Score by Ground")
                if 'ground' in high_df.columns:
                    pivot = high_df[high_df['team'].isin(sel_teams)].groupby(
                        ['team','ground'])['runs'].mean().unstack(fill_value=0)
                    fig = px.imshow(pivot, color_continuous_scale='Blues', aspect='auto',
                                    title='Heatmap: Avg Score by Team × Ground')
                    fig.update_layout(**LAYOUT, height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Highest totals data not available.")

        with tab3:
            if not low_df.empty:
                st.subheader("Lowest Totals — Collapse Analysis")
                low_filt = low_df[low_df['team'].isin(sel_teams)] if 'team' in low_df.columns else low_df
                fig = px.scatter(low_filt, x='match_year' if 'match_year' in low_filt.columns else low_filt.index,
                                 y='runs', color='team',
                                 color_discrete_map=TEAM_COLORS,
                                 hover_data={c: True for c in ['ground','opposition','wickets'] if c in low_filt.columns},
                                 title='Lowest Totals by Team (All Seasons)',
                                 labels={'runs': 'Runs Scored', 'match_year': 'Year'})
                fig.update_layout(**LAYOUT, height=400)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Teams with Most Low-Score Appearances")
                low_counts = low_df['team'].value_counts().reset_index()
                low_counts.columns = ['team', 'appearances']
                low_counts = low_counts[low_counts['team'].isin(sel_teams)]
                fig = px.bar(low_counts, x='team', y='appearances', color='team',
                             color_discrete_map=TEAM_COLORS,
                             title='Appearances in Lowest Totals Dataset')
                fig.update_layout(**LAYOUT, showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Lowest totals data not available.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PLAYER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif PAGE == "🏏 Player Analysis":
    st.title("🏏 Player Analysis")

    runs_df = data['runs']
    bowl_df = data['wickets']
    six_df  = data['sixes']

    tab1, tab2, tab3, tab4 = st.tabs(["🏏 Top Batters", "🎳 Top Bowlers", "⭐ All-Rounders", "🔎 Player Lookup"])

    with tab1:
        st.subheader("Top Run Scorers")
        n_bat = st.slider("Show top N batters", 10, 50, 20, key='n_bat')
        min_inn = st.slider("Minimum innings (for avg/SR charts)", 5, 40, 15, key='min_inn')

        top_bat = runs_df.nlargest(n_bat, 'runs')
        fig = px.bar(top_bat[::-1], x='runs', y='player_name', orientation='h',
                     color='runs', color_continuous_scale='Blues',
                     text='runs', title=f'Top {n_bat} PSL Run Scorers',
                     labels={'runs': 'Career Runs', 'player_name': ''})
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(**LAYOUT, coloraxis_showscale=False, height=max(400, n_bat * 22))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Strike Rate vs. Average (Bubble = Career Runs)")
        bat_filt = runs_df[runs_df['innings'] >= min_inn]
        if 'bat_efficiency_score' not in bat_filt.columns:
            bat_filt = bat_filt.copy()
            bat_filt['bat_efficiency_score'] = 50
        fig = px.scatter(bat_filt, x='average', y='strike_rate', size='runs',
                         color='bat_efficiency_score', hover_name='player_name',
                         color_continuous_scale='Viridis', size_max=30,
                         title=f'SR vs Average — min {min_inn} innings',
                         labels={'average': 'Batting Average', 'strike_rate': 'Strike Rate'})
        fig.update_layout(**LAYOUT, height=460)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Six Hitters")
        if not six_df.empty and 'sixes' in six_df.columns:
            top_six = six_df.nlargest(15, 'sixes')
            fig = px.bar(top_six[::-1], x='sixes', y='player_name', orientation='h',
                         color='sixes', color_continuous_scale='Oranges',
                         text='sixes', title='Top 15 Six-Hitters in PSL History')
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(**LAYOUT, coloraxis_showscale=False, height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Top Wicket Takers")
        n_bowl = st.slider("Show top N bowlers", 10, 50, 20, key='n_bowl')
        min_wkts = st.slider("Min wickets for scatter", 10, 40, 20, key='min_wkts')

        top_bowl = bowl_df.nlargest(n_bowl, 'wickets')
        fig = px.bar(top_bowl[::-1], x='wickets', y='player_name', orientation='h',
                     color='wickets', color_continuous_scale='Greens',
                     text='wickets', title=f'Top {n_bowl} PSL Wicket Takers',
                     labels={'wickets': 'Career Wickets', 'player_name': ''})
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(**LAYOUT, coloraxis_showscale=False, height=max(400, n_bowl * 22))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Economy vs. Wickets")
        bowl_filt = bowl_df[bowl_df['wickets'] >= min_wkts]
        if 'bowl_impact_score' not in bowl_filt.columns:
            bowl_filt = bowl_filt.copy()
            bowl_filt['bowl_impact_score'] = 50
        fig = px.scatter(bowl_filt, x='economy', y='wickets', size='matches',
                         color='bowl_impact_score', hover_name='player_name',
                         color_continuous_scale='RdYlGn',
                         title=f'Economy vs. Wickets (min {min_wkts} wickets)')
        fig.update_layout(**LAYOUT, height=440)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Bowling Average Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=bowl_df['bowling_avg'], nbinsx=20,
                                   marker_color='#3FB950', opacity=0.85))
        fig.update_layout(**LAYOUT, title='Distribution of Bowling Averages',
                           xaxis_title='Bowling Average', height=320)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("PSL All-Rounders (300+ runs & 20+ wickets)")
        if not runs_df.empty and not bowl_df.empty:
            bat_names  = set(runs_df[runs_df['runs'] >= 300]['player_name'])
            bowl_names = set(bowl_df[bowl_df['wickets'] >= 20]['player_name'])
            ar_names   = bat_names & bowl_names

            ar_rows = []
            for name in ar_names:
                b = runs_df[runs_df['player_name'] == name].iloc[0]
                w = bowl_df[bowl_df['player_name'] == name].iloc[0]
                ar_rows.append({
                    'Player'  : name,
                    'Runs'    : int(b['runs']),
                    'Bat Avg' : round(b['average'], 1),
                    'SR'      : round(b['strike_rate'], 1),
                    'Wickets' : int(w['wickets']),
                    'Economy' : round(w['economy'], 2),
                    'Bowl Avg': round(w['bowling_avg'], 1),
                })
            ar_df_show = pd.DataFrame(ar_rows).sort_values('Wickets', ascending=False)
            st.dataframe(ar_df_show, use_container_width=True)

            if not ar_df_show.empty:
                fig = px.scatter(ar_df_show, x='Runs', y='Wickets',
                                 size='SR', color='Economy',
                                 hover_name='Player',
                                 color_continuous_scale='RdYlGn_r',
                                 title='All-Rounders: Runs vs. Wickets (size=SR, color=Economy)',
                                 size_max=25)
                fig.update_layout(**LAYOUT, height=420)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🔎 Individual Player Lookup")
        all_bat_names = sorted(runs_df['player_name'].dropna().unique().tolist()) if not runs_df.empty else []
        sel_player = st.selectbox("Search player", all_bat_names)
        if sel_player:
            p_bat = runs_df[runs_df['player_name'] == sel_player]
            p_bowl = bowl_df[bowl_df['player_name'] == sel_player]

            c1, c2 = st.columns(2)
            if not p_bat.empty:
                r = p_bat.iloc[0]
                with c1:
                    st.markdown(f"### 🏏 Batting — {sel_player}")
                    metrics = {
                        'Career Runs': f"{int(r['runs']):,}",
                        'Matches': f"{int(r['matches'])}",
                        'Innings': f"{int(r['innings'])}",
                        'Batting Average': f"{r['average']:.2f}",
                        'Strike Rate': f"{r['strike_rate']:.2f}",
                        'High Score': f"{int(r['high_score'])}",
                        'Centuries': str(r.get('centuries', 0)),
                        'Fifties': str(r.get('fifties', 0)),
                        'Fours': str(r.get('fours', 0)),
                        'Teams': str(r.get('teams_played', '–')),
                    }
                    for k, v in metrics.items():
                        st.markdown(f"**{k}:** {v}")
            if not p_bowl.empty:
                r = p_bowl.iloc[0]
                with c2:
                    st.markdown(f"### 🎳 Bowling — {sel_player}")
                    metrics = {
                        'Wickets': f"{int(r['wickets'])}",
                        'Bowling Average': f"{r['bowling_avg']:.2f}",
                        'Economy': f"{r['economy']:.2f}",
                        'Bowling SR': f"{r['bowling_sr']:.2f}",
                        'Best Figures': f"{r.get('bbi_wkts', '–')}/{r.get('bbi_runs', '–')}",
                        '4-Wkt Hauls': str(r.get('four_wicket_hauls', 0)),
                        '5-Wkt Hauls': str(r.get('five_wicket_hauls', 0)),
                    }
                    for k, v in metrics.items():
                        st.markdown(f"**{k}:** {v}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MATCH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif PAGE == "🎯 Match Analysis":
    st.title("🎯 Match Analysis")

    high_df  = data['high_tot']
    low_df   = data['low_tot']
    hs_df    = data['hs']
    bowl_df2 = data['best_bowl']

    tab1, tab2, tab3 = st.tabs(["📊 Venue Analysis", "💥 Record Performances", "🔄 Innings Type"])

    with tab1:
        st.subheader("Venue Performance Overview")
        if not high_df.empty and 'ground' in high_df.columns:
            venue_stats = (high_df[high_df['team'].isin(sel_teams)]
                           .groupby('ground')['runs']
                           .agg(avg_score='mean', max_score='max', matches='count')
                           .reset_index()
                           .sort_values('avg_score', ascending=False)
                           .head(20))

            fig = px.bar(venue_stats, x='avg_score', y='ground', orientation='h',
                         color='avg_score', color_continuous_scale='Blues',
                         hover_data={'max_score': True, 'matches': True},
                         title='Average High Total by Venue (Top 20)',
                         labels={'avg_score': 'Avg Runs', 'ground': 'Ground'})
            fig.update_layout(**LAYOUT, coloraxis_showscale=False, height=550)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Wins vs. Losses at Each Ground (Batting Teams)")
            if 'won' in high_df.columns:
                venue_wl = high_df[high_df['team'].isin(sel_teams)].groupby(['ground','won']).size().reset_index()
                venue_wl.columns = ['ground','won','count']
                venue_wl['Result'] = venue_wl['won'].map({True:'Won', False:'Lost'})
                top_grounds = high_df['ground'].value_counts().head(12).index.tolist()
                venue_wl_filt = venue_wl[venue_wl['ground'].isin(top_grounds)]
                fig = px.bar(venue_wl_filt, x='ground', y='count', color='Result',
                             barmode='group',
                             color_discrete_map={'Won': '#3FB950', 'Lost': '#FF6E6E'},
                             title='Win/Loss When Posting High Total — Top 12 Grounds')
                fig.update_layout(**LAYOUT, height=420,
                                  xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Highest Individual Scores")
        if not hs_df.empty:
            fig = px.scatter(hs_df, x='balls' if 'balls' in hs_df.columns else hs_df.index,
                             y='runs', color='team' if 'team' in hs_df.columns else None,
                             color_discrete_map=TEAM_COLORS,
                             hover_name='player' if 'player' in hs_df.columns else None,
                             hover_data={c: True for c in ['ground','opposition','sr'] if c in hs_df.columns},
                             title='Highest Individual Scores — Balls vs Runs',
                             labels={'balls': 'Balls Faced', 'runs': 'Runs Scored'})
            fig.update_layout(**LAYOUT, height=420)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top 20 Highest Individual Scores")
            show_hs = [c for c in ['player','runs','balls','sr','team','opposition','ground','match_date']
                       if c in hs_df.columns]
            st.dataframe(hs_df.nlargest(20, 'runs')[show_hs].reset_index(drop=True),
                         use_container_width=True)

        st.subheader("Best Bowling Figures")
        if not bowl_df2.empty:
            show_bowl = [c for c in ['player','wkts','runs','overs','econ','team','opposition','ground']
                         if c in bowl_df2.columns]
            st.dataframe(bowl_df2.sort_values('wkts').head(20)[show_bowl].reset_index(drop=True),
                         use_container_width=True)

    with tab3:
        st.subheader("Batting First vs. Chasing — Win Analysis")
        if not high_df.empty and 'innings_type' in high_df.columns:
            inn_wr = (high_df[high_df['team'].isin(sel_teams)]
                      .groupby('innings_type')['won'].agg(['sum','count'])
                      .reset_index())
            inn_wr.columns = ['Innings Type','Wins','Total']
            inn_wr['Win Rate %'] = (inn_wr['Wins'] / inn_wr['Total'] * 100).round(1)

            fig = px.bar(inn_wr, x='Innings Type', y='Win Rate %',
                         color='Innings Type',
                         color_discrete_sequence=['#58A6FF', '#F0883E'],
                         title='Win Rate: Batting First vs. Chasing (High-Score Matches)',
                         text='Win Rate %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(**LAYOUT, showlegend=False, yaxis_range=[0, 110], height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Notebook 03 (Feature Engineering) to enable innings-type analysis.")

        if not high_df.empty and 'era' in high_df.columns:
            st.subheader("PSL Era Trends")
            era_stats = high_df.groupby('era')['runs'].agg(['mean','max','count']).reset_index()
            era_stats.columns = ['era','avg_runs','max_runs','matches']
            fig = px.bar(era_stats, x='era', y='avg_runs',
                         color='avg_runs', color_continuous_scale='Blues',
                         title='Average High Score by PSL Era',
                         text='avg_runs')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(**LAYOUT, coloraxis_showscale=False, height=360)
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SEASON TIMELINE
# ─────────────────────────────────────────────────────────────────────────────
elif PAGE == "📈 Season Timeline":
    st.title("📈 Season Timeline")

    tl_df   = data['timeline']
    team_df = data['team_res']

    if tl_df.empty:
        st.warning("Timeline data not available.")
    else:
        TEAM_COLS = [c for c in tl_df.columns if c not in ('match_number', 'leader', 'lead_gap')
                     and 'match_wins' not in c]
        filt_tcols = [c for c in TEAM_COLS if c in sel_teams]

        if not filt_tcols:
            filt_tcols = TEAM_COLS

        st.subheader("Cumulative Win Race — All Seasons")
        fig = go.Figure()
        for team in filt_tcols:
            if team in tl_df.columns:
                fig.add_trace(go.Scatter(
                    x=tl_df['match_number'], y=tl_df[team],
                    mode='lines', name=team,
                    line=dict(color=TEAM_COLORS.get(team, '#888'), width=2.5),
                ))
        fig.update_layout(
            **LAYOUT, height=480,
            title='Cumulative PSL Wins Over All 278 Matches',
            xaxis_title='Match Number', yaxis_title='Total Wins',
            legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='#444'),
        )
        st.plotly_chart(fig, use_container_width=True)

        if 'leader' in tl_df.columns:
            st.subheader("Leadership Transitions")
            leader_counts = tl_df['leader'].value_counts().reset_index()
            leader_counts.columns = ['team', 'matches_as_leader']
            fig = px.pie(leader_counts, names='team', values='matches_as_leader',
                         color='team', color_discrete_map=TEAM_COLORS,
                         title='Matches Spent in Cumulative Lead (All-Time)',
                         hole=0.35)
            fig.update_layout(**LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)

        if 'lead_gap' in tl_df.columns:
            st.subheader("Lead Gap Over Time (1st vs 2nd)")
            fig = px.area(tl_df, x='match_number', y='lead_gap',
                          title='Gap Between 1st and 2nd Place Team (Cumulative Wins)',
                          labels={'match_number': 'Match Number', 'lead_gap': 'Win Gap'},
                          color_discrete_sequence=['#58A6FF'])
            fig.update_layout(**LAYOUT, height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Final standings table
        st.subheader("Final Standings (End of 278 Matches)")
        last_row = tl_df.iloc[-1]
        standings = pd.DataFrame([
            {'Rank': i+1, 'Team': team, 'Cumulative Wins': int(last_row[team])}
            for i, team in enumerate(sorted(filt_tcols, key=lambda t: last_row.get(t, 0), reverse=True))
            if team in tl_df.columns
        ])
        st.dataframe(standings, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("""
<div style="text-align:center; color:#8B949E; font-size:0.75rem;">
PSL Analytics Portfolio<br>Data: 2016 – 2024<br>Built with Streamlit + Plotly
</div>""", unsafe_allow_html=True)
