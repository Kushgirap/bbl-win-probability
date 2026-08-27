"""
Cricsheet JSON -> one row per delivery.

Single source of truth: the notebook and anything else that needs raw match
states (tests included) call `extract_match_state` rather than keeping their own
copy. Two details here caused bugs before this was pulled out, worth keeping in
mind if this ever changes:

Match identity comes from the filename, not `match_date` - the BBL plays several
games on the same day, and deriving match_id from the date merges them.

Only legal deliveries increment the ball count. Wides and no-balls do not consume
a ball; counting them pushes innings past 120 and sends `balls_remaining` (and
everything derived from it) negative.
"""


def extract_match_state(match_data, innings_num, match_id):
    """One row per delivery, carrying the running state of the innings."""

    innings = match_data['innings'][innings_num]
    batting_team = innings['team']
    match_winner = match_data['info']['outcome']['winner']

    states = []
    cumulative_runs = 0
    cumulative_wickets = 0
    ball_count = 0

    for over in innings['overs']:
        for delivery in over['deliveries']:
            cumulative_runs += delivery['runs']['total']

            if 'wickets' in delivery:
                cumulative_wickets += len(delivery['wickets'])

            # Wides and no-balls don't consume a legal ball
            extras = delivery.get('extras', {})
            if 'wides' not in extras and 'noballs' not in extras:
                ball_count += 1

            states.append({
                'match_id':           match_id,
                'innings_number':     innings_num + 1,
                'match_date':         match_data['info']['dates'][0],
                'venue':              match_data['info']['venue'],
                'batting_team':       batting_team,
                'batter':             delivery['batter'],
                'bowler':             delivery['bowler'],
                'cumulative_runs':    cumulative_runs,
                'cumulative_wickets': cumulative_wickets,
                'balls_faced':        ball_count,
                'overs_completed':    over['over'],
                'did_team_win':       batting_team == match_winner,
            })

    return states
