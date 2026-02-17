"""
market scoring service -- composit demand pressure adn opportunity scoring
configurable weights for different market types (senior living, multifamily, etc)
"""

import pandas as pd
import numpy as np
from typing import Optional


# default weights for senior living markets
# these should be calibrated against historical deal outcomes in production
DEFAULT_WEIGHTS = {
    'senior_concentration': 0.25,
    'occupancy_pressure': 0.25,
    'supply_gap': 0.25,
    'quality_gap': 0.25,
}

# benchmarks for normalizing -- based on national avgs
BENCHMARKS = {
    'senior_pct_max': 20.0,       # 20% senior pop is high concentration
    'occupancy_target': 100.0,     # normalize against full occupancy
    'beds_per_1k_max': 30.0,       # 30 beds per 1k seniors is well supplied
}


class MarketScorer:
    """score markets for senior livign opportunity based on configurable metrics"""

    def __init__(self, weights: dict = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def demand_pressure_score(
        self,
        senior_pct: float,
        avg_occupancy: float,
        beds_per_1k_seniors: float,
        low_quality_pct: float,
    ) -> dict:
        """
        compute composite demand pressure score 0-100
        higher = more unmet demand = bigger opportunity
        """
        # each componet scored 0-1 then weighted
        components = {
            'senior_concentration': senior_pct / BENCHMARKS['senior_pct_max'],
            'occupancy_pressure': avg_occupancy / BENCHMARKS['occupancy_target'],
            'supply_gap': 1 - (beds_per_1k_seniors / BENCHMARKS['beds_per_1k_max']),
            'quality_gap': low_quality_pct,
        }

        # clamp to 0-1
        for k in components:
            components[k] = max(0, min(1, components[k]))

        # weighted sum scaled to 100
        raw_score = sum(
            components[k] * self.weights[k]
            for k in components
        )
        final_score = round(raw_score * 100, 1)

        return {
            'score': final_score,
            'components': {k: round(v, 3) for k, v in components.items()},
            'weights': self.weights,
            'interpretation': self._interpret_score(final_score),
        }

    def _interpret_score(self, score: float) -> str:
        """Translate score into actionable language"""
        if score >= 75:
            return 'strong opportunity -- high unmet demand'
        elif score >= 60:
            return 'moderate-strong -- worth pursuing with focused positioning'
        elif score >= 45:
            return 'moderate -- needs submarket validation before committing'
        elif score >= 30:
            return 'weak-moderate -- limited opportunity without clear differentiator'
        else:
            return 'weak -- market appears well served or low demand'

    def competitive_position_score(self, facility_df: pd.DataFrame) -> dict:
        """
        assess competitive landscape strength
        looks at quality distribution, ownership mix, adn capacity utilzation
        """
        if len(facility_df) == 0:
            return {'score': 0, 'interpretation': 'no data'}

        # quality distribution -- more low rated = more displacement opportunity
        if 'overall_rating' in facility_df.columns:
            avg_rating = facility_df['overall_rating'].mean()
            low_rated_pct = len(facility_df[facility_df['overall_rating'] <= 2]) / len(facility_df)
            high_rated_pct = len(facility_df[facility_df['overall_rating'] >= 4]) / len(facility_df)
        else:
            avg_rating = 3.0
            low_rated_pct = 0
            high_rated_pct = 0.5

        # ownership -- high nonprofit share means entrenched incumbants
        if 'ownership' in facility_df.columns:
            nonprofit_pct = len(facility_df[facility_df['ownership'] == 'Non profit']) / len(facility_df)
        else:
            nonprofit_pct = 0.5

        # occupancy spread -- wide spread means market segmentation exists
        if 'occupancy_pct' in facility_df.columns:
            occ_std = facility_df['occupancy_pct'].std()
            occ_spread_score = min(occ_std / 10, 1)  # normalize, high spread = fragmented
        else:
            occ_spread_score = 0.5

        # displacement opporunity score
        displacement = round(
            low_rated_pct * 40 +           # more low rated = more to displace
            (1 - high_rated_pct) * 20 +    # fewer high rated = less competition at top
            (1 - nonprofit_pct) * 20 +     # less nonprofit = weaker community ties
            occ_spread_score * 20,          # more spread = niche gaps exist
            1
        )

        return {
            'score': displacement,
            'avg_rating': round(avg_rating, 1),
            'low_rated_pct': round(low_rated_pct * 100, 1),
            'high_rated_pct': round(high_rated_pct * 100, 1),
            'nonprofit_pct': round(nonprofit_pct * 100, 1),
            'occupancy_spread': round(occ_spread_score, 3),
            'interpretation': 'high displacement opportunity' if displacement > 50 else 'moderate' if displacement > 30 else 'limited displacement opportunity',
        }

    def full_market_assessment(
        self,
        census_metrics: dict,
        facility_df: pd.DataFrame,
    ) -> dict:
        """
        full market assessment combining demand adn competitive scores
        returns everything needed for the briefing
        """
        # extract what we need from census
        senior_pct = census_metrics.get('senior_pct', 12)
        beds_per_1k = census_metrics.get('beds_per_1k_seniors', 15)

        # occupancy from facilities
        avg_occ = facility_df['occupancy_pct'].mean() if 'occupancy_pct' in facility_df.columns else 80
        low_quality_pct = len(facility_df[facility_df['overall_rating'] <= 2]) / max(len(facility_df), 1) if 'overall_rating' in facility_df.columns else 0

        demand = self.demand_pressure_score(senior_pct, avg_occ, beds_per_1k, low_quality_pct)
        competitive = self.competitive_position_score(facility_df)

        # overall opportunity is weighted combo
        overall = round(demand['score'] * 0.6 + competitive['score'] * 0.4, 1)

        return {
            'overall_opportunity_score': overall,
            'demand_pressure': demand,
            'competitive_position': competitive,
            'recommendation': self._interpret_score(overall),
        }
