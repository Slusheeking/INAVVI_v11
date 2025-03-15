"""
Timeframe Selector

This module provides the TimeframeSelector class for selecting the appropriate
timeframes for analysis and trading based on market conditions and strategy requirements.
"""

from src.utils.logging import get_logger

logger = get_logger("trading_strategy.selection.timeframe_selector")


class TimeframeSelector:
    """
    Selects appropriate timeframes for analysis and trading.

    The timeframe selection process considers:
    1. Strategy type (trend following, mean reversion, etc.)
    2. Volatility regime
    3. Trading frequency
    4. Holding period
    5. Market microstructure

    This approach ensures that the system analyzes data at the most relevant timeframes
    for the current market conditions and strategy objectives.
    """

    def __init__(
        self,
        base_timeframes: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"],
        primary_timeframe: str = "1h",
        max_timeframes: int = 3,
        volatility_adaptive: bool = True,
        include_higher_timeframes: bool = True,
        include_lower_timeframes: bool = True,
    ):
        """
        Initialize the TimeframeSelector.

        Args:
            base_timeframes: List of available timeframes
            primary_timeframe: Primary timeframe for analysis
            max_timeframes: Maximum number of timeframes to select
            volatility_adaptive: Whether to adapt timeframes based on volatility
            include_higher_timeframes: Whether to include higher timeframes
            include_lower_timeframes: Whether to include lower timeframes
        """
        self.base_timeframes = base_timeframes
        self.primary_timeframe = primary_timeframe
        self.max_timeframes = max_timeframes
        self.volatility_adaptive = volatility_adaptive
        self.include_higher_timeframes = include_higher_timeframes
        self.include_lower_timeframes = include_lower_timeframes

        # Timeframe metadata
        self.timeframe_minutes = self._calculate_timeframe_minutes()

        # Current selections
        self.current_timeframes: dict[str, list[str]] = {}

        logger.info(
            f"Initialized TimeframeSelector with primary_timeframe={primary_timeframe}, "
            f"max_timeframes={max_timeframes}"
        )

    def _calculate_timeframe_minutes(self) -> dict[str, int]:
        """
        Calculate the number of minutes for each timeframe.

        Returns:
            Dictionary mapping timeframes to minutes
        """
        minutes = {}

        for tf in self.base_timeframes:
            # Parse timeframe
            if tf.endswith("m"):
                minutes[tf] = int(tf[:-1])
            elif tf.endswith("h"):
                minutes[tf] = int(tf[:-1]) * 60
            elif tf.endswith("d"):
                minutes[tf] = int(tf[:-1]) * 60 * 24
            elif tf.endswith("w"):
                minutes[tf] = int(tf[:-1]) * 60 * 24 * 7
            else:
                logger.warning(f"Unknown timeframe format: {tf}")
                minutes[tf] = 0

        return minutes

    def select_timeframes(
        self,
        strategy_type: str,
        volatility: float | None = None,
        holding_period_minutes: int | None = None,
        trading_frequency: str = "medium",
    ) -> list[str]:
        """
        Select appropriate timeframes based on strategy and market conditions.

        Args:
            strategy_type: Type of strategy ('trend_following', 'mean_reversion', etc.)
            volatility: Volatility level (ATR percentage)
            holding_period_minutes: Expected holding period in minutes
            trading_frequency: Trading frequency ('high', 'medium', 'low')

        Returns:
            List of selected timeframes
        """
        # Start with primary timeframe
        selected_timeframes = [self.primary_timeframe]

        # Get primary timeframe minutes
        primary_minutes = self.timeframe_minutes.get(self.primary_timeframe, 60)

        # Adjust based on strategy type
        if strategy_type == "trend_following":
            # Trend following strategies benefit from higher timeframes
            higher_timeframes = self._get_higher_timeframes(self.primary_timeframe, 1)
            selected_timeframes.extend(higher_timeframes)
        elif strategy_type == "mean_reversion":
            # Mean reversion strategies benefit from lower timeframes
            lower_timeframes = self._get_lower_timeframes(self.primary_timeframe, 1)
            selected_timeframes.extend(lower_timeframes)
        elif strategy_type == "breakout":
            # Breakout strategies benefit from multiple timeframes
            higher_timeframes = self._get_higher_timeframes(self.primary_timeframe, 1)
            lower_timeframes = self._get_lower_timeframes(self.primary_timeframe, 1)
            selected_timeframes.extend(higher_timeframes + lower_timeframes)
        elif strategy_type == "scalping":
            # Scalping strategies focus on lower timeframes
            lower_timeframes = self._get_lower_timeframes(self.primary_timeframe, 2)
            selected_timeframes.extend(lower_timeframes)

        # Adjust based on volatility
        if self.volatility_adaptive and volatility is not None:
            if volatility > 0.03:  # High volatility
                # Add lower timeframes for more responsive trading
                lower_timeframes = self._get_lower_timeframes(self.primary_timeframe, 1)
                selected_timeframes.extend(lower_timeframes)
            elif volatility < 0.01:  # Low volatility
                # Add higher timeframes for more stable signals
                higher_timeframes = self._get_higher_timeframes(
                    self.primary_timeframe, 1
                )
                selected_timeframes.extend(higher_timeframes)

        # Adjust based on holding period
        if holding_period_minutes is not None:
            # Add timeframes that match the holding period
            matching_timeframes = self._get_matching_holding_period_timeframes(
                holding_period_minutes
            )
            selected_timeframes.extend(matching_timeframes)

        # Adjust based on trading frequency
        if trading_frequency == "high":
            # High-frequency trading focuses on lower timeframes
            lower_timeframes = self._get_lower_timeframes(self.primary_timeframe, 1)
            selected_timeframes.extend(lower_timeframes)
        elif trading_frequency == "low":
            # Low-frequency trading focuses on higher timeframes
            higher_timeframes = self._get_higher_timeframes(self.primary_timeframe, 1)
            selected_timeframes.extend(higher_timeframes)

        # Remove duplicates and sort by timeframe minutes
        unique_timeframes = list(set(selected_timeframes))
        sorted_timeframes = sorted(
            unique_timeframes, key=lambda tf: self.timeframe_minutes.get(tf, 0)
        )

        # Limit to max_timeframes
        if len(sorted_timeframes) > self.max_timeframes:
            # Always include primary timeframe
            final_timeframes = [self.primary_timeframe]

            # Add higher timeframes if enabled
            if self.include_higher_timeframes:
                higher_tfs = [
                    tf
                    for tf in sorted_timeframes
                    if self.timeframe_minutes.get(tf, 0) > primary_minutes
                ]
                higher_tfs = higher_tfs[
                    : min(len(higher_tfs), (self.max_timeframes - 1) // 2)
                ]
                final_timeframes.extend(higher_tfs)

            # Add lower timeframes if enabled
            if self.include_lower_timeframes:
                lower_tfs = [
                    tf
                    for tf in sorted_timeframes
                    if self.timeframe_minutes.get(tf, 0) < primary_minutes
                ]
                lower_tfs = lower_tfs[
                    -min(len(lower_tfs), (self.max_timeframes - 1) // 2) :
                ]
                final_timeframes.extend(lower_tfs)

            # Remove duplicates and sort
            final_timeframes = list(set(final_timeframes))
            sorted_timeframes = sorted(
                final_timeframes, key=lambda tf: self.timeframe_minutes.get(tf, 0)
            )

        # Store current selection for this strategy
        self.current_timeframes[strategy_type] = sorted_timeframes

        logger.info(
            f"Selected timeframes for {strategy_type} strategy: {sorted_timeframes}"
        )

        return sorted_timeframes

    def _get_higher_timeframes(self, timeframe: str, count: int = 1) -> list[str]:
        """
        Get higher timeframes than the specified timeframe.

        Args:
            timeframe: Reference timeframe
            count: Number of higher timeframes to return

        Returns:
            List of higher timeframes
        """
        # Get timeframe minutes
        minutes = self.timeframe_minutes.get(timeframe, 0)

        # Get higher timeframes
        higher_timeframes = [
            tf
            for tf in self.base_timeframes
            if self.timeframe_minutes.get(tf, 0) > minutes
        ]

        # Sort by minutes (ascending)
        higher_timeframes = sorted(
            higher_timeframes, key=lambda tf: self.timeframe_minutes.get(tf, 0)
        )

        return higher_timeframes[:count]

    def _get_lower_timeframes(self, timeframe: str, count: int = 1) -> list[str]:
        """
        Get lower timeframes than the specified timeframe.

        Args:
            timeframe: Reference timeframe
            count: Number of lower timeframes to return

        Returns:
            List of lower timeframes
        """
        # Get timeframe minutes
        minutes = self.timeframe_minutes.get(timeframe, 0)

        # Get lower timeframes
        lower_timeframes = [
            tf
            for tf in self.base_timeframes
            if self.timeframe_minutes.get(tf, 0) < minutes
        ]

        # Sort by minutes (descending)
        lower_timeframes = sorted(
            lower_timeframes,
            key=lambda tf: self.timeframe_minutes.get(tf, 0),
            reverse=True,
        )

        return lower_timeframes[:count]

    def _get_matching_holding_period_timeframes(
        self, holding_period_minutes: int
    ) -> list[str]:
        """
        Get timeframes that match the holding period.

        Args:
            holding_period_minutes: Holding period in minutes

        Returns:
            List of matching timeframes
        """
        # Find timeframes that are approximately 1/4 to 1/10 of the holding period
        min_minutes = holding_period_minutes / 10
        max_minutes = holding_period_minutes / 4

        matching_timeframes = [
            tf
            for tf in self.base_timeframes
            if min_minutes <= self.timeframe_minutes.get(tf, 0) <= max_minutes
        ]

        return matching_timeframes

    def get_optimal_timeframe_for_holding_period(
        self, holding_period_minutes: int
    ) -> str:
        """
        Get the optimal timeframe for a specific holding period.

        Args:
            holding_period_minutes: Holding period in minutes

        Returns:
            Optimal timeframe
        """
        # Target timeframe is approximately 1/6 of the holding period
        target_minutes = holding_period_minutes / 6

        # Find closest timeframe
        closest_timeframe = self.primary_timeframe
        closest_diff = float("inf")

        for tf in self.base_timeframes:
            tf_minutes = self.timeframe_minutes.get(tf, 0)
            diff = abs(tf_minutes - target_minutes)

            if diff < closest_diff:
                closest_timeframe = tf
                closest_diff = diff

        return closest_timeframe

    def get_multi_timeframe_set(
        self, primary_timeframe: str | None = None, include_count: int = 2
    ) -> list[str]:
        """
        Get a set of timeframes for multi-timeframe analysis.

        Args:
            primary_timeframe: Primary timeframe (defaults to self.primary_timeframe)
            include_count: Number of additional timeframes to include

        Returns:
            List of timeframes for multi-timeframe analysis
        """
        # Use default primary timeframe if not specified
        if primary_timeframe is None:
            primary_timeframe = self.primary_timeframe

        # Start with primary timeframe
        timeframes = [primary_timeframe]

        # Add higher timeframes
        higher_count = include_count // 2
        if higher_count > 0:
            higher_timeframes = self._get_higher_timeframes(
                primary_timeframe, higher_count
            )
            timeframes.extend(higher_timeframes)

        # Add lower timeframes
        lower_count = include_count - higher_count
        if lower_count > 0:
            lower_timeframes = self._get_lower_timeframes(
                primary_timeframe, lower_count
            )
            timeframes.extend(lower_timeframes)

        # Sort by timeframe minutes
        sorted_timeframes = sorted(
            timeframes, key=lambda tf: self.timeframe_minutes.get(tf, 0)
        )

        return sorted_timeframes

    def get_current_timeframes(
        self, strategy_type: str | None = None
    ) -> dict[str, list[str]]:
        """
        Get current timeframe selections.

        Args:
            strategy_type: Strategy type to get timeframes for

        Returns:
            Dictionary mapping strategy types to selected timeframes
        """
        if strategy_type:
            return {strategy_type: self.current_timeframes.get(strategy_type, [])}
        else:
            return self.current_timeframes
