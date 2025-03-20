#!/usr/bin/env python3
"""
Reporting System

This module provides a comprehensive reporting system that sends automated reports to Slack.
Features include:
1. System notifications for critical events
2. Daily/weekly trading performance reports
3. Portfolio status and analysis
4. Current positions and active orders tracking

The system uses Slack webhooks to send formatted messages to dedicated channels.
"""

import os
import time
import json
import logging
import requests
import threading
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import redis
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reporting_system')


class ReportingSystem:
    """
    Reporting system for sending automated reports to Slack
    """

    def __init__(self, redis_client):
        """Initialize the reporting system"""
        self.redis = redis_client

        # Load Slack configuration from environment variables
        self.slack_bot_token = os.environ.get('SLACK_BOT_TOKEN', '')
        self.webhooks = {
            'system_notifications': os.environ.get('SLACK_WEBHOOK_SYSTEM_NOTIFICATIONS', ''),
            'reports': os.environ.get('SLACK_WEBHOOK_REPORTS', ''),
            'portfolio': os.environ.get('SLACK_WEBHOOK_PORTFOLIO', ''),
            'current_positions': os.environ.get('SLACK_WEBHOOK_CURRENT_POSITIONS', '')
        }

        # Configuration
        self.config = {
            'report_schedule': {
                'daily_summary': '17:00',  # 5:00 PM
                'weekly_summary': 'Friday 17:30',  # Friday 5:30 PM
                'monthly_summary': '1 17:45',  # 1st day of month 5:45 PM
                # 9:35 AM, 12:00 PM, 3:55 PM
                'position_updates': ['09:35', '12:00', '15:55'],
                'portfolio_snapshot': '16:15'  # 4:15 PM
            },
            'notification_levels': {
                'critical': True,
                'warning': True,
                'info': True
            }
        }

        # State tracking
        self.running = False
        self.scheduler_thread = None

        logger.info("Reporting System initialized")

    def start(self):
        """Start the reporting system"""
        if self.running:
            logger.warning("Reporting system already running")
            return

        self.running = True
        logger.info("Starting reporting system")

        # Schedule reports
        self._schedule_reports()

        # Start scheduler in a separate thread
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        # Send startup notification
        self.send_system_notification(
            "Trading System Started",
            "The trading system has been successfully started and is now operational.",
            "info"
        )

        logger.info("Reporting system started")

    def stop(self):
        """Stop the reporting system"""
        if not self.running:
            logger.warning("Reporting system not running")
            return

        logger.info("Stopping reporting system")
        self.running = False

        # Send shutdown notification
        self.send_system_notification(
            "Trading System Shutting Down",
            "The trading system is shutting down. All positions and orders will be preserved.",
            "warning"
        )

        # Wait for scheduler thread to terminate
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)

        logger.info("Reporting system stopped")

    def _schedule_reports(self):
        """Schedule all reports based on configuration"""
        # Daily summary report
        schedule.every().day.at(self.config['report_schedule']['daily_summary']).do(
            self.send_daily_summary_report
        )

        # Weekly summary report
        day, time = self.config['report_schedule']['weekly_summary'].split()
        getattr(schedule.every(), day.lower()).at(time).do(
            self.send_weekly_summary_report
        )

        # Monthly summary report
        day, time = self.config['report_schedule']['monthly_summary'].split()
        schedule.every().month.at(f"{day} {time}").do(
            self.send_monthly_summary_report
        )

        # Position updates
        for time_str in self.config['report_schedule']['position_updates']:
            schedule.every().day.at(time_str).do(
                self.send_positions_report
            )

        # Portfolio snapshot
        schedule.every().day.at(self.config['report_schedule']['portfolio_snapshot']).do(
            self.send_portfolio_report
        )

        logger.info("Report schedules configured")

    def _run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("Starting scheduler thread")

        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler: {str(e)}")
                time.sleep(10)

    def send_slack_message(self, webhook_url, blocks):
        """
        Send a formatted message to Slack

        Args:
            webhook_url: The webhook URL to send the message to
            blocks: The formatted blocks for the message
        """
        try:
            if not webhook_url:
                logger.error("No webhook URL provided")
                return False

            payload = {
                "blocks": blocks
            }

            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                logger.info(f"Message sent to Slack successfully")
                return True
            else:
                logger.error(
                    f"Failed to send message to Slack: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending message to Slack: {str(e)}")
            return False

    def send_system_notification(self, title, message, level="info"):
        """
        Send a system notification to the system-notifications channel

        Args:
            title: The notification title
            message: The notification message
            level: The notification level (info, warning, critical)
        """
        if not self.config['notification_levels'].get(level, False):
            return

        # Set emoji and color based on level
        if level == "critical":
            emoji = ":red_circle:"
            color = "#FF0000"
        elif level == "warning":
            emoji = ":warning:"
            color = "#FFA500"
        else:  # info
            emoji = ":information_source:"
            color = "#0000FF"

        # Format timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Level:* {level.upper()} | *Time:* {timestamp}"
                    }
                ]
            },
            {
                "type": "divider"
            }
        ]

        # Send to Slack
        return self.send_slack_message(self.webhooks['system_notifications'], blocks)

    def send_daily_summary_report(self):
        """Send daily trading summary report to the reports channel"""
        try:
            # Get daily stats from Redis
            daily_stats = self.redis.hgetall("execution:daily_stats")

            if not daily_stats:
                logger.warning("No daily stats available for report")
                return

            # Convert bytes to string for keys
            stats = {k.decode('utf-8') if isinstance(k, bytes) else k:
                     v.decode('utf-8') if isinstance(v, bytes) else v
                     for k, v in daily_stats.items()}

            # Get trade history for today
            trades = self._get_todays_trades()

            # Calculate metrics
            trades_executed = int(stats.get('trades_executed', 0))
            profitable_trades = int(stats.get('profitable_trades', 0))
            win_rate = (profitable_trades / trades_executed *
                        100) if trades_executed > 0 else 0
            total_pnl = float(stats.get('total_pnl', 0))

            # Format timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

            # Create blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f":chart_with_upwards_trend: Daily Trading Summary - {timestamp}",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Trades Executed:*\n{trades_executed}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Win Rate:*\n{win_rate:.1f}%"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Total P&L:*\n${total_pnl:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Current Exposure:*\n${float(stats.get('current_exposure', 0)):.2f}"
                        }
                    ]
                },
                {
                    "type": "divider"
                }
            ]

            # Add trade summary if available
            if trades:
                trade_text = "*Today's Trades:*\n"
                for i, trade in enumerate(trades[:5]):  # Show top 5 trades
                    direction = ":arrow_up:" if trade.get(
                        'direction') == 'long' else ":arrow_down:"
                    pnl = float(trade.get('realized_pnl', 0))
                    pnl_emoji = ":large_green_circle:" if pnl > 0 else ":red_circle:"

                    trade_text += f"{direction} *{trade.get('ticker')}*: {trade.get('quantity')} shares, PnL: {pnl_emoji} ${pnl:.2f}\n"

                if len(trades) > 5:
                    trade_text += f"_...and {len(trades) - 5} more trades_"

                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": trade_text
                    }
                })

            # Add footer
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Generated at {datetime.datetime.now().strftime('%H:%M:%S')}"
                    }
                ]
            })

            # Send to Slack
            return self.send_slack_message(self.webhooks['reports'], blocks)

        except Exception as e:
            logger.error(f"Error generating daily summary report: {str(e)}")
            return False

    def send_weekly_summary_report(self):
        """Send weekly trading summary report to the reports channel"""
        try:
            # Get weekly stats
            weekly_stats = self._calculate_weekly_stats()

            if not weekly_stats:
                logger.warning("No weekly stats available for report")
                return

            # Format date range
            today = datetime.datetime.now()
            start_of_week = (
                today - datetime.timedelta(days=today.weekday())).strftime("%Y-%m-%d")
            end_of_week = today.strftime("%Y-%m-%d")

            # Create blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f":calendar: Weekly Trading Summary ({start_of_week} to {end_of_week})",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Total Trades:*\n{weekly_stats['total_trades']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Win Rate:*\n{weekly_stats['win_rate']:.1f}%"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Total P&L:*\n${weekly_stats['total_pnl']:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Best Day:*\n{weekly_stats['best_day']} (${weekly_stats['best_day_pnl']:.2f})"
                        }
                    ]
                },
                {
                    "type": "divider"
                }
            ]

            # Add performance chart (placeholder - in a real system, you'd generate a chart)
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Weekly Performance Breakdown:*"
                }
            })

            # Add best/worst performers
            if weekly_stats['best_performers'] or weekly_stats['worst_performers']:
                best_text = "*Best Performers:*\n"
                for ticker, pnl in weekly_stats['best_performers']:
                    best_text += f"• {ticker}: ${pnl:.2f}\n"

                worst_text = "*Worst Performers:*\n"
                for ticker, pnl in weekly_stats['worst_performers']:
                    worst_text += f"• {ticker}: ${pnl:.2f}\n"

                blocks.append({
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": best_text
                        },
                        {
                            "type": "mrkdwn",
                            "text": worst_text
                        }
                    ]
                })

            # Add footer
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            })

            # Send to Slack
            return self.send_slack_message(self.webhooks['reports'], blocks)

        except Exception as e:
            logger.error(f"Error generating weekly summary report: {str(e)}")
            return False

    def send_monthly_summary_report(self):
        """Send monthly trading summary report to the reports channel"""
        try:
            # Get monthly stats
            monthly_stats = self._calculate_monthly_stats()

            if not monthly_stats:
                logger.warning("No monthly stats available for report")
                return

            # Format date range
            today = datetime.datetime.now()
            first_day = today.replace(day=1).strftime("%Y-%m-%d")
            last_day = today.strftime("%Y-%m-%d")

            # Create blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f":calendar: Monthly Trading Summary ({first_day} to {last_day})",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Total Trades:*\n{monthly_stats['total_trades']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Win Rate:*\n{monthly_stats['win_rate']:.1f}%"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Total P&L:*\n${monthly_stats['total_pnl']:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Max Drawdown:*\n{monthly_stats['max_drawdown']:.2f}%"
                        }
                    ]
                },
                {
                    "type": "divider"
                }
            ]

            # Add strategy performance
            if monthly_stats['strategy_performance']:
                strategy_text = "*Strategy Performance:*\n"
                for strategy, metrics in monthly_stats['strategy_performance'].items():
                    strategy_text += f"• {strategy}: ${metrics['pnl']:.2f} ({metrics['win_rate']:.1f}% win rate)\n"

                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": strategy_text
                    }
                })

            # Add monthly metrics
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Sharpe Ratio:*\n{monthly_stats.get('sharpe_ratio', 'N/A')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Avg Trade Duration:*\n{monthly_stats.get('avg_trade_duration', 'N/A')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Profit Factor:*\n{monthly_stats.get('profit_factor', 'N/A')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Avg Win/Loss Ratio:*\n{monthly_stats.get('avg_win_loss_ratio', 'N/A')}"
                    }
                ]
            })

            # Add footer
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            })

            # Send to Slack
            return self.send_slack_message(self.webhooks['reports'], blocks)

        except Exception as e:
            logger.error(f"Error generating monthly summary report: {str(e)}")
            return False

    def send_positions_report(self):
        """Send current positions report to the current-positions channel"""
        try:
            # Get active positions from Redis
            positions = self.redis.hgetall("positions:active")

            if not positions:
                # Check if market is open before sending "no positions" message
                is_market_open = self._is_market_open()
                if not is_market_open:
                    return  # Don't send empty position reports when market is closed

                blocks = [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": ":chart_with_upwards_trend: Current Positions",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "No active positions at this time."
                        }
                    }
                ]

                return self.send_slack_message(self.webhooks['current_positions'], blocks)

            # Process positions
            position_list = []
            total_value = 0
            total_pnl = 0

            for key, pos_data in positions.items():
                position = json.loads(pos_data.decode(
                    'utf-8') if isinstance(pos_data, bytes) else pos_data)
                position_list.append(position)
                total_value += position.get('current_value', 0)
                total_pnl += position.get('unrealized_pnl', 0)

            # Sort positions by unrealized P&L
            position_list.sort(key=lambda x: x.get(
                'unrealized_pnl', 0), reverse=True)

            # Create blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f":chart_with_upwards_trend: Current Positions ({len(position_list)})",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Total Value:*\n${total_value:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Unrealized P&L:*\n${total_pnl:.2f}"
                        }
                    ]
                },
                {
                    "type": "divider"
                }
            ]

            # Add position details
            for position in position_list:
                ticker = position.get('ticker', 'UNKNOWN')
                direction = "LONG" if position.get(
                    'direction', '') == 'long' else "SHORT"
                quantity = position.get('quantity', 0)
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', 0)
                unrealized_pnl = position.get('unrealized_pnl', 0)
                unrealized_pnl_pct = position.get('unrealized_pnl_pct', 0)

                # Determine emoji based on P&L
                if unrealized_pnl > 0:
                    emoji = ":large_green_circle:"
                elif unrealized_pnl < 0:
                    emoji = ":red_circle:"
                else:
                    emoji = ":white_circle:"

                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{emoji} *{ticker}* ({direction})\n{quantity} shares @ ${entry_price:.2f}\nCurrent: ${current_price:.2f} | P&L: ${unrealized_pnl:.2f} ({unrealized_pnl_pct:.2f}%)"
                    }
                })

            # Add footer with timestamp
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Updated at {datetime.datetime.now().strftime('%H:%M:%S')}"
                    }
                ]
            })

            # Send to Slack
            return self.send_slack_message(self.webhooks['current_positions'], blocks)

        except Exception as e:
            logger.error(f"Error generating positions report: {str(e)}")
            return False

    def send_portfolio_report(self):
        """Send portfolio status report to the portfolio channel"""
        try:
            # Get account information
            account_info = self._get_account_info()

            if not account_info:
                logger.warning(
                    "No account information available for portfolio report")
                return

            # Get historical equity data
            equity_history = self._get_equity_history()

            # Calculate portfolio metrics
            daily_change = 0
            weekly_change = 0
            monthly_change = 0

            if equity_history:
                current_equity = account_info.get('equity', 0)
                if len(equity_history) >= 1:
                    daily_change = (current_equity /
                                    equity_history[0] - 1) * 100
                if len(equity_history) >= 5:
                    weekly_change = (current_equity /
                                     equity_history[4] - 1) * 100
                if len(equity_history) >= 20:
                    monthly_change = (
                        current_equity / equity_history[19] - 1) * 100

            # Create blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": ":briefcase: Portfolio Status",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Equity:*\n${account_info.get('equity', 0):.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Cash:*\n${account_info.get('cash', 0):.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Buying Power:*\n${account_info.get('buying_power', 0):.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Day Trading BP:*\n${account_info.get('daytrading_buying_power', 0):.2f}"
                        }
                    ]
                },
                {
                    "type": "divider"
                }
            ]

            # Add performance metrics
            daily_emoji = ":chart_with_upwards_trend:" if daily_change >= 0 else ":chart_with_downwards_trend:"
            weekly_emoji = ":chart_with_upwards_trend:" if weekly_change >= 0 else ":chart_with_downwards_trend:"
            monthly_emoji = ":chart_with_upwards_trend:" if monthly_change >= 0 else ":chart_with_downwards_trend:"

            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Daily Change:*\n{daily_emoji} {daily_change:.2f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Weekly Change:*\n{weekly_emoji} {weekly_change:.2f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Monthly Change:*\n{monthly_emoji} {monthly_change:.2f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*YTD Change:*\n{monthly_emoji} {account_info.get('ytd_change', 0):.2f}%"
                    }
                ]
            })

            # Add allocation breakdown
            allocation = self._get_portfolio_allocation()

            if allocation:
                allocation_text = "*Portfolio Allocation:*\n"
                for category, percentage in allocation.items():
                    allocation_text += f"• {category}: {percentage:.1f}%\n"

                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": allocation_text
                    }
                })

            # Add footer
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            })

            # Send to Slack
            return self.send_slack_message(self.webhooks['portfolio'], blocks)

        except Exception as e:
            logger.error(f"Error generating portfolio report: {str(e)}")
            return False

    def _get_todays_trades(self):
        """Get today's trades from Redis"""
        try:
            trades = self.redis.hgetall("trades:history")

            if not trades:
                return []

            # Process trades
            trade_list = []
            today = datetime.datetime.now().date()

            for trade_id, trade_data in trades.items():
                trade = json.loads(trade_data.decode(
                    'utf-8') if isinstance(trade_data, bytes) else trade_data)

                # Check if trade was executed today
                trade_time = datetime.datetime.fromtimestamp(
                    trade.get('exit_time', 0)).date()
                if trade_time == today:
                    trade_list.append(trade)

            # Sort by P&L
            trade_list.sort(key=lambda x: x.get(
                'realized_pnl', 0), reverse=True)

            return trade_list

        except Exception as e:
            logger.error(f"Error getting today's trades: {str(e)}")
            return []

    def _calculate_weekly_stats(self):
        """Calculate weekly trading statistics"""
        try:
            trades = self.redis.hgetall("trades:history")

            if not trades:
                return None

            # Process trades
            trade_list = []
            today = datetime.datetime.now().date()
            start_of_week = today - datetime.timedelta(days=today.weekday())

            for trade_id, trade_data in trades.items():
                trade = json.loads(trade_data.decode(
                    'utf-8') if isinstance(trade_data, bytes) else trade_data)

                # Check if trade was executed this week
                trade_time = datetime.datetime.fromtimestamp(
                    trade.get('exit_time', 0)).date()
                if trade_time >= start_of_week:
                    trade_list.append(trade)

            if not trade_list:
                return None

            # Calculate statistics
            total_trades = len(trade_list)
            profitable_trades = sum(
                1 for t in trade_list if t.get('realized_pnl', 0) > 0)
            win_rate = (profitable_trades / total_trades *
                        100) if total_trades > 0 else 0
            total_pnl = sum(t.get('realized_pnl', 0) for t in trade_list)

            # Group trades by day
            trades_by_day = {}
            for trade in trade_list:
                day = datetime.datetime.fromtimestamp(
                    trade.get('exit_time', 0)).strftime('%Y-%m-%d')
                if day not in trades_by_day:
                    trades_by_day[day] = []
                trades_by_day[day].append(trade)

            # Find best day
            best_day = None
            best_day_pnl = 0
            for day, day_trades in trades_by_day.items():
                day_pnl = sum(t.get('realized_pnl', 0) for t in day_trades)
                if day_pnl > best_day_pnl:
                    best_day = day
                    best_day_pnl = day_pnl

            # Group trades by ticker
            trades_by_ticker = {}
            for trade in trade_list:
                ticker = trade.get('ticker', 'UNKNOWN')
                if ticker not in trades_by_ticker:
                    trades_by_ticker[ticker] = []
                trades_by_ticker[ticker].append(trade)

            # Calculate PnL by ticker
            pnl_by_ticker = {}
            for ticker, ticker_trades in trades_by_ticker.items():
                pnl_by_ticker[ticker] = sum(
                    t.get('realized_pnl', 0) for t in ticker_trades)

            # Get best and worst performers
            sorted_tickers = sorted(
                pnl_by_ticker.items(), key=lambda x: x[1], reverse=True)
            best_performers = sorted_tickers[:3]  # Top 3
            worst_performers = sorted_tickers[-3:]  # Bottom 3

            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'best_day': best_day,
                'best_day_pnl': best_day_pnl,
                'best_performers': best_performers,
                'worst_performers': worst_performers
            }

        except Exception as e:
            logger.error(f"Error calculating weekly stats: {str(e)}")
            return None

    def _calculate_monthly_stats(self):
        """Calculate monthly trading statistics"""
        try:
            trades = self.redis.hgetall("trades:history")

            if not trades:
                return None

            # Process trades
            trade_list = []
            today = datetime.datetime.now().date()
            start_of_month = today.replace(day=1)

            for trade_id, trade_data in trades.items():
                trade = json.loads(trade_data.decode(
                    'utf-8') if isinstance(trade_data, bytes) else trade_data)

                # Check if trade was executed this month
                trade_time = datetime.datetime.fromtimestamp(
                    trade.get('exit_time', 0)).date()
                if trade_time >= start_of_month:
                    trade_list.append(trade)

            if not trade_list:
                return None

            # Calculate statistics
            total_trades = len(trade_list)
            profitable_trades = sum(
                1 for t in trade_list if t.get('realized_pnl', 0) > 0)
            win_rate = (profitable_trades / total_trades *
                        100) if total_trades > 0 else 0
            total_pnl = sum(t.get('realized_pnl', 0) for t in trade_list)

            # Calculate max drawdown (simplified)
            max_drawdown = 0

            # Group trades by strategy (using signal source as proxy)
            trades_by_strategy = {}
            for trade in trade_list:
                strategy = trade.get('signal', {}).get('source', 'Unknown')
                if strategy not in trades_by_strategy:
                    trades_by_strategy[strategy] = []
                trades_by_strategy[strategy].append(trade)

            # Calculate strategy performance
            strategy_performance = {}
            for strategy, strategy_trades in trades_by_strategy.items():
                strategy_pnl = sum(t.get('realized_pnl', 0)
                                   for t in strategy_trades)
                strategy_win_rate = (sum(1 for t in strategy_trades if t.get(
                    'realized_pnl', 0) > 0) / len(strategy_trades) * 100) if strategy_trades else 0

                strategy_performance[strategy] = {
                    'pnl': strategy_pnl,
                    'win_rate': strategy_win_rate,
                    'trade_count': len(strategy_trades)
                }

            # Calculate additional metrics
            winning_trades = [
                t for t in trade_list if t.get('realized_pnl', 0) > 0]
            losing_trades = [
                t for t in trade_list if t.get('realized_pnl', 0) < 0]

            avg_win = sum(t.get('realized_pnl', 0) for t in winning_trades) / \
                len(winning_trades) if winning_trades else 0
            avg_loss = sum(abs(t.get('realized_pnl', 0))
                           for t in losing_trades) / len(losing_trades) if losing_trades else 0

            avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            profit_factor = sum(t.get('realized_pnl', 0) for t in winning_trades) / sum(
                abs(t.get('realized_pnl', 0)) for t in losing_trades) if losing_trades else 0

            # Calculate average trade duration
            durations = []
            for trade in trade_list:
                entry_time = trade.get('entry_time', 0)
                exit_time = trade.get('exit_time', 0)
                if entry_time and exit_time:
                    duration_seconds = exit_time - entry_time
                    durations.append(duration_seconds)

            avg_duration_seconds = sum(
                durations) / len(durations) if durations else 0
            avg_duration_minutes = avg_duration_seconds / 60

            # Format duration
            if avg_duration_minutes < 60:
                avg_trade_duration = f"{avg_duration_minutes:.1f} min"
            else:
                avg_trade_duration = f"{avg_duration_minutes/60:.1f} hours"

            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'strategy_performance': strategy_performance,
                'avg_win_loss_ratio': f"{avg_win_loss_ratio:.2f}",
                'profit_factor': f"{profit_factor:.2f}",
                'avg_trade_duration': avg_trade_duration,
                'sharpe_ratio': "N/A"  # Would require daily returns calculation
            }

        except Exception as e:
            logger.error(f"Error calculating monthly stats: {str(e)}")
            return None

    def _get_account_info(self):
        """Get account information from Redis"""
        try:
            # Try to get from Redis first
            account_info = self.redis.get("account:info")

            if account_info:
                return json.loads(account_info.decode('utf-8') if isinstance(account_info, bytes) else account_info)

            # If not in Redis, return placeholder data
            return {
                'equity': 100000.0,
                'cash': 50000.0,
                'buying_power': 200000.0,
                'daytrading_buying_power': 400000.0,
                'ytd_change': 5.2
            }

        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None

    def _get_equity_history(self):
        """Get historical equity data from Redis"""
        try:
            # Try to get from Redis first
            equity_history = self.redis.get("account:equity_history")

            if equity_history:
                return json.loads(equity_history.decode('utf-8') if isinstance(equity_history, bytes) else equity_history)

            # If not in Redis, return placeholder data
            return [100000.0] * 20  # 20 days of data

        except Exception as e:
            logger.error(f"Error getting equity history: {str(e)}")
            return None

    def _get_portfolio_allocation(self):
        """Get portfolio allocation by sector/asset class"""
        try:
            # Try to get from Redis first
            allocation = self.redis.get("portfolio:allocation")

            if allocation:
                return json.loads(allocation.decode('utf-8') if isinstance(allocation, bytes) else allocation)

            # If not in Redis, calculate from positions
            positions = self.redis.hgetall("positions:active")

            if not positions:
                return {
                    "Cash": 100.0
                }

            # Process positions
            total_value = 0
            sector_values = {}

            for key, pos_data in positions.items():
                position = json.loads(pos_data.decode(
                    'utf-8') if isinstance(pos_data, bytes) else pos_data)
                value = position.get('current_value', 0)
                total_value += value

                # Get sector from position metadata or use default
                sector = position.get('metadata', {}).get('sector', 'Other')

                if sector not in sector_values:
                    sector_values[sector] = 0

                sector_values[sector] += value

            # Calculate percentages
            allocation = {}
            for sector, value in sector_values.items():
                allocation[sector] = (
                    value / total_value * 100) if total_value > 0 else 0

            return allocation

        except Exception as e:
            logger.error(f"Error getting portfolio allocation: {str(e)}")
            return None

    def _is_market_open(self):
        """Check if market is currently open"""
        try:
            # Try to get from Redis first
            market_status = self.redis.get("market:is_open")

            if market_status:
                return market_status.decode('utf-8') == "1"

            # If not in Redis, use time-based check
            now = datetime.datetime.now(datetime.timezone.utc)
            eastern = now.astimezone(
                datetime.timezone(-datetime.timedelta(hours=5)))  # Eastern Time

            # Check if it's a weekday
            if eastern.weekday() >= 5:  # Saturday or Sunday
                return False

            # Check if it's during market hours (9:30 AM - 4:00 PM ET)
            market_open = eastern.replace(
                hour=9, minute=30, second=0, microsecond=0)
            market_close = eastern.replace(
                hour=16, minute=0, second=0, microsecond=0)

            return market_open <= eastern <= market_close

        except Exception as e:
            logger.error(f"Error checking if market is open: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    import redis

    # Create Redis client
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6380)),
        db=int(os.environ.get('REDIS_DB', 0))
    )

    # Create reporting system
    reporting_system = ReportingSystem(redis_client)

    # Start system
    reporting_system.start()

    try:
        # Run for a while
        print("Reporting system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop system
        reporting_system.stop()
