# Unusual Whales API Documentation

This document provides comprehensive information about the Unusual Whales API endpoints, request parameters, and response formats.

## Base URL

```
https://api.unusualwhales.com/api
```

## Authentication

Provide your bearer token in the Authorization header when making requests to protected resources.

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Flow Alerts

#### Get Flow Alerts for a Ticker

```
GET /stock/{ticker}/flow-alerts
```

Returns the latest flow alerts for the given ticker.

**Path Parameters:**
- `ticker` (string, required): A single ticker (e.g., "AAPL")

**Query Parameters:**
- `is_ask_side` (boolean, default: true): Boolean flag whether a transaction is ask side
- `is_bid_side` (boolean, default: true): Boolean flag whether a transaction is bid side
- `limit` (integer, default: 100, max: 200, min: 1): How many items to return

**Example Response:**
```json
{
  "data": [
    {
      "alert_rule": "RepeatedHits",
      "all_opening_trades": false,
      "created_at": "2023-12-12T16:35:52.168490Z",
      "expiry": "2023-12-22",
      "expiry_count": 1,
      "has_floor": false,
      "has_multileg": false,
      "has_singleleg": true,
      "has_sweep": true,
      "open_interest": 7913,
      "option_chain": "MSFT231222C00375000",
      "price": "4.05",
      "strike": "375",
      "ticker": "MSFT",
      "total_ask_side_prem": "151875",
      "total_bid_side_prem": "405",
      "total_premium": "186705",
      "total_size": 461,
      "trade_count": 32,
      "type": "call",
      "underlying_price": "372.99",
      "volume": 2442,
      "volume_oi_ratio": "0.30860609124226"
    }
  ]
}
```

### Alerts

#### Get All Alerts

```
GET /alerts
```

Returns all the alerts that have been triggered for the user.

**Query Parameters:**
- `config_ids[]` (array): A list of alert ids to filter by
- `intraday_only` (boolean, default: true): Boolean flag whether to return only intraday alerts
- `limit` (integer, default: 100, max: 200, min: 1): How many items to return
- `noti_types[]` (array[string]): A list of notification types
- `page` (integer, default: 0, min: 0): The page number to return
- `ticker_symbols` (string): A comma separated list of tickers

**Example Response:**
```json
{
  "data": [
    {
      "created_at": "2024-12-11T14:00:00Z",
      "id": "fdc2cf91-d387-480f-a79e-28026447a6f5",
      "meta": {
        "alert_type": "PayDate",
        "date": "2024-11-21",
        "div_yield": "0.0503",
        "dividend": "0.1275",
        "frequency": "Quarterly",
        "payment_date": "2024-12-11",
        "prev_dividend": "0.125"
      },
      "name": "S&P 500 Dividends",
      "noti_type": "dividends",
      "symbol": "AMCR",
      "symbol_type": "stock",
      "tape_time": "2024-12-11T14:00:00Z",
      "user_noti_config_id": "cb70c287-f10a-4e63-98ad-571b7dafc8e4"
    }
  ]
}
```

### Alert Configurations

#### Get Alert Configurations

```
GET /alerts/configuration
```

Returns all alert configurations of the user.

**Example Response:**
```json
{
  "data": [
    {
      "config": {
        "option_symbols": [
          "TGT241122C00177500"
        ],
        "symbols": "all"
      },
      "created_at": "2024-11-19T18:19:14Z",
      "id": "ebe24953-a0bf-4b4d-98be-14f721a1199a",
      "mobile_only": false,
      "name": "Chain OI Chg: TGT241122C00177500",
      "noti_type": "chain_oi_change",
      "status": "active"
    }
  ]
}
```

### Dark Pool Data

#### Get Recent Dark Pool Trades

```
GET /darkpool/recent
```

Returns the latest darkpool trades.

**Query Parameters:**
- `date` (string): A trading date in the format of YYYY-MM-DD
- `limit` (integer, default: 100, max: 200, min: 1): How many items to return
- `max_premium` (integer): The maximum premium requested trades should have
- `max_size` (integer): The maximum size requested trades should have
- `max_volume` (integer): The maximum consolidated volume requested trades should have
- `min_premium` (integer, default: 0): The minimum premium requested trades should have
- `min_size` (integer, default: 0): The minimum size requested trades should have
- `min_volume` (integer, default: 0): The minimum consolidated volume requested trades should have

**Example Response:**
```json
{
  "data": [
    {
      "canceled": false,
      "executed_at": "2023-02-16T00:59:44Z",
      "ext_hour_sold_codes": "extended_hours_trade",
      "market_center": "L",
      "nbbo_ask": "19",
      "nbbo_ask_quantity": 6600,
      "nbbo_bid": "18.99",
      "nbbo_bid_quantity": 29100,
      "premium": "121538.56",
      "price": "18.9904",
      "sale_cond_codes": null,
      "size": 6400,
      "ticker": "QID",
      "tracking_id": 71984388012245,
      "trade_code": null,
      "trade_settlement": "regular_settlement",
      "volume": 9946819
    }
  ]
}
```

#### Get Dark Pool Trades for a Ticker

```
GET /darkpool/{ticker}
```

Returns the darkpool trades for the given ticker on a given day.

**Path Parameters:**
- `ticker` (string, required): A single ticker (e.g., "AAPL")

**Query Parameters:**
- `date` (string): A trading date in the format of YYYY-MM-DD
- `limit` (integer, default: 500, max: 500, min: 1): How many items to return
- `max_premium` (integer): The maximum premium requested trades should have
- `max_size` (integer): The maximum size requested trades should have
- `max_volume` (integer): The maximum consolidated volume requested trades should have
- `min_premium` (integer, default: 0): The minimum premium requested trades should have
- `min_size` (integer, default: 0): The minimum size requested trades should have
- `min_volume` (integer, default: 0): The minimum consolidated volume requested trades should have
- `newer_than` (string): The unix time in milliseconds or seconds
- `older_than` (string): The unix time in milliseconds or seconds

**Example Response:**
```json
{
  "data": [
    {
      "canceled": false,
      "executed_at": "2023-02-16T00:59:44Z",
      "ext_hour_sold_codes": "extended_hours_trade",
      "market_center": "L",
      "nbbo_ask": "19",
      "nbbo_ask_quantity": 6600,
      "nbbo_bid": "18.99",
      "nbbo_bid_quantity": 29100,
      "premium": "121538.56",
      "price": "18.9904",
      "sale_cond_codes": null,
      "size": 6400,
      "ticker": "QID",
      "tracking_id": 71984388012245,
      "trade_code": null,
      "trade_settlement": "regular_settlement",
      "volume": 9946819
    }
  ]
}
```

### Insider Trading Data

#### Get Insider Transactions

```
GET /insider/transactions
```

Returns the latest insider transactions.

**Query Parameters:**
- `common_stock_only` (string): Filter by common stock only
- `industries` (string): Filter by industries
- `is_director` (string): Filter by director status
- `is_officer` (string): Filter by officer status
- `is_s_p_500` (string): Filter by S&P 500 status
- `is_ten_percent_owner` (string): Filter by 10% owner status
- `market_cap_size` (string): Filter by market cap size
- `max_marketcap` (integer): The maximum marketcap
- `min_marketcap` (integer): The minimum marketcap
- `ticker_symbol` (string): A comma separated list of tickers
- `transaction_codes` (string): Filter by transaction codes

**Example Response:**
```json
{
  "data": [
    {
      "amount": -35921,
      "date_excercisable": null,
      "director_indirect": null,
      "expiration_date": null,
      "filing_date": "2024-12-12",
      "formtype": "144",
      "id": "2662b73d-0ad6-4568-adb0-739b96c18090",
      "ids": [
        "26f2911d-8e40-4fd4-9b74-c450e203a0ad"
      ],
      "is_director": true,
      "is_officer": true,
      "is_s_p_500": true,
      "is_ten_percent_owner": true,
      "marketcap": "1375122749418",
      "natureofownership": null,
      "next_earnings_date": "2025-02-06",
      "officer_title": "COB and CEO",
      "owner_name": "ZUCKERBERG MARK",
      "price": "632",
      "price_excercisable": null,
      "sector": "Communication Services",
      "security_ad_code": null,
      "security_title": null,
      "shares_owned_after": null,
      "shares_owned_before": null,
      "stock_price": null,
      "ticker": "META",
      "transaction_code": "S",
      "transaction_date": "2024-12-12",
      "transactions": 1
    }
  ]
}
```

#### Get Sector Flow

```
GET /insider/{sector}/sector-flow
```

Returns an aggregated view of the insider flow for the given sector.

**Path Parameters:**
- `sector` (string, required): A financial sector (e.g., "Technology")

**Example Response:**
```json
{
  "data": [
    {
      "avg_price": 162.32,
      "buy_sell": "sell",
      "date": "2024-12-12",
      "premium": "664386",
      "transactions": 54,
      "uniq_insiders": 10,
      "volume": 244331
    }
  ]
}
```

#### Get Insiders for a Ticker

```
GET /insider/{ticker}
```

Returns all insiders for the given ticker.

**Path Parameters:**
- `ticker` (string, required): A single ticker (e.g., "AAPL")

**Example Response:**
```json
{
  "data": [
    {
      "display_name": "KARP ALEXANDER",
      "id": 10343,
      "is_person": true,
      "logo_url": "https://storage.googleapis.com/uwassets/insiders/10343",
      "name": "KARP ALEXANDER",
      "name_slug": "karp-alexander",
      "social_links": [],
      "ticker": "PLTR"
    }
  ]
}
```

#### Get Ticker Flow

```
GET /insider/{ticker}/ticker-flow
```

Returns an aggregated view of the insider flow for the given ticker.

**Path Parameters:**
- `ticker` (string, required): A single ticker (e.g., "AAPL")

**Example Response:**
```json
{
  "data": [
    {
      "avg_price": 162.32,
      "buy_sell": "sell",
      "date": "2024-12-12",
      "premium": "664386",
      "transactions": 54,
      "uniq_insiders": 10,
      "volume": 244331
    }
  ]
}