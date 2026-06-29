# O'Donoghues Suffolk Street Demand Prediction System

## Overview
This project is a practical operations tool for O'Donoghues on Suffolk Street in Dublin, intended to help the kitchen and bar predict busy periods, improve prep planning, support staffing decisions, and reduce stock risk. The venue operates in Dublin city centre, serves food from 9 AM to 9 PM, and has live music in the evenings, which means demand is likely to vary significantly across the day and be influenced by both venue-specific and city-wide factors.[1]

The goal is not academic novelty. The goal is to build a usable forecasting system that helps answer simple operational questions such as: how busy will lunch be, will the evening food service be heavy, and is there a high-risk late rush that the team should prepare for.[1]

## Problem Statement
The kitchen and bar need better visibility into short-term demand so they can plan prep, staffing, and stock more effectively. A forecasting system can estimate hourly or shift-level demand and turn it into actionable signals such as quiet, normal, busy, or slammed.[2][3]

This is especially relevant in a city-centre Dublin pub because demand may be driven not only by internal patterns like weekday and hour, but also by weather, live music, tourism flows, airport arrivals, cruises, and major events.[4][5][1]

## Core Use Cases
The system should support decisions such as:

- How many food tickets are expected during lunch.
- Whether the evening kitchen service is likely to be quiet or heavy.
- Whether extra prep is needed for fast-moving menu items.
- Whether bar and kitchen staffing should be increased for a given shift.
- Whether a specific day should be flagged early as high-risk because several demand drivers line up at once.

## Forecast Targets
The first version should avoid trying to predict everything at once. The most useful targets are:

- Hourly food tickets.
- Hourly POS transactions.
- Hourly sales total.
- Covers or customer count, if available.
- Shift label such as quiet, normal, busy, or slammed.

A useful structure is to create three separate predictions:

1. Lunch kitchen demand.
2. Evening food demand.
3. Late bar rush risk.

This split is helpful because food service hours differ from overall opening hours, and tourism or live music may affect different parts of the day differently.[1]

## Internal Data To Collect
The internal data is the most important part of the system. Public data can improve the model, but the strongest signal usually comes from the business's own operating history.[3][2]

Recommended internal columns:

| Column | Purpose |
|---|---|
| `timestamp_hour` | Main time key for training and prediction |
| `orders_count` | Core operational demand measure |
| `food_tickets_count` | Best direct kitchen demand measure |
| `sales_total` | Useful backup or secondary target |
| `covers_count` | Direct customer-flow proxy if available |
| `reservations_count` | Forward-looking demand signal |
| `walkins_estimate` | Helps separate booking-led vs street traffic |
| `bar_staff_count` | Useful for operational context |
| `kitchen_staff_count` | Helps interpret throughput and service pressure |
| `live_music_flag` | Important local demand driver |
| `special_event_flag` | Match day, promotions, private bookings |
| `busy_label` | Manual label such as quiet, normal, busy, slammed |
| `stockout_flag` | Explains unusual low sales caused by missing items |
| `menu_change_flag` | Helps account for demand shifts from menu changes |

If item-level data is available, it is also useful to collect top-selling menu items per hour and key high-turnover dish counts.

## Public Data Available
Several useful signals are publicly available and can be added to the internal dataset.

### Weather
Met Éireann provides historical climate and weather data for Dublin, including public access to historical datasets and open-data resources. The most useful weather variables are temperature, rainfall, wind speed, and severe-weather conditions because they may affect walk-ins and city-centre footfall.[6][7][8]

Recommended weather columns:

- `temp_c`
- `rain_mm`
- `wind_speed`
- `weather_severity_flag`

### Airport and tourism flow
Airport traffic is a useful tourism proxy. Dublin Airport and the CSO publish passenger statistics, and Smart Dublin provides a dataset for Dublin Airport passenger arrivals. These variables are not a replacement for internal pub data, but they can act as demand modifiers, especially on weekends and high-tourism periods.[9][10][11][4]

Recommended airport-related columns:

- `airport_arrivals_proxy`
- `airport_arrivals_prev_day`
- `high_travel_day_flag`

### Cruise traffic
Dublin Port publishes cruise liner information and ship/arrival information, and public cruise schedules can be used to identify cruise days and estimate passenger volumes. Cruise signals are useful because they may create concentrated tourism spikes on certain dates rather than smooth daily demand.[5][12][13]

Recommended cruise-related columns:

- `cruise_ship_flag`
- `cruise_passenger_estimate`
- `ships_in_port_count`

### Calendar and event features
Some features can be generated locally from the date without needing a separate public feed.

Recommended date-derived columns:

- `hour`
- `weekday`
- `is_weekend`
- `month`
- `bank_holiday_flag`
- `payday_period_flag`
- `school_holiday_flag`

Recommended manually maintained event columns:

- `major_sports_event_flag`
- `city_event_flag`
- `promo_flag`
- `private_booking_flag`

## Suggested Dataset Design
The simplest and most useful design is one row per hour. That gives enough detail for kitchen and bar planning without making the dataset too difficult to maintain.

A strong first schema is:

| Category | Example fields |
|---|---|
| Time | `timestamp_hour`, `hour`, `weekday`, `month`, `is_weekend` |
| Demand | `orders_count`, `food_tickets_count`, `sales_total`, `covers_count` |
| Operations | `bar_staff_count`, `kitchen_staff_count`, `stockout_flag`, `busy_label` |
| Venue events | `live_music_flag`, `special_event_flag`, `promo_flag` |
| Weather | `temp_c`, `rain_mm`, `wind_speed` |
| Travel/tourism | `airport_arrivals_proxy`, `cruise_ship_flag`, `cruise_passenger_estimate` |
| Calendar | `bank_holiday_flag`, `payday_period_flag`, `school_holiday_flag` |

## MVP Scope
The first version should stay simple and focus on useful outputs instead of trying to build a perfect system.

Recommended MVP:

1. Build a historical hourly dataset from POS and kitchen records.
2. Add basic date features.
3. Add weather, airport, and cruise signals.
4. Train a simple forecasting model.
5. Output next-day and next-shift demand forecasts.
6. Show results in a very simple dashboard or sheet.

The first release should answer operational questions quickly rather than optimize for technical complexity.

## Modeling Approach
A practical first approach is to compare a simple baseline against a machine-learning model. Public examples and applied restaurant forecasting studies show that structured time features plus external variables can already produce useful demand forecasts.[14][15][3]

Recommended model progression:

- Baseline: same hour last week, moving average, or simple rolling mean.
- ML model: XGBoost, Random Forest, or LightGBM.
- Optional later model: Prophet or SARIMAX for comparison.

The system should also test whether flights and cruise-related features actually improve prediction quality over a model that uses only internal venue data.[2][3]

## Outputs For Staff
The output should be easy for a chef or manager to use in seconds.

Recommended outputs:

- Predicted orders by hour.
- Predicted food tickets by hour.
- Busy risk label for each shift.
- Suggested staffing intensity for lunch/evening.
- Suggested prep signal for key menu items.
- Warning banner when multiple high-demand signals align.

Examples of rules that can later be layered on top of model predictions:

- If rain is low, cruise flag is on, and it is a weekend, raise daytime demand alert.
- If airport arrivals are high before a weekend, raise evening demand expectation.
- If live music and tourism signals align, flag high-risk service pressure.[4][5][1]

## Publicly Available Sources Mentioned
The following public sources are useful for enrichment 

- O'Donoghues Bar website for venue context and hours.[1]
- Met Éireann historical and open weather data.[7][8][6]
- CSO aviation statistics and air/sea travel statistics.[10][9]
- Smart Dublin airport arrivals dataset.[11]
- Dublin Port cruise liner and arrivals information.[12][13]

## Recommended Next Step
The best next step is to create a spreadsheet or CSV with one row per hour and start collecting historical internal data first. Once the internal data exists, public weather, airport, and cruise features can be merged in and tested to see whether they improve forecast accuracy and practical usefulness.[3][6][2]

Sources
[1] O'Donoghues Bar https://odonoghuesbars.ie
[2] Beyond the Cloud https://beyondthecloud.digital/visitor-forecasting-with-machine-learning/
[3] Forecasting daily customer flow in restaurants: a multifactor machine ... https://www.aimspress.com/article/doi/10.3934/aci.2025011?viewType=HTML
[4] Dublin Airport passenger numbers up 7% so far this year - RTE https://www.rte.ie/news/business/2026/0610/1577669-dublin-airport-passenger-numbers/
[5] Dublin, Ireland Port Schedule: Arrivals 2025, 2026 & 2027 https://cruisedig.com/ports/dublin-ireland/arrivals
[6] Historical Data - Met Éireann - The Irish Meteorological Service https://www.met.ie/climate/available-data/historical-data
[7] Available Data - Met Éireann - The Irish Meteorological Service https://www.met.ie/climate/available-data
[8] Met Éireann's new Open Data Portal now live! - Dublin https://www.met.ie/met-eireanns-new-open-data-portal-now-live
[9] Aviation Statistics Quarter 1 2026 - CSO https://www.cso.ie/en/releasesandpublications/ep/p-as/aviationstatisticsquarter12026/
[10] Air and Sea Travel Statistics - CSO https://www.cso.ie/en/methods/surveybackgroundnotes/airandseatravelstatistics/
[11] Data Indicator 9 - Dublin Airport Passenger Arrivals https://data.smartdublin.ie/dataset/dublin-economic-monitor/resource/f6297d1e-2d25-4a35-a499-164f24511dd7
[12] Cruise Liners - Dublin Port https://www.dublinport.ie/cruise-liners/
[13] Arrivals - Dublin Port https://www.dublinport.ie/information-centre/next-100-arrivals/
[14] GitHub - jieunjeon/kaggle-recruit_restaurant_visitor_forecasting: kaggle project submission: recruit_restaurant_visitor_forecasting (time series) https://github.com/jieunjeon/kaggle-recruit_restaurant_visitor_forecasting
[15] A restaurant analytical tool + sales forecasting model · GitHub https://github.com/maks-p/restaurant_sales_forecasting
