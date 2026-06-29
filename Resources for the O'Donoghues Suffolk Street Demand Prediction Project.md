# Resources for the O'Donoghues Suffolk Street Demand Prediction Project

## Overview
This document lists GitHub repositories and research sources that are useful for building a restaurant, bar, or pub demand forecasting system. The focus is on resources that can help with feature engineering, time-series forecasting, machine-learning model comparison, weather/calendar enrichment, and practical restaurant operations use cases.[1][2][3]

## Best GitHub Repositories

### 1. MaxHalford/kaggle-recruit-restaurant
Link: https://github.com/MaxHalford/kaggle-recruit-restaurant  
Why it matters: This is an 8th-place Kaggle solution for restaurant visitor forecasting and is one of the strongest examples for feature engineering and structured restaurant demand prediction workflows.[4]

Useful takeaways:
- Strong feature engineering ideas.
- Good reference for LightGBM-based forecasting.
- Good example of turning reservation and visitation history into model inputs.[4]

### 2. jieunjeon/kaggle-recruit_restaurant_visitor_forecasting
Link: https://github.com/jieunjeon/kaggle-recruit_restaurant_visitor_forecasting  
Why it matters: A practical notebook-style implementation for restaurant visitor forecasting using time-series ideas and useful merged features.[1]

Useful takeaways:
- Rolling features and temporal features.
- Joining different sources of restaurant data.
- Useful structure for experimentation and notebook-based prototyping.[1]

### 3. michellekli/visitor-forecasting
Link: https://github.com/michellekli/visitor-forecasting  
Why it matters: Helpful for classical forecasting baselines such as ARIMA, SARIMAX, and BSTS.[5]

Useful takeaways:
- How to build baseline statistical models.
- How to compare classical time-series approaches.
- Useful if an explainable baseline is needed before moving to XGBoost or LightGBM.[5]

### 4. Architectshwet/Recruit-Restaurant-Visitor-Forecasting-
Link: https://github.com/Architectshwet/Recruit-Restaurant-Visitor-Forecasting-  
Why it matters: Useful because it explores Prophet, ARIMA, and H2O AutoML-style forecasting in the restaurant visitor context.[6]

Useful takeaways:
- Model comparison ideas.
- Prophet-based workflow reference.
- Useful for benchmarking quick forecasting pipelines.[6]

### 5. rashijain/Recruit-Restaurant-Visitor-Forecasting
Link: https://github.com/rashijain/Recruit-Restaurant-Visitor-Forecasting  
Why it matters: This repo compares multiple model families including linear models, ARIMA variants, LightGBM, and RNNs for restaurant customer prediction.[7]

Useful takeaways:
- Broad model comparison ideas.
- Helps identify what is worth trying and what may be overkill.
- Good reference for experimentation strategy.[7]

### 6. maks-p/restaurant_sales_forecasting
Link: https://github.com/maks-p/restaurant_sales_forecasting  
Why it matters: This is closer to a restaurant analytics tool than a pure competition notebook, so it is useful for thinking about a deployable end-to-end system.[3]

Useful takeaways:
- Restaurant analytics framing.
- End-to-end workflow ideas.
- Better alignment with a real venue operations tool.[3]

### 7. ericyaang/forecasting-sales-for-a-restaurant
Link: https://github.com/ericyaang/forecasting-sales-for-a-restaurant  
Why it matters: A reproducible project focused on daily restaurant sales prediction using supervised machine learning.[8]

Useful takeaways:
- Clean supervised ML framing.
- Useful if the project target becomes sales instead of footfall.
- Good reference for daily aggregation logic.[8]

### 8. shubhammehra11/Demand-Forecast
Link: https://github.com/shubhammehra11/Demand-Forecast  
Why it matters: Useful for food demand forecasting and minimizing food waste with XGBoost and Random Forest.[9]

Useful takeaways:
- Inventory and waste-reduction angle.
- Relevant for prep planning in the kitchen.
- Helpful if the project later predicts item-level demand.[9]

## Most Relevant Research

### 1. Forecasting daily customer flow in restaurants: a multifactor machine learning approach
Why it matters: This is the most directly relevant paper found for the project. It studies daily restaurant customer flow using multiple machine-learning techniques and feature sets, including temporal variables, weather, holidays, and menu-related information, and reports XGBoost as the best model in that case study.[2]

Useful takeaways:
- Strong problem framing for restaurant customer-flow forecasting.
- Clear example of comparing multiple models.
- Good reference for weather and holiday features.
- Important note that preprocessing and missing-data handling can strongly affect results.[2]

### 2. Review of forecasting studies for the restaurant industry
Why it matters: This is useful for understanding the broader landscape of restaurant forecasting approaches and where your project fits in practice.[10]

Useful takeaways:
- Overview of methods used in restaurant forecasting.
- Good reference for positioning the problem.
- Useful for identifying common data sources and forecasting targets.[10]

### 3. Demand forecasting in restaurants using machine learning and mathematical models
Why it matters: This source is useful because it covers restaurant demand forecasting with both statistical and machine-learning approaches.[11]

Useful takeaways:
- Good reference for comparing traditional forecasting and ML methods.
- Reinforces the value of exogenous variables such as weather and holidays.[11]

### 4. Forecasting daily customer flow in restaurants
Why it matters: A relevant paper specifically focused on forecasting daily customer numbers in a restaurant setting.[12]

Useful takeaways:
- Helpful for feature selection ideas.
- Good example of restaurant demand framed as a forecasting problem.[12]

## How These Resources Help the O'Donoghues Project
The project for O'Donoghues on Suffolk Street is not just a generic forecasting problem. It is an operations problem where the kitchen and bar need useful signals for prep, staffing, and rush planning. That means the most helpful resources are not necessarily the ones with the most advanced models, but the ones that show how to:[13]

- Build demand datasets from historical restaurant activity.
- Engineer lag, rolling, and calendar features.
- Merge external signals like weather and tourism proxies.
- Evaluate models using time-aware validation.
- Translate forecasts into practical actions.[2][3][1]

## Best Learning Path
A good order for using these resources is:

1. Start with the 2025 restaurant customer-flow paper for a clear real-world framing.[2]
2. Study MaxHalford and jieunjeon for feature engineering and restaurant forecasting workflow ideas.[4][1]
3. Use michellekli for classical baselines like SARIMAX.[5]
4. Use maks-p for thinking about a usable analytics or dashboard system instead of only a notebook.[3]
5. Use shubhammehra11 if the project expands toward item-level demand and waste reduction.[9]

## What To Reuse vs What To Avoid
The most reusable parts of these resources are:

- Temporal feature engineering.
- Weather/calendar merging.
- Time-series validation setup.
- Restaurant demand target design.
- Baseline model construction.[1][4][5][2]

The parts that should not be copied blindly are:

- Kaggle-specific leaderboard tricks.
- Overly complex neural models without enough local data.
- Generic metrics without linking them to kitchen or bar decisions.

The O'Donoghues project should stay focused on real operational usefulness rather than competition-style optimization.[13][3]

## Shortlist To Prioritize
If time is limited, prioritize these four first:

1. MaxHalford/kaggle-recruit-restaurant.[4]
2. jieunjeon/kaggle-recruit_restaurant_visitor_forecasting.[1]
3. michellekli/visitor-forecasting.[5]
4. Forecasting daily customer flow in restaurants: a multifactor machine learning approach.[2]

Sources
[1] jieunjeon/kaggle-recruit_restaurant_visitor_forecasting - GitHub https://github.com/jieunjeon/kaggle-recruit_restaurant_visitor_forecasting
[2] Forecasting daily customer flow in restaurants: a multifactor machine ... https://www.aimspress.com/article/doi/10.3934/aci.2025011?viewType=HTML
[3] A restaurant analytical tool + sales forecasting model · GitHub https://github.com/maks-p/restaurant_sales_forecasting
[4] GitHub - MaxHalford/kaggle-recruit-restaurant: :trophy: Kaggle 8th place solution https://github.com/MaxHalford/kaggle-recruit-restaurant
[5] GitHub - michellekli/visitor-forecasting: Time series analysis and forecasting with ARIMA, SARIMAX, and BSTS. https://github.com/michellekli/visitor-forecasting
[6] GitHub - Architectshwet/Recruit-Restaurant-Visitor-Forecasting https://github.com/Architectshwet/Recruit-Restaurant-Visitor-Forecasting-
[7] GitHub - rashijain/Recruit-Restaurant-Visitor-Forecasting https://github.com/rashijain/Recruit-Restaurant-Visitor-Forecasting
[8] GitHub - ericyaang/forecasting-sales-for-a-restaurant: This reproducible project predicts daily restaurant sales using supervised machine learning techniques. https://github.com/ericyaang/forecasting-sales-for-a-restaurant
[9] In PySpark: Demand forecasting using XGboost and ... - GitHub https://github.com/shubhammehra11/Demand-Forecast
[10] [PDF] A review of forecasting studies for the restaurant industry - EconStor https://www.econstor.eu/bitstream/10419/305845/1/id478.pdf
[11] Demand forecasting in restaurants using machine learning and ... https://www.sciencedirect.com/science/article/pii/S2212827119301568
[12] Forecasting daily customer flow in restaurants https://www.aimspress.com/data/article/preview/pdf/68a6f025ba35de58686da43e.pdf
[13] O'Donoghues Bar https://odonoghuesbars.ie
