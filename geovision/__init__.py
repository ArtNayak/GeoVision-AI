"""
GeoVision AI
============

A cloud-based, near real-time environmental intelligence and forecasting
platform. This package implements the 5-layer system architecture and the
8-stage methodology described in the project report:

Architecture layers
--------------------
1. Data-source layer      -> geovision.ingestion (Landsat/Sentinel/MODIS via
                              Microsoft Planetary Computer, ERA5, NASA POWER
                              precipitation, OpenAQ)
2. Ingestion/processing    -> geovision.ingestion.pipeline, geovision.preprocessing
3. ML/DL analytics engine  -> geovision.features, geovision.models, geovision.index
4. Application-services    -> geovision.alerting, geovision.index (comparisons)
5. Presentation layer      -> geovision.api (serves the web dashboard in /dashboard)

Methodology stages (see docs/report)
-------------------------------------
(1) near real-time ingestion -> ingestion/*
(2) preprocessing (cloud masking, correction) -> preprocessing/*
(3) feature engineering (spectral indices, PCA, normalization) -> features/*
(4) parallel ML/DL training -> models/classifiers.py, models/forecasting.py
(5) validation -> models/evaluate.py
(6) fusion into Environmental Health Index -> index/health_index.py
(7) interactive dashboard -> api/ + dashboard/
(8) near real-time deployment + alerting -> alerting/*
"""

__version__ = "0.1.0"
