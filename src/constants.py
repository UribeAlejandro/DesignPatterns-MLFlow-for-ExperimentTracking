#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------- #
# Created By: Uribe
# Created Date: 28/4/23
# --------------------------------------------------------------------------- #
# ** Description **

"""

"""

# --------------------------------------------------------------------------- #
# ** Required libraries **

from pathlib import Path

DATASET_ID = "higgs"
DATA_LOCATION = "data"
OUTPUT_PATH = Path(DATA_LOCATION, DATASET_ID).with_suffix(".parquet")
