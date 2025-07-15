#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configure Python path for tests to find project modules.
This file is automatically loaded by pytest.
"""

import os
import sys

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
