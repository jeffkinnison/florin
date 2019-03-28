"""Common logger for FLoRIN messages.

Functions
---------
debug
info
error
warning
"""
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%y %I:%M:%S')
florin_logger = logging.logger('florin')
