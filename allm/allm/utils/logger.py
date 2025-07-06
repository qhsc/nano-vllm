import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)
