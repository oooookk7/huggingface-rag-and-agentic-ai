try:
    from loguru import logger  # type: ignore

    logger.add(
        "app.log",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
except ImportError:
    import logging

    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("docchat")
