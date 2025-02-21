import logging
from application import launch_app

if __name__ == '__main__':
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging is configured.')

    # Launch the web-based GUI
    launch_app()