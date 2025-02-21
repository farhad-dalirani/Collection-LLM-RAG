import logging
from application import launch_app

if __name__ == '__main__':
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging is configured.')

    # Launch the web-based GUI (Demo Mode)
    # This runs the application with limited functionality.
    # To enable full capabilities, including creating or deleting query engines, 
    # run `main.py` or set `enable_query_engine_management=True`.
    launch_app(enable_query_engine_management=False)