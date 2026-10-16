import os
from app import create_app
from app.config import get_config

config_name = os.getenv('APP_ENV', os.getenv('FLASK_ENV', 'development'))
app = create_app(config_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
