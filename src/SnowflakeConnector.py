import os
import yaml
from snowflake.snowpark import Session
from typing import Dict, Optional


class SnowflakeConnector:
    def __init__(self):
        self._session = None
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'snowflake.yml')
        self.config_path = os.path.abspath(self.config_path)

    def get_connection_parameters(self) -> Dict[str, str]:
        """Get Snowflake connection parameters from YAML file."""
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        return config.get("snowflake", {})

    def get_session(self) -> Session:
        """Get or create Snowflake session."""
        if self._session is None:
            connection_parameters = self.get_connection_parameters()
            self._session = Session.builder.configs(connection_parameters).create()
        return self._session

    def close_session(self) -> None:
        """Close the Snowflake session if it exists."""
        if self._session is not None:
            self._session.close()
            self._session = None

# Create a singleton instance
snowflake_connector = SnowflakeConnector()

def create_active_session() -> Session:
    """Helper function to get Snowflake session."""
    snowflake_connector = SnowflakeConnector()
    return snowflake_connector.get_session()

def close_snowflake_session() -> None:
    """Helper function to close Snowflake session."""
    snowflake_connector.close_session()
