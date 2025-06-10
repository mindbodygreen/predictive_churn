from snowflake.snowpark import Session
from typing import Dict, Optional

class SnowflakeConnector:
    def __init__(self):
        self._session = None

    def get_connection_parameters(self) -> Dict[str, str]:
        """Get Snowflake connection parameters."""
        return {
            "account": "zfb96811.us-east-1",
            "role": "ACCOUNTADMIN",
            "database": "ANALYTICS",
            "schema": "PUBLIC",
            "warehouse": "COMPUTE_WH",
            "user": "IMAHAKUMARA",
            "password": "gEfnun-dewnin-0nadpy"
        }

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

def get_snowflake_session() -> Session:
    """Helper function to get Snowflake session."""
    return snowflake_connector.get_session()

def close_snowflake_session() -> None:
    """Helper function to close Snowflake session."""
    snowflake_connector.close_session()
