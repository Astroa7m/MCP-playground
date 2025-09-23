from typing import Any
import httpx
from fastmcp import FastMCP
import asyncio

# init mcp server
mcp = FastMCP("weather")
USER_AGENT = "my-weather-app"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the weather API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except:
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]

    return f"""
    Event: {props.get('event', 'Unknown')}
    Area: {props.get('areaDesc', 'Unknown')}
    Severity: {props.get('severity', 'Unknown')}
    Description: {props.get('description', 'No description available')}
    Instructions: {props.get('instruction', 'No specific instructions provided')}
    """

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get current weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m,wind_speed_10m,weather_code"
        f"&timezone=auto"
    )

    data = await make_nws_request(url)
    if not data or "current" not in data:
        return "Unable to fetch forecast data for this location."

    current = data["current"]
    units = data.get("current_units", {})

    temperature = current.get("temperature_2m", "N/A")
    wind_speed = current.get("wind_speed_10m", "N/A")
    weather_code = current.get("weather_code", "N/A")

    return (
        f"🌡️ Temperature: {temperature} {units.get('temperature_2m', '°C')}\n"
        f"💨 Wind Speed: {wind_speed} {units.get('wind_speed_10m', 'km/h')}\n"
        f"🌦️ Weather Code: {weather_code}"
    )




if __name__ == "__main__":
    # run the server
    mcp.run(transport='stdio')
