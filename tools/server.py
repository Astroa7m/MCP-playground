from typing import Any, List

import httpx
from fastmcp import FastMCP



# init mcp server
mcp = FastMCP("my_cool_tools")
USER_AGENT = "my-tools-app"

async def make_request(url: str, USER_AGENT) -> dict[str, Any] | None:
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

@mcp.tool()
async def get_subreddit_news(subreddit: str, limit: int = 5)-> List[dict]:
    """Gets by default hot 5 posts from subreddits param
    :param subreddit: the subredit to look for, e.g. worldnews, tech, news, etc.
    :return: a list of dict features posts properties
    """
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"

    response = await make_request(url, USER_AGENT)

    if not response or "data" not in response:
        return "Unable to fetch subreddit data."

    posts = []

    for post in response["data"]["children"]:
        d = post["data"]
        posts.append({
            "title": d["title"],
            "url": d["url"],
            "permalink": f"https://www.reddit.com{d['permalink']}",
            "subreddit": d["subreddit"],
            "created_utc": d["created_utc"],
            "ups": d["ups"],
            "num_comments": d["num_comments"],
            "thumbnail": d.get("thumbnail") if d.get("thumbnail", "").startswith("http") else None
        })

    return posts
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

    data = await make_request(url, USER_AGENT)
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




# rename weather to tools and all all the functions insdie


if __name__ == "__main__":
    # run the server
    mcp.run(transport='stdio')
