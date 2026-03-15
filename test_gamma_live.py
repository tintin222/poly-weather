"""Quick integration test for the Gamma API client."""

import asyncio
from src.polymarket.gamma_api import get_temperature_events, parse_city_and_date, parse_brackets


async def main():
    events = await get_temperature_events()
    print(f"Found {len(events)} temperature events")
    for e in events[:5]:
        print(f"  {e.title}")
        parsed = parse_city_and_date(e)
        if parsed:
            city, date = parsed
            print(f"    -> City: {city}, Date: {date}")
        brackets = parse_brackets(e)
        for b in brackets[:3]:
            print(f"    -> Bracket: {b.label} | Yes: {b.yes_price} | No: {b.no_price}")

if __name__ == "__main__":
    asyncio.run(main())
