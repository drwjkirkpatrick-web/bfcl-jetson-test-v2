#!/usr/bin/env python3
"""
BFCL v2 Jetson Test — Test Definitions (data only).

Categories:
  - SIMPLE (30):            single tool call, tricky argument extraction
                            (dates, times, units, booleans, arrays) — BFCL
                            "non-live simple" style.
  - MODERATE_PARALLEL (12): 2-3 independent calls in one response —
                            BFCL "multiple" style.
  - MODERATE_CHAINED (8):   multi-step: harness executes the step-1 call
                            with a deterministic mock, feeds the result back
                            as a tool message, model must derive step-2
                            arguments FROM THE RESULT (incl. conditionals) —
                            BFCL V3 multi-turn/multi-step style.

Evaluation rules (in bfcl_v2_harness.py) follow BFCL AST-style checks:
strict on required parameters, lenient on extras, string normalization,
int/float coercion, order-independent matching for parallel calls.

All dates are ISO 8601 (YYYY-MM-DD), times 24-hour (HH:MM) — stated in the
tool descriptions so expectations are fair.
"""

# ---------------------------------------------------------------------------
# Tool schema helper
# ---------------------------------------------------------------------------

def fn(name: str, desc: str, params: dict, required: list | None = None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required if required is not None else list(params.keys()),
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool registry — every tool used by any test
# ---------------------------------------------------------------------------

DATE = {"type": "string", "description": "Date in ISO 8601 format YYYY-MM-DD"}
TIME = {"type": "string", "description": "Time in 24-hour format HH:MM"}

TOOLS = {
    # weather / conversion
    "get_weather": fn("get_weather", "Get the current weather conditions for a city.",
                      {"location": {"type": "string", "description": "City name"}}),
    "convert_temperature": fn("convert_temperature", "Convert a temperature between units.",
                              {"value": {"type": "number", "description": "Temperature value"},
                               "from_unit": {"type": "string", "description": "celsius or fahrenheit"},
                               "to_unit": {"type": "string", "description": "celsius or fahrenheit"}}),
    # finance
    "convert_currency": fn("convert_currency", "Convert an amount between currencies.",
                           {"amount": {"type": "number"},
                            "from_currency": {"type": "string", "description": "ISO currency code, e.g. USD"},
                            "to_currency": {"type": "string", "description": "ISO currency code, e.g. EUR"}}),
    "get_stock_price": fn("get_stock_price", "Get the current stock price for a ticker symbol.",
                          {"symbol": {"type": "string"}}),
    "place_order": fn("place_order", "Place a stock order at a given price.",
                      {"symbol": {"type": "string"},
                       "quantity": {"type": "integer"},
                       "price": {"type": "number", "description": "Price per share"}}),
    "calculate_mortgage": fn("calculate_mortgage", "Monthly mortgage payment.",
                             {"principal": {"type": "number", "description": "Loan amount in dollars"},
                              "annual_rate": {"type": "number", "description": "Annual interest rate in percent"},
                              "years": {"type": "integer"}}),
    "calculate_tip": fn("calculate_tip", "Calculate a tip on a bill.",
                        {"bill_amount": {"type": "number"},
                         "tip_percentage": {"type": "number", "description": "Tip in percent"}}),
    "calculate_fuel_cost": fn("calculate_fuel_cost", "Fuel cost for a trip.",
                              {"distance_miles": {"type": "number"},
                               "mpg": {"type": "number", "description": "Miles per gallon"},
                               "price_per_gallon": {"type": "number"}}),
    "calculate_cart_total": fn("calculate_cart_total", "Total the items in a shopping cart.",
                               {"cart_id": {"type": "string"}}),
    "apply_discount": fn("apply_discount", "Apply a coupon to an order total.",
                         {"order_total": {"type": "number"},
                          "coupon_code": {"type": "string"}}),
    # calendar / alarms
    "create_calendar_event": fn("create_calendar_event", "Create a calendar event.",
                                {"title": {"type": "string"}, "date": DATE, "time": TIME}),
    "set_alarm": fn("set_alarm", "Set an alarm.",
                    {"time": TIME, "label": {"type": "string"}}),
    # communications
    "send_email": fn("send_email", "Send an email.",
                     {"to": {"type": "string", "description": "Recipient email address"},
                      "subject": {"type": "string"}, "body": {"type": "string"}}),
    "send_text": fn("send_text", "Send an SMS text message.",
                    {"to": {"type": "string", "description": "Phone number"},
                     "message": {"type": "string"}}),
    # travel
    "search_flights": fn("search_flights", "Search flights between two airports.",
                         {"origin": {"type": "string", "description": "Airport code"},
                          "destination": {"type": "string", "description": "Airport code"},
                          "date": DATE}),
    "book_flight": fn("book_flight", "Book a flight by its flight ID.",
                      {"flight_id": {"type": "string"},
                       "passenger": {"type": "string", "description": "Passenger name"}}),
    "get_driving_distance": fn("get_driving_distance", "Driving distance between two cities.",
                               {"origin": {"type": "string"}, "destination": {"type": "string"}}),
    "book_hotel": fn("book_hotel", "Book a hotel room.",
                     {"city": {"type": "string"}, "check_in": DATE, "nights": {"type": "integer"}}),
    # music / media
    "play_song": fn("play_song", "Play a song.",
                    {"title": {"type": "string"}, "artist": {"type": "string"}}),
    "get_movie_showtimes": fn("get_movie_showtimes", "Showtimes for a movie at a theater.",
                              {"movie": {"type": "string"}, "theater": {"type": "string"}}),
    # food
    "find_restaurant": fn("find_restaurant", "Find restaurants matching cuisine and price.",
                          {"cuisine": {"type": "string"}, "location": {"type": "string"},
                           "max_price_per_person": {"type": "number"}}),
    "book_table": fn("book_table", "Reserve a table at a restaurant by its ID.",
                     {"restaurant_id": {"type": "string"}, "party_size": {"type": "integer"}, "time": TIME}),
    "get_recipe": fn("get_recipe", "Find a recipe that uses given ingredients.",
                     {"ingredients": {"type": "array", "items": {"type": "string"}}}),
    # shopping / productivity
    "add_to_shopping_list": fn("add_to_shopping_list", "Add an item to the shopping list.",
                               {"item": {"type": "string"}, "quantity": {"type": "integer"}}),
    "create_purchase_order": fn("create_purchase_order", "Create a restocking purchase order.",
                                {"item": {"type": "string"}, "quantity": {"type": "integer"}}),
    "create_todo": fn("create_todo", "Create a todo item.",
                      {"task": {"type": "string"}, "due_date": DATE,
                       "priority": {"type": "string", "description": "low, medium, or high"}}),
    "generate_password": fn("generate_password", "Generate a random password.",
                            {"length": {"type": "integer"},
                             "include_symbols": {"type": "boolean"}}),
    "random_number": fn("random_number", "Generate a random integer in an inclusive range.",
                        {"min": {"type": "integer"}, "max": {"type": "integer"}}),
    # home automation
    "set_thermostat": fn("set_thermostat", "Set a thermostat for a room or zone.",
                         {"room": {"type": "string"}, "temperature": {"type": "number",
                                          "description": "Target in degrees Celsius"}}),
    "control_smart_light": fn("control_smart_light", "Turn a smart light on or off.",
                              {"room": {"type": "string"}, "state": {"type": "string",
                                                  "description": "on or off"}}),
    # health
    "calculate_bmi": fn("calculate_bmi", "Body mass index.",
                       {"weight_kg": {"type": "number"}, "height_m": {"type": "number"}}),
    "log_workout": fn("log_workout", "Log a workout session.",
                      {"activity": {"type": "string"}, "duration_minutes": {"type": "integer"},
                       "calories": {"type": "integer"}}),
    # data / info
    "get_customer": fn("get_customer", "Look up a customer by ID.",
                       {"customer_id": {"type": "integer"}}),
    "search_inventory": fn("search_inventory", "Check inventory stock for an item.",
                           {"item": {"type": "string"}}),
    "query_sales": fn("query_sales", "Total sales for a region and quarter.",
                      {"region": {"type": "string"}, "quarter": {"type": "string",
                                       "description": "e.g. Q1, Q2, Q3, Q4"}}),
    "track_package": fn("track_package", "Track a package by tracking ID.",
                        {"tracking_id": {"type": "string"}}),
    "get_definition": fn("get_definition", "Get the dictionary definition of a word.",
                         {"word": {"type": "string"}}),
    "translate_text": fn("translate_text", "Translate text into a target language.",
                         {"text": {"type": "string"}, "target_language": {"type": "string"}}),
    "get_sunrise_sunset": fn("get_sunrise_sunset", "Sunrise and sunset times for a location and date.",
                             {"location": {"type": "string"}, "date": DATE}),
    "get_user": fn("get_user", "Look up a user profile by user ID.",
                   {"user_id": {"type": "integer"}}),
}


def T(*names):
    """Build a tools list from tool-registry keys."""
    return [TOOLS[n] for n in names]


# ---------------------------------------------------------------------------
# SIMPLE tests — 30 single-call cases
# Each: id, question, tools, expected calls.
# ---------------------------------------------------------------------------

SIMPLE_TESTS = [
    {"id": "simple_01", "question": "What's the current weather in Tokyo?",
     "tools": T("get_weather"), "expected": [{"name": "get_weather", "args": {"location": "Tokyo"}}]},
    {"id": "simple_02", "question": "Convert 250 US dollars into euros.",
     "tools": T("convert_currency"),
     "expected": [{"name": "convert_currency", "args": {"amount": 250, "from_currency": "USD", "to_currency": "EUR"}}]},
    {"id": "simple_03", "question": "What's the monthly mortgage payment on a $320,000 loan at 4.2% annual interest over 30 years?",
     "tools": T("calculate_mortgage"),
     "expected": [{"name": "calculate_mortgage", "args": {"principal": 320000, "annual_rate": 4.2, "years": 30}}]},
    {"id": "simple_04", "question": "Set my alarm for 7:30 AM with the label 'workout'.",
     "tools": T("set_alarm"),
     "expected": [{"name": "set_alarm", "args": {"time": "07:30", "label": "workout"}}]},
    {"id": "simple_05", "question": "Send an email to maria@example.com with the subject 'Q3 Invoice' and the body 'Hi Maria, please find the Q3 invoice attached.'",
     "tools": T("send_email"),
     "expected": [{"name": "send_email", "args": {"to": "maria@example.com", "subject": "Q3 Invoice",
                    "body": "Hi Maria, please find the Q3 invoice attached."}}]},
    {"id": "simple_06", "question": "How far is the drive from Seattle to Portland?",
     "tools": T("get_driving_distance"),
     "expected": [{"name": "get_driving_distance", "args": {"origin": "Seattle", "destination": "Portland"}}]},
    {"id": "simple_07", "question": "Play Bohemian Rhapsody by Queen.",
     "tools": T("play_song"),
     "expected": [{"name": "play_song", "args": {"title": "Bohemian Rhapsody", "artist": "Queen"}}]},
    {"id": "simple_08", "question": "What's Apple's stock price right now? Ticker AAPL.",
     "tools": T("get_stock_price"),
     "expected": [{"name": "get_stock_price", "args": {"symbol": "AAPL"}}]},
    {"id": "simple_09", "question": "I weigh 70 kilograms and I'm 1.75 meters tall. What's my BMI?",
     "tools": T("calculate_bmi"),
     "expected": [{"name": "calculate_bmi", "args": {"weight_kg": 70, "height_m": 1.75}}]},
    {"id": "simple_10", "question": "Translate 'Good morning' into Spanish.",
     "tools": T("translate_text"),
     "expected": [{"name": "translate_text", "args": {"text": "Good morning", "target_language": "Spanish"}}]},
    {"id": "simple_11", "question": "Create a calendar event called 'Dentist appointment' at 2 PM on October 12, 2026.",
     "tools": T("create_calendar_event"),
     "expected": [{"name": "create_calendar_event", "args": {"title": "Dentist appointment",
                    "date": "2026-10-12", "time": "14:00"}}]},
    {"id": "simple_12", "question": "Find me a Thai restaurant in Portland that costs under $20 per person.",
     "tools": T("find_restaurant"),
     "expected": [{"name": "find_restaurant", "args": {"cuisine": "Thai", "location": "Portland",
                    "max_price_per_person": 20}}]},
    {"id": "simple_13", "question": "Pick a random number between 1 and 100.",
     "tools": T("random_number"),
     "expected": [{"name": "random_number", "args": {"min": 1, "max": 100}}]},
    {"id": "simple_14", "question": "Generate a password that is 16 characters long and includes symbols.",
     "tools": T("generate_password"),
     "expected": [{"name": "generate_password", "args": {"length": 16, "include_symbols": True}}]},
    {"id": "simple_15", "question": "Track package 1Z999AA10123456784 for me.",
     "tools": T("track_package"),
     "expected": [{"name": "track_package", "args": {"tracking_id": "1Z999AA10123456784"}}]},
    {"id": "simple_16", "question": "What's an 18% tip on an $85.40 bill?",
     "tools": T("calculate_tip"),
     "expected": [{"name": "calculate_tip", "args": {"bill_amount": 85.40, "tip_percentage": 18}}]},
    {"id": "simple_17", "question": "Set the living room thermostat to 22 degrees Celsius.",
     "tools": T("set_thermostat"),
     "expected": [{"name": "set_thermostat", "args": {"room": "living room", "temperature": 22}}]},
    {"id": "simple_18", "question": "Log my 45-minute run that burned 430 calories.",
     "tools": T("log_workout"),
     "expected": [{"name": "log_workout", "args": {"activity": "running", "duration_minutes": 45, "calories": 430}}]},
    {"id": "simple_19", "question": "Look up the customer with ID 4821.",
     "tools": T("get_customer"),
     "expected": [{"name": "get_customer", "args": {"customer_id": 4821}}]},
    {"id": "simple_20", "question": "Add two cartons of milk to my shopping list.",
     "tools": T("add_to_shopping_list"),
     "expected": [{"name": "add_to_shopping_list", "args": {"item": "milk", "quantity": 2}}]},
    {"id": "simple_21", "question": "When is Dune: Part Two showing at the downtown theater?",
     "tools": T("get_movie_showtimes"),
     "expected": [{"name": "get_movie_showtimes", "args": {"movie": "Dune: Part Two", "theater": "downtown"}}]},
    {"id": "simple_22", "question": "What does the word 'ephemeral' mean?",
     "tools": T("get_definition"),
     "expected": [{"name": "get_definition", "args": {"word": "ephemeral"}}]},
    {"id": "simple_23", "question": "Add a high-priority todo to buy a birthday cake, due on October 2, 2026.",
     "tools": T("create_todo"),
     "expected": [{"name": "create_todo", "args": {"task": "buy a birthday cake",
                    "due_date": "2026-10-02", "priority": "high"}}]},
    {"id": "simple_24", "question": "What are the sunrise and sunset times in Reykjavik on June 21, 2026?",
     "tools": T("get_sunrise_sunset"),
     "expected": [{"name": "get_sunrise_sunset", "args": {"location": "Reykjavik", "date": "2026-06-21"}}]},
    {"id": "simple_25", "question": "Turn off the bedroom light.",
     "tools": T("control_smart_light"),
     "expected": [{"name": "control_smart_light", "args": {"room": "bedroom", "state": "off"}}]},
    {"id": "simple_26", "question": "Book a hotel in Barcelona, checking in on November 20, 2026, for 4 nights.",
     "tools": T("book_hotel"),
     "expected": [{"name": "book_hotel", "args": {"city": "Barcelona", "check_in": "2026-11-20", "nights": 4}}]},
    {"id": "simple_27", "question": "Find a recipe that uses chicken, rice, and broccoli.",
     "tools": T("get_recipe"),
     "expected": [{"name": "get_recipe", "args": {"ingredients": ["chicken", "rice", "broccoli"]}}]},
    {"id": "simple_28", "question": "Send a text to 555-0142 saying 'Running late, start without me'.",
     "tools": T("send_text"),
     "expected": [{"name": "send_text", "args": {"to": "555-0142", "message": "Running late, start without me"}}]},
    {"id": "simple_29", "question": "What's the fuel cost for a 450-mile trip if my car gets 32 miles per gallon and gas is $3.85 a gallon?",
     "tools": T("calculate_fuel_cost"),
     "expected": [{"name": "calculate_fuel_cost", "args": {"distance_miles": 450, "mpg": 32,
                    "price_per_gallon": 3.85}}]},
    {"id": "simple_30", "question": "Show me total sales for the west region in Q3.",
     "tools": T("query_sales"),
     "expected": [{"name": "query_sales", "args": {"region": "west", "quarter": "Q3"}}]},
]


# ---------------------------------------------------------------------------
# MODERATE PARALLEL — 12 multi-call cases (order-independent matching)
# ---------------------------------------------------------------------------

MODERATE_PARALLEL = [
    {"id": "par_01", "question": "What's the weather like in Paris and Tokyo right now?",
     "tools": T("get_weather"),
     "expected": [{"name": "get_weather", "args": {"location": "Paris"}},
                   {"name": "get_weather", "args": {"location": "Tokyo"}}]},
    {"id": "par_02", "question": "Convert 100 US dollars to euros, and also 100 US dollars to Japanese yen.",
     "tools": T("convert_currency"),
     "expected": [{"name": "convert_currency", "args": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}},
                   {"name": "convert_currency", "args": {"amount": 100, "from_currency": "USD", "to_currency": "JPY"}}]},
    {"id": "par_03", "question": "Get the current stock prices for Apple, Microsoft, and Tesla — tickers AAPL, MSFT, and TSLA.",
     "tools": T("get_stock_price"),
     "expected": [{"name": "get_stock_price", "args": {"symbol": "AAPL"}},
                   {"name": "get_stock_price", "args": {"symbol": "MSFT"}},
                   {"name": "get_stock_price", "args": {"symbol": "TSLA"}}]},
    {"id": "par_04", "question": "Create two calendar events on October 9, 2026: 'Lunch with Sam' at 12:00 and 'Gym session' at 18:00.",
     "tools": T("create_calendar_event"),
     "expected": [{"name": "create_calendar_event", "args": {"title": "Lunch with Sam", "date": "2026-10-09", "time": "12:00"}},
                   {"name": "create_calendar_event", "args": {"title": "Gym session", "date": "2026-10-09", "time": "18:00"}}]},
    {"id": "par_05", "question": "Send a text to 555-0101 saying 'Love you!' and a text to 555-0303 saying 'Happy birthday!'",
     "tools": T("send_text"),
     "expected": [{"name": "send_text", "args": {"to": "555-0101", "message": "Love you!"}},
                   {"name": "send_text", "args": {"to": "555-0303", "message": "Happy birthday!"}}]},
    {"id": "par_06", "question": "Set the bedroom thermostat to 21 degrees Celsius and turn the kitchen light off.",
     "tools": T("set_thermostat", "control_smart_light"),
     "expected": [{"name": "set_thermostat", "args": {"room": "bedroom", "temperature": 21}},
                   {"name": "control_smart_light", "args": {"room": "kitchen", "state": "off"}}]},
    {"id": "par_07", "question": "Get showtimes for Dune: Part Two at the downtown theater and Interstellar at the mall theater.",
     "tools": T("get_movie_showtimes"),
     "expected": [{"name": "get_movie_showtimes", "args": {"movie": "Dune: Part Two", "theater": "downtown"}},
                   {"name": "get_movie_showtimes", "args": {"movie": "Interstellar", "theater": "mall"}}]},
    {"id": "par_08", "question": "Add milk, eggs, and bread to my shopping list.",
     "tools": T("add_to_shopping_list"),
     "expected": [{"name": "add_to_shopping_list", "args": {"item": "milk", "quantity": 1}},
                   {"name": "add_to_shopping_list", "args": {"item": "eggs", "quantity": 1}},
                   {"name": "add_to_shopping_list", "args": {"item": "bread", "quantity": 1}}]},
    {"id": "par_09", "question": "Play Bohemian Rhapsody by Queen, then queue up Hotel California by Eagles.",
     "tools": T("play_song"),
     "expected": [{"name": "play_song", "args": {"title": "Bohemian Rhapsody", "artist": "Queen"}},
                   {"name": "play_song", "args": {"title": "Hotel California", "artist": "Eagles"}}]},
    {"id": "par_10", "question": "Compute BMI for someone who is 70 kg and 1.75 m tall, and for someone who is 85 kg and 1.80 m tall.",
     "tools": T("calculate_bmi"),
     "expected": [{"name": "calculate_bmi", "args": {"weight_kg": 70, "height_m": 1.75}},
                   {"name": "calculate_bmi", "args": {"weight_kg": 85, "height_m": 1.80}}]},
    {"id": "par_11", "question": "What's the driving distance from Seattle to Portland, and from Portland to Vancouver?",
     "tools": T("get_driving_distance"),
     "expected": [{"name": "get_driving_distance", "args": {"origin": "Seattle", "destination": "Portland"}},
                   {"name": "get_driving_distance", "args": {"origin": "Portland", "destination": "Vancouver"}}]},
    {"id": "par_12", "question": "Generate two passwords: one 12 characters long with symbols, and one 20 characters long without symbols.",
     "tools": T("generate_password"),
     "expected": [{"name": "generate_password", "args": {"length": 12, "include_symbols": True}},
                   {"name": "generate_password", "args": {"length": 20, "include_symbols": False}}]},
]


# ---------------------------------------------------------------------------
# MODERATE CHAINED — 8 multi-step cases.
# steps: list of rounds. Each round: {"expected": [...], "mocks": {fn_name: result_dict}}
# The harness executes the model's round-1 call(s) against `mocks` and feeds
# the results back as tool messages; round 2 arguments must be DERIVED from
# the returned data (including conditional logic).
# A test passes only if EVERY step matches.
# ---------------------------------------------------------------------------

MODERATE_CHAINED = [
    {"id": "chain_01",
     "question": "What's the weather in Chicago? Then convert the temperature to Fahrenheit.",
     "tools": T("get_weather", "convert_temperature"),
     "steps": [
         {"expected": [{"name": "get_weather", "args": {"location": "Chicago"}}],
          "mocks": {"get_weather": {"location": "Chicago", "temp_c": 8, "condition": "rain"}}},
         {"expected": [{"name": "convert_temperature", "args": {"value": 8, "from_unit": "celsius",
                                                                "to_unit": "fahrenheit"}}],
          "mocks": {}},
     ]},
    {"id": "chain_02",
     "question": "Get the current price of NVDA, then place an order for 10 shares at that price.",
     "tools": T("get_stock_price", "place_order"),
     "steps": [
         {"expected": [{"name": "get_stock_price", "args": {"symbol": "NVDA"}}],
          "mocks": {"get_stock_price": {"symbol": "NVDA", "price": 131.50}}},
         {"expected": [{"name": "place_order", "args": {"symbol": "NVDA", "quantity": 10, "price": 131.50}}],
          "mocks": {}},
     ]},
    {"id": "chain_03",
     "question": "Search flights from JFK to LHR on 2026-11-15, then book the cheapest one for passenger Walker.",
     "tools": T("search_flights", "book_flight"),
     "steps": [
         {"expected": [{"name": "search_flights", "args": {"origin": "JFK", "destination": "LHR",
                                                           "date": "2026-11-15"}}],
          "mocks": {"search_flights": {"flights": [{"flight_id": "BA117", "price": 420.50},
                                                    {"flight_id": "VS3", "price": 510.00}]}}},
         {"expected": [{"name": "book_flight", "args": {"flight_id": "BA117", "passenger": "Walker"}}],
          "mocks": {}},
     ]},
    {"id": "chain_04",
     "question": "Look up user 42, then send them an email about membership renewal to their address.",
     "tools": T("get_user", "send_email"),
     "steps": [
         {"expected": [{"name": "get_user", "args": {"user_id": 42}}],
          "mocks": {"get_user": {"name": "Ana", "email": "ana@rivera.com"}}},
         {"expected": [{"name": "send_email", "args": {"to": "ana@rivera.com", "subject": "Membership renewal"}}],
          "mocks": {}},
     ]},
    {"id": "chain_05",
     "question": "Total up shopping cart 77, then apply the SAVE10 coupon to that total.",
     "tools": T("calculate_cart_total", "apply_discount"),
     "steps": [
         {"expected": [{"name": "calculate_cart_total", "args": {"cart_id": "77"}}],
          "mocks": {"calculate_cart_total": {"cart_id": "77", "total": 59.97}}},
         {"expected": [{"name": "apply_discount", "args": {"order_total": 59.97, "coupon_code": "SAVE10"}}],
          "mocks": {}},
     ]},
    {"id": "chain_06",
     "question": "Check the weather in Denver. If it's below zero degrees Celsius, set the home thermostat to 22.",
     "tools": T("get_weather", "set_thermostat"),
     "steps": [
         {"expected": [{"name": "get_weather", "args": {"location": "Denver"}}],
          "mocks": {"get_weather": {"location": "Denver", "temp_c": -5, "condition": "snow"}}},
         {"expected": [{"name": "set_thermostat", "args": {"room": "home", "temperature": 22}}],
          "mocks": {}},
     ]},
    {"id": "chain_07",
     "question": "Find a Thai restaurant in Portland for under $25 a person, then book a table for 2 there at 19:00.",
     "tools": T("find_restaurant", "book_table"),
     "steps": [
         {"expected": [{"name": "find_restaurant", "args": {"cuisine": "Thai", "location": "Portland",
                                                            "max_price_per_person": 25}}],
          "mocks": {"find_restaurant": {"restaurants": [{"restaurant_id": "R7", "name": "Thai Orchid",
                                                          "price_per_person": 18, "rating": 4.5}]}}},
         {"expected": [{"name": "book_table", "args": {"restaurant_id": "R7", "party_size": 2, "time": "19:00"}}],
          "mocks": {}},
     ]},
    {"id": "chain_08",
     "question": "Check the laptop inventory. If there are fewer than 5 in stock, create a purchase order for 10 laptops.",
     "tools": T("search_inventory", "create_purchase_order"),
     "steps": [
         {"expected": [{"name": "search_inventory", "args": {"item": "laptop"}}],
          "mocks": {"search_inventory": {"item": "laptop", "stock": 3}}},
         {"expected": [{"name": "create_purchase_order", "args": {"item": "laptop", "quantity": 10}}],
          "mocks": {}},
     ]},
]


if __name__ == "__main__":
    print(f"Simple: {len(SIMPLE_TESTS)} tests")
    print(f"Moderate parallel: {len(MODERATE_PARALLEL)} tests")
    print(f"Moderate chained: {len(MODERATE_CHAINED)} tests")
    print(f"Total tool tests: {len(SIMPLE_TESTS) + len(MODERATE_PARALLEL) + len(MODERATE_CHAINED)}")
    print(f"Tool registry: {len(TOOLS)} tools")
    # Sanity: every referenced tool exists
    for suite in (SIMPLE_TESTS, MODERATE_PARALLEL):
        for t in suite:
            assert t["expected"], t["id"]
    for t in MODERATE_CHAINED:
        assert len(t["steps"]) >= 2, t["id"]
    print("All test definitions valid.")
