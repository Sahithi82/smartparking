import os
import random
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# CONFIG
# -------------------------------
PARKING_DATA_FILE = "parking_data.csv"
LOCATIONS_FILE = "parking_locations.csv"
BOOKINGS_FILE = "bookings.csv"
SEQ_LEN = 10  # sequence length for time series


# -------------------------------
# DATA PREPARATION & MODEL
# -------------------------------
@st.cache_data
def load_and_prepare(csv_path=PARKING_DATA_FILE, seq_len=SEQ_LEN):
    """
    Load parking_data.csv, scale status, and build sequences for training.
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    scaler = MinMaxScaler()
    df["status_scaled"] = scaler.fit_transform(df[["status"]])

    X, y = [], []
    values = df["status_scaled"].values
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])

    return np.array(X), np.array(y), scaler, df


X, y, scaler, df = load_and_prepare()


@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


model = train_model(X, y)


# -------------------------------
# LOCATION & BOOKING HELPERS
# -------------------------------
def load_locations():
    """
    Load parking locations (id, name, lat, lng, total_slots).
    If file doesn't exist, use some default demo locations.
    """
    if os.path.exists(LOCATIONS_FILE):
        loc_df = pd.read_csv(LOCATIONS_FILE)
    else:
        # Demo locations – you can edit these or replace with your own file
        loc_df = pd.DataFrame(
            [
                {
                    "location_id": 1,
                    "name": "SR Nagar Parking",
                    "latitude": 17.4412,
                    "longitude": 78.4480,
                    "total_slots": 40,
                },
                {
                    "location_id": 2,
                    "name": "KPHB Metro Lot",
                    "latitude": 17.4945,
                    "longitude": 78.3995,
                    "total_slots": 60,
                },
                {
                    "location_id": 3,
                    "name": "Forum Mall Parking",
                    "latitude": 17.4376,
                    "longitude": 78.4489,
                    "total_slots": 80,
                },
            ]
        )
    # Ensure types
    loc_df["location_id"] = loc_df["location_id"].astype(int)
    loc_df["total_slots"] = loc_df["total_slots"].astype(int)
    return loc_df


def load_bookings():
    """
    Load existing bookings; if file doesn't exist, return empty DataFrame.
    """
    if os.path.exists(BOOKINGS_FILE):
        bdf = pd.read_csv(BOOKINGS_FILE)
        if not bdf.empty:
            bdf["start_time"] = pd.to_datetime(bdf["start_time"])
            bdf["end_time"] = pd.to_datetime(bdf["end_time"])
            bdf["location_id"] = bdf["location_id"].astype(int)
        return bdf
    else:
        return pd.DataFrame(
            columns=["booking_id", "location_id", "start_time", "end_time", "user_id"]
        )


def save_bookings(bdf: pd.DataFrame):
    bdf.to_csv(BOOKINGS_FILE, index=False)


# -------------------------------
# PREDICTION HELPERS
# -------------------------------
def predict_free_slots_for_day(model, scaler, base_seq, date_obj):
    """
    Predict free slots for a full day (144 × 10-min intervals).
    Note: model isn't really date-aware, but we keep date in signature.
    """
    seq = base_seq.copy()
    preds = []
    for _ in range(144):
        p = model.predict([seq])[0]
        preds.append(p)
        seq = np.append(seq[1:], p)

    actuals = np.round(
        scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    ).astype(int)
    return actuals


def get_predicted_slots_at_datetime(model, scaler, base_seq, dt: datetime):
    """
    Get predicted free slots at a specific datetime, based on 10-min index.
    """
    minutes = dt.hour * 60 + dt.minute
    index = minutes // 10  # 0-143
    day_preds = predict_free_slots_for_day(model, scaler, base_seq, dt.date())
    if 0 <= index < len(day_preds):
        return int(day_preds[index])
    return 0


def count_bookings(bookings_df, location_id, target_start, target_end):
    """
    Count overlapping bookings for a location in [target_start, target_end).
    """
    if bookings_df.empty:
        return 0

    mask = (
        (bookings_df["location_id"] == int(location_id))
        & ~(bookings_df["end_time"] <= target_start)
        & ~(bookings_df["start_time"] >= target_end)
    )
    return mask.sum()


def get_available_slots(location_row, dt_start, dt_end, model, scaler, base_seq, bookings_df):
    """
    Combine model prediction + existing bookings to estimate available slots.
    """
    predicted_free = get_predicted_slots_at_datetime(model, scaler, base_seq, dt_start)
    total_slots = int(location_row["total_slots"])
    booked = count_bookings(bookings_df, location_row["location_id"], dt_start, dt_end)

    # Very simple rule: can't exceed physical capacity
    available = min(predicted_free, total_slots - booked)
    return max(available, 0), predicted_free, booked


# -------------------------------
# VISUAL HELPERS
# -------------------------------
def draw_route_map(free_slots, best_slot=None, title="Route Map"):
    """
    Original schematic route map (for single-lot demo).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    slots = list(free_slots.keys())

    # Define 2-row × 5-col layout (spaced)
    cols = 5
    spacing = 3
    lots = {
        slots[i]: ((i % cols) * spacing + 4, -(i // cols) * spacing)
        for i in range(len(slots))
    }

    # Entrance and exit horizontally aligned
    entrance = (0, -1.5)  # Far left, mid of row gap
    exit_pos = (20, -1.5)  # Far right

    # Draw parking lots
    for lot, (x, y) in lots.items():
        ax.add_patch(
            plt.Rectangle(
                (x - 1, y - 1), 2, 2, color="green" if lot == best_slot else "gray"
            )
        )
        ax.text(
            x,
            y,
            lot,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if lot == best_slot else "black",
        )

    # Entrance/Exit
    ax.plot(*entrance, "go")
    ax.text(entrance[0] - 0.5, entrance[1], "Entrance", ha="right")
    ax.plot(*exit_pos, "ro")
    ax.text(exit_pos[0] + 0.5, exit_pos[1], "Exit", ha="left")

    # --- Route Logic ---
    if best_slot:
        bx, by = lots[best_slot]
        path_y = entrance[1]  # horizontal path

        route_x = [entrance[0], bx, bx]
        route_y = [entrance[1], path_y, by]
        ax.plot(route_x, route_y, "r--", lw=2)
    else:
        ax.text(
            10,
            -6,
            "No Free Parking Slot Available!",
            ha="center",
            fontsize=12,
            color="red",
        )

    ax.set_xlim(-3, 24)
    ax.set_ylim(-7, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.title(title)
    st.pyplot(fig)


def render_google_map(location_row, zoom=16):
    """
    Show Google Map for a given location.
    If google_maps_api_key isn't configured, fall back to a basic map.
    """
    lat = float(location_row["latitude"])
    lng = float(location_row["longitude"])

    try:
        api_key = st.secrets["google_maps_api_key"]
        src = (
            "https://www.google.com/maps/embed/v1/place"
            f"?key={api_key}&q={lat},{lng}&zoom={zoom}"
        )
        components.iframe(src, height=400)
    except Exception:
        st.info(
            "Google Maps API key not configured. "
            "Showing a simple Streamlit map instead."
        )
        st.map(pd.DataFrame({"lat": [lat], "lon": [lng]}))


# -------------------------------
# STREAMLIT APP
# -------------------------------
st.title("🅿️ Smart Parking Forecasting & Booking")

mode = st.radio(
    "Prediction Mode:",
    options=["No Date (Real-Time)", "Specific Date", "Map & Booking"],
)


# --- Mode 1: No Date (Real-Time) ---
if mode == "No Date (Real-Time)":
    st.subheader("⏱️ Real-Time Free Slot Prediction (Demo)")

    if st.button("Predict Now"):
        nsteps = 10
        preds = []
        seq = X[-1].copy()

        for _ in range(nsteps):
            p = model.predict([seq])[0]
            preds.append(p)
            seq = np.append(seq[1:], p)

        preds = np.round(
            scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        )

        # Fake 'real-time' values for comparison
        realtime = [random.randint(0, 35) for _ in range(nsteps)]
        times = [
            (datetime.now() + timedelta(minutes=10 * i)).strftime("%H:%M")
            for i in range(nsteps)
        ]
        slots = [f"Lot_{i+1}" for i in range(nsteps)]

        for i in range(nsteps):
            st.write(f"{times[i]} → {int(preds[i])} free slots")

        # Bar Chart
        fig, ax = plt.subplots()
        idx = np.arange(nsteps)
        w = 0.35
        ax.bar(idx, realtime, w, label="Real-Time", color="skyblue")
        ax.bar(idx + w, preds, w, label="Predicted", color="salmon")
        ax.set_xticks(idx + w / 2)
        ax.set_xticklabels(times, rotation=45)
        ax.legend()
        ax.set_title("Real-Time vs Predicted Free Slots")
        st.pyplot(fig)

        # Route Map
        free_slots = {slots[i]: int(realtime[i]) for i in range(nsteps)}
        available = [s for s in free_slots if free_slots[s] > 0]
        best_slot = max(available, key=lambda x: free_slots[x]) if available else None
        draw_route_map(free_slots, best_slot, "Best Slot & Route")


# --- Mode 2: Specific Date Predictions ---
elif mode == "Specific Date":
    st.subheader("📆 Full-Day Prediction for a Specific Date")

    d = st.date_input("Choose a date:")
    if st.button("Predict for Date"):
        today = datetime.today().date()
        if d < today:
            st.error("Cannot predict for past dates.")
        else:
            seq = X[-1].copy()
            preds = []
            for _ in range(144):  # Full day at 10-min intervals
                p = model.predict([seq])[0]
                preds.append(p)
                seq = np.append(seq[1:], p)

            actuals = np.round(
                scaler
                .inverse_transform(np.array(preds).reshape(-1, 1))
                .flatten()
            )

            for i in range(144):
                ts = datetime.combine(d, datetime.min.time()) + timedelta(
                    minutes=10 * i
                )
                st.write(f"{ts.strftime('%H:%M')} → {int(actuals[i])} free slots")

            avg_slots = int(np.mean(actuals[:10]))
            free_slots = {
                f"Lot_{i+1}": random.randint(5, 35) if avg_slots > 0 else 0
                for i in range(10)
            }
            best_slot = (
                max(free_slots, key=lambda k: free_slots[k])
                if any(free_slots.values())
                else None
            )
            draw_route_map(
                free_slots,
                best_slot,
                f"Best Slot on {d.strftime('%Y-%m-%d')}",
            )


# --- Mode 3: Map & Advance Booking ---
elif mode == "Map & Booking":
    st.subheader("🗺️ Map View & Advance Booking (up to 30 days)")

    locations_df = load_locations()
    bookings_df = load_bookings()

    if locations_df.empty:
        st.error(
            "No parking locations configured. "
            f"Please create '{LOCATIONS_FILE}' with your locations."
        )
    else:
        loc_names = locations_df["name"].tolist()
        selected_name = st.selectbox("Select Parking Location", loc_names)
        selected_loc = locations_df[locations_df["name"] == selected_name].iloc[0]

        st.markdown(f"**Total slots at this location:** {int(selected_loc['total_slots'])}")
        render_google_map(selected_loc)

        # Date & time selection
        today = datetime.today().date()
        max_date = today + timedelta(days=30)
        date = st.date_input(
            "Select Date (booking allowed up to 30 days from today)",
            min_value=today,
            max_value=max_date,
            value=today,
        )

        col1, col2 = st.columns(2)
        with col1:
            hour = st.number_input("Start Hour (0–23)", min_value=0, max_value=23, value=datetime.now().hour)
        with col2:
            minute = st.selectbox("Start Minute", [0, 10, 20, 30, 40, 50], index=0)

        duration_hours = st.slider("Duration (hours)", min_value=1, max_value=8, value=1)

        start_dt = datetime.combine(date, dtime(int(hour), int(minute)))
        end_dt = start_dt + timedelta(hours=int(duration_hours))

        base_seq = X[-1]
        available, predicted_free, already_booked = get_available_slots(
            selected_loc, start_dt, end_dt, model, scaler, base_seq, bookings_df
        )

        st.write(f"🧠 Predicted free slots at start time: **{predicted_free}**")
        st.write(f"📚 Existing overlapping bookings in this period: **{already_booked}**")
        st.write(f"✅ Slots available to book now: **{available}**")

        if st.button("Book Slot"):
            if available > 0:
                if bookings_df.empty:
                    new_id = 1
                else:
                    new_id = int(bookings_df["booking_id"].max()) + 1

                new_row = {
                    "booking_id": new_id,
                    "location_id": int(selected_loc["location_id"]),
                    "start_time": start_dt,
                    "end_time": end_dt,
                    "user_id": "guest",  # you can extend this later with login/usernames
                }
                bookings_df = pd.concat(
                    [bookings_df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                save_bookings(bookings_df)
                st.success("🎉 Slot booked successfully!")
            else:
                st.error("❌ No slots available for this time range.")

        st.markdown("### 📋 Bookings for this location on selected date")
        if not bookings_df.empty:
            mask = (
                (bookings_df["location_id"] == int(selected_loc["location_id"]))
                & (bookings_df["start_time"].dt.date == date)
            )
            day_bookings = bookings_df[mask].sort_values("start_time")
            if day_bookings.empty:
                st.info("No bookings for this location on the selected date yet.")
            else:
                st.dataframe(day_bookings)
        else:
            st.info("No bookings have been made yet.")
