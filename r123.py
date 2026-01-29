import os
import random
import string
from datetime import datetime, timedelta, time as dtime
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # <<< added
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import qrcode  # for QR code generation
from PIL import Image
import cv2  # for QR image decoding with OpenCV

# -------------------------------
# CONFIG
# -------------------------------
PARKING_DATA_FILE = "parking_data.csv"
LOCATIONS_FILE = "parking_locations.csv"
BOOKINGS_FILE = "bookings.csv"
SLOT_STATUS_FILE = "slot_status.csv"   # summary for management side
WAITLIST_FILE = "waitlist.csv"
RATINGS_FILE = "ratings.csv"
SEQ_LEN = 10  # sequence length for time series
ADMIN_PASSWORD = "admin123"  # change this for your project

# Session state for rating popup after "End Now"
if "rating_booking_id" not in st.session_state:
    st.session_state["rating_booking_id"] = None


# -------------------------------
# HELPERS
# -------------------------------
def generate_captcha_code(length=6):
    """Simple random alphanumeric booking code (used like a captcha)."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def send_sms(phone: str, message: str):
    """
    Send an SMS to phone with the given message.

    Implementation uses Twilio IF you configure it, otherwise it silently does nothing.

    To actually send SMS, you need:
      - pip install twilio
      - Set environment variables:
          TWILIO_ACCOUNT_SID
          TWILIO_AUTH_TOKEN
          TWILIO_FROM_NUMBER
    """
    try:
        from twilio.rest import Client  # type: ignore

        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")

        if not account_sid or not auth_token or not from_number:
            return

        client = Client(account_sid, auth_token)
        client.messages.create(
            body=message,
            from_=from_number,
            to=phone,
        )
    except ImportError:
        return
    except Exception:
        return


# -------------------------------
# DATA PREPARATION & MODEL
# -------------------------------
@st.cache_data
def load_and_prepare(csv_path=PARKING_DATA_FILE, seq_len=SEQ_LEN):
    """Load parking_data.csv, scale status, and build sequences for training."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    scaler = MinMaxScaler()
    df["status_scaled"] = scaler.fit_transform(df[["status"]])

    X, y = [], []
    values = df["status_scaled"].values
    for i in range(len(values) - seq_len):
        X.append(values[i: i + seq_len])
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

    loc_df["location_id"] = loc_df["location_id"].astype(int)
    loc_df["total_slots"] = loc_df["total_slots"].astype(int)
    return loc_df


def load_bookings():
    """
    Load existing bookings; if file doesn't exist, return empty DataFrame
    with the expected columns (including user details + status + captcha_code).
    Phone is treated strictly as a string, stored exactly as it was entered.

    ALSO: auto-deletes any booking whose end_time is already over (expired).
    """
    cols = [
        "booking_id",
        "location_id",
        "start_time",
        "end_time",
        "user_id",
        "user_name",
        "phone",
        "status",
        "captcha_code",
    ]

    if os.path.exists(BOOKINGS_FILE):
        bdf = pd.read_csv(BOOKINGS_FILE, dtype={"phone": str})
        if not bdf.empty:
            if "start_time" in bdf.columns:
                bdf["start_time"] = pd.to_datetime(bdf["start_time"])
            if "end_time" in bdf.columns:
                bdf["end_time"] = pd.to_datetime(bdf["end_time"])
            if "location_id" in bdf.columns:
                bdf["location_id"] = bdf["location_id"].astype(int)
            if "booking_id" in bdf.columns:
                bdf["booking_id"] = pd.to_numeric(
                    bdf["booking_id"], errors="coerce"
                ).astype("Int64")

        for c in cols:
            if c not in bdf.columns:
                if c in ["user_id", "user_name", "phone", "status", "captcha_code"]:
                    if c == "status":
                        bdf[c] = "active"
                    else:
                        bdf[c] = ""
                else:
                    bdf[c] = np.nan

        bdf["phone"] = bdf["phone"].astype(str).str.strip()

        # auto-delete expired bookings
        now = datetime.now()
        if "end_time" in bdf.columns:
            keep_mask = bdf["end_time"].isna() | (bdf["end_time"] >= now)
            bdf = bdf[keep_mask].copy()

        bdf = bdf[cols]
        bdf.to_csv(BOOKINGS_FILE, index=False)

        return bdf
    else:
        bdf = pd.DataFrame(columns=cols)
        return bdf


def save_bookings(bdf: pd.DataFrame):
    """Save bookings to CSV, forcing phone to be stored as plain string."""
    if "phone" in bdf.columns:
        bdf["phone"] = bdf["phone"].astype(str).str.strip()
    bdf.to_csv(BOOKINGS_FILE, index=False)


def update_slot_status(locations_df: pd.DataFrame, bookings_df: pd.DataFrame):
    """
    Create/update a summary file with booked and free slots per location & date.
    Only counts bookings with status 'active' or 'extended'.
    """
    rows = []

    if bookings_df.empty:
        today = datetime.today().date()
        for _, loc in locations_df.iterrows():
            total_slots = int(loc["total_slots"])
            rows.append(
                {
                    "location_id": int(loc["location_id"]),
                    "name": loc["name"],
                    "date": today.isoformat(),
                    "total_slots": total_slots,
                    "booked_slots": 0,
                    "free_slots": total_slots,
                }
            )
    else:
        bookings_active = bookings_df[
            bookings_df["status"].isin(["active", "extended"])
        ].copy()

        if bookings_active.empty:
            today = datetime.today().date()
            for _, loc in locations_df.iterrows():
                total_slots = int(loc["total_slots"])
                rows.append(
                    {
                        "location_id": int(loc["location_id"]),
                        "name": loc["name"],
                        "date": today.isoformat(),
                        "total_slots": total_slots,
                        "booked_slots": 0,
                        "free_slots": total_slots,
                    }
                )
        else:
            bookings_active["date"] = bookings_active["start_time"].dt.date

            for _, loc in locations_df.iterrows():
                loc_id = int(loc["location_id"])
                total_slots = int(loc["total_slots"])
                loc_bookings = bookings_active[
                    bookings_active["location_id"] == loc_id
                ]
                for date in sorted(loc_bookings["date"].unique()):
                    mask = loc_bookings["date"] == date
                    booked_count = mask.sum()
                    free_slots = max(total_slots - booked_count, 0)
                    rows.append(
                        {
                            "location_id": loc_id,
                            "name": loc["name"],
                            "date": date.isoformat(),
                            "total_slots": total_slots,
                            "booked_slots": int(booked_count),
                            "free_slots": int(free_slots),
                        }
                    )

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(SLOT_STATUS_FILE, index=False)


# --------- WAITLIST HELPERS ----------
def load_waitlist():
    cols = [
        "phone",
        "name",
        "location_id",
        "desired_start",
        "desired_end",
        "timestamp",
    ]
    if os.path.exists(WAITLIST_FILE):
        wdf = pd.read_csv(WAITLIST_FILE, dtype={"phone": str})
        if not wdf.empty:
            if "desired_start" in wdf.columns:
                wdf["desired_start"] = pd.to_datetime(wdf["desired_start"])
            if "desired_end" in wdf.columns:
                wdf["desired_end"] = pd.to_datetime(wdf["desired_end"])
            if "timestamp" in wdf.columns:
                wdf["timestamp"] = pd.to_datetime(wdf["timestamp"])
            if "location_id" in wdf.columns:
                wdf["location_id"] = wdf["location_id"].astype(int)
        for c in cols:
            if c not in wdf.columns:
                wdf[c] = "" if c in ["phone", "name"] else np.nan
        return wdf[cols]
    else:
        return pd.DataFrame(columns=cols)


def save_waitlist(wdf: pd.DataFrame):
    if "phone" in wdf.columns:
        wdf["phone"] = wdf["phone"].astype(str).str.strip()
    wdf.to_csv(WAITLIST_FILE, index=False)


# --------- RATINGS HELPERS ----------
def load_ratings():
    cols = [
        "booking_id",
        "location_id",
        "phone",
        "rating",
        "feedback",
        "timestamp",
    ]
    if os.path.exists(RATINGS_FILE):
        rdf = pd.read_csv(RATINGS_FILE, dtype={"phone": str})
        if not rdf.empty:
            if "timestamp" in rdf.columns:
                rdf["timestamp"] = pd.to_datetime(rdf["timestamp"])
            if "location_id" in rdf.columns:
                rdf["location_id"] = rdf["location_id"].astype(int)
            if "booking_id" in rdf.columns:
                rdf["booking_id"] = pd.to_numeric(
                    rdf["booking_id"], errors="coerce"
                ).astype("Int64")
            if "rating" in rdf.columns:
                rdf["rating"] = pd.to_numeric(
                    rdf["rating"], errors="coerce"
                ).astype("float")
        for c in cols:
            if c not in rdf.columns:
                rdf[c] = "" if c in ["phone", "feedback"] else np.nan
        return rdf[cols]
    else:
        return pd.DataFrame(columns=cols)


def save_ratings(rdf: pd.DataFrame):
    if "phone" in rdf.columns:
        rdf["phone"] = rdf["phone"].astype(str).str.strip()
    rdf.to_csv(RATINGS_FILE, index=False)


# --------- NO-SHOW & LOYALTY HELPERS ----------
def get_no_show_stats(bookings_df):
    if bookings_df is None or bookings_df.empty:
        return pd.DataFrame({
            "phone": [],
            "no_show_count": [],
            "last_no_show_date": []
        })

    df = bookings_df[bookings_df["status"] == "no_show"].copy()
    if df.empty:
        return pd.DataFrame({
            "phone": [],
            "no_show_count": [],
            "last_no_show_date": []
        })

    stats = (
        df.groupby("phone")
          .agg(
              no_show_count=("booking_id", "count"),
              last_no_show_date=("start_time", lambda x: x.max().date()),
          )
          .reset_index()
    )
    return stats



def get_loyalty_stats(bookings_df: pd.DataFrame) -> pd.DataFrame:
    """Return phone-wise loyalty stats: phone, total_completed, last_booking_date."""
    if bookings_df.empty:
        return pd.DataFrame(columns=["phone", "total_completed", "last_booking_date"])

    df = bookings_df[bookings_df["status"] == "completed"].copy()
    if df.empty:
        return pd.DataFrame(columns=["phone", "total_completed", "last_booking_date"])

    stats = (
        df.groupby("phone")
        .agg(
            total_completed=("booking_id", "count"),
            last_booking_date=("end_time", lambda x: x.max().date()),
        )
        .reset_index()
    )
    return stats


def get_membership_level(total_completed: int) -> str:
    if total_completed >= 10:
        return "Platinum"
    elif total_completed >= 5:
        return "Gold"
    elif total_completed >= 1:
        return "Silver"
    else:
        return "New User"


# -------------------------------
# PREDICTION HELPERS
# -------------------------------
def predict_free_slots_for_day(model, scaler, base_seq, date_obj):
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
    minutes = dt.hour * 60 + dt.minute
    index = minutes // 10  # 0-143
    day_preds = predict_free_slots_for_day(model, scaler, base_seq, dt.date())
    if 0 <= index < len(day_preds):
        return int(day_preds[index])
    return 0


def count_bookings(bookings_df, location_id, target_start, target_end):
    if bookings_df.empty:
        return 0

    active_df = bookings_df[bookings_df["status"].isin(["active", "extended"])]
    if active_df.empty:
        return 0

    mask = (
        (active_df["location_id"] == int(location_id))
        & ~(active_df["end_time"] <= target_start)
        & ~(active_df["start_time"] >= target_end)
    )
    return mask.sum()


def get_available_slots(
    location_row,
    dt_start,
    dt_end,
    model,
    scaler,
    base_seq,
    bookings_df,
):
    predicted_free = get_predicted_slots_at_datetime(model, scaler, base_seq, dt_start)
    total_slots = int(location_row["total_slots"])
    booked = count_bookings(
        bookings_df, location_row["location_id"], dt_start, dt_end
    )

    available = min(predicted_free, total_slots - booked)
    return max(available, 0), predicted_free, booked


# -------------------------------
# VISUAL HELPERS
# -------------------------------
def draw_route_map(free_slots, best_slot=None, title="Route Map"):
    fig, ax = plt.subplots(figsize=(10, 6))
    slots = list(free_slots.keys())

    cols = 5
    spacing = 3
    lots = {
        slots[i]: ((i % cols) * spacing + 4, -(i // cols) * spacing)
        for i in range(len(slots))
    }

    entrance = (0, -1.5)
    exit_pos = (20, -1.5)

    for lot, (x, y) in lots.items():
        ax.add_patch(
            plt.Rectangle(
                (x - 1, y - 1),
                2,
                2,
                color="green" if lot == best_slot else "gray",
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

    ax.plot(*entrance, "go")
    ax.text(entrance[0] - 0.5, entrance[1], "Entrance", ha="right")
    ax.plot(*exit_pos, "ro")
    ax.text(exit_pos[0] + 0.5, exit_pos[1], "Exit", ha="left")

    if best_slot:
        bx, by = lots[best_slot]
        path_y = entrance[1]
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


def render_google_map(location_row):
    lat = float(location_row["latitude"])
    lng = float(location_row["longitude"])
    st.map(pd.DataFrame({"lat": [lat], "lon": [lng]}))


def generate_qr_image(payload: str):
    img = qrcode.make(payload)
    return img


def parse_qr_payload(qr_text: str):
    parts = [p.strip() for p in qr_text.split(";") if p.strip()]
    data = {}
    for p in parts:
        if "=" in p:
            key, val = p.split("=", 1)
            data[key.strip()] = val.strip()
    return data


# -------------------------------
# STREAMLIT APP
# -------------------------------
st.title("🅿️ Smart Parking Forecasting, Booking & Management")

mode = st.radio(
    "Choose Mode:",
    options=[
        "No Date (Real-Time)",
        "Specific Date",
        "Map & Booking (User)",
        "My Bookings",
        "QR Verify (Staff)",
        "Management Dashboard",
    ],
)


# --- Mode 1: No Date (Real-Time) ---
if mode == "No Date (Real-Time)":
    st.subheader("Real-Time Free Slot Prediction (Demo)")

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

        realtime = [random.randint(0, 35) for _ in range(nsteps)]
        times = [
            (datetime.now() + timedelta(minutes=10 * i)).strftime("%H:%M")
            for i in range(nsteps)
        ]
        slots = [f"Lot_{i+1}" for i in range(nsteps)]

        for i in range(nsteps):
            st.write(f"{times[i]} → {int(preds[i])} predicted free slots")

        fig, ax = plt.subplots()
        idx = np.arange(nsteps)
        w = 0.35
        ax.bar(idx, realtime, w, label="Real-Time")
        ax.bar(idx + w, preds, w, label="Predicted")
        ax.set_xticks(idx + w / 2)
        ax.set_xticklabels(times, rotation=45)
        ax.legend()
        ax.set_title("Real-Time vs Predicted Free Slots")
        st.pyplot(fig)

        free_slots = {slots[i]: int(realtime[i]) for i in range(nsteps)}
        available = [s for s in free_slots if free_slots[s] > 0]
        best_slot = max(available, key=lambda x: free_slots[x]) if available else None
        draw_route_map(free_slots, best_slot, "Best Slot & Route")


# --- Mode 2: Specific Date Predictions ---
elif mode == "Specific Date":
    st.subheader("Full-Day Prediction for a Specific Date")

    d = st.date_input("Choose a date:")
    if st.button("Predict for Date"):
        today = datetime.today().date()
        if d < today:
            st.error("Cannot predict for past dates.")
        else:
            seq = X[-1].copy()
            preds = []
            for _ in range(144):
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


# --- Mode 3: Map & Advance Booking (User side) ---
elif mode == "Map & Booking (User)":
    st.subheader("Map View & Advance Booking")

    locations_df = load_locations()
    bookings_df = load_bookings()
    no_show_stats = get_no_show_stats(bookings_df)
    banned_phones = set(
        no_show_stats[no_show_stats["no_show_count"] >= 3]["phone"].astype(str)
    )

    if locations_df.empty:
        st.error(
            "No parking locations configured. "
            f"Please create '{LOCATIONS_FILE}' with your locations."
        )
    else:
        st.map(locations_df.rename(columns={"latitude": "lat", "longitude": "lon"}))

        loc_names = locations_df["name"].tolist()
        selected_name = st.selectbox("Select Parking Location", loc_names)
        selected_loc = locations_df[locations_df["name"] == selected_name].iloc[0]

        st.markdown(
            f"**Total slots at this location:** {int(selected_loc['total_slots'])}"
        )
        render_google_map(selected_loc)

        st.markdown("### Your Details")
        user_name = st.text_input("Your Name")
        phone_input = st.text_input("Phone Number (10 digits)")
        phone_clean = phone_input.strip()

        if phone_clean and phone_clean in banned_phones:
            st.error(
                "This phone number has been flagged for multiple no-shows "
                "and cannot book new slots. Please contact the administrator."
            )

        st.markdown("### Booking Time")
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
            hour = st.number_input(
                "Start Hour (0–23)", min_value=0, max_value=23, value=datetime.now().hour
            )
        with col2:
            minute = st.selectbox(
                "Start Minute (10-min steps)", [0, 10, 20, 30, 40, 50], index=0
            )

        duration_hours = st.slider(
            "How many hours do you want the booking?",
            min_value=1,
            max_value=8,
            value=1,
        )

        start_dt = datetime.combine(date, dtime(int(hour), int(minute)))
        end_dt = start_dt + timedelta(hours=int(duration_hours))

        base_seq = X[-1]
        bookings_df = load_bookings()
        available, predicted_free, already_booked = get_available_slots(
            selected_loc, start_dt, end_dt, model, scaler, base_seq, bookings_df
        )

        st.write(f"Predicted free slots at start time: **{predicted_free}**")
        st.write(
            f"Existing overlapping bookings in this period: **{already_booked}**"
        )
        st.write(f"Slots available to book now: **{available}**")

        st.markdown(
            f"**Booking Summary:** {selected_loc['name']} on {date} "
            f"from {start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')}"
        )

        # ---- Waitlist message if full ----
        if available <= 0:
            st.warning(
                "All slots are currently full for this time. "
                "You can join the waitlist and staff may contact you if a slot opens."
            )
            if st.button("Join Waitlist"):
                if not user_name.strip():
                    st.error("Please enter your name.")
                elif not phone_clean:
                    st.error("Please enter your phone number.")
                elif (not phone_clean.isdigit()) or len(phone_clean) != 10:
                    st.error("Phone number must be exactly 10 digits (only numbers).")
                else:
                    wdf = load_waitlist()
                    new_w = {
                        "phone": phone_clean,
                        "name": user_name.strip(),
                        "location_id": int(selected_loc["location_id"]),
                        "desired_start": start_dt,
                        "desired_end": end_dt,
                        "timestamp": datetime.now(),
                    }
                    wdf = pd.concat([wdf, pd.DataFrame([new_w])], ignore_index=True)
                    save_waitlist(wdf)
                    st.success(
                        "You have been added to the waitlist for this timeslot. "
                        "This is just a queue list for staff, no slot is booked."
                    )

        if st.button("Book Slot"):
            phone_clean = phone_input.strip()

            if phone_clean and phone_clean in banned_phones:
                st.error(
                    "You are currently banned from booking due to repeated no-shows."
                )
            elif not user_name.strip():
                st.error("Please enter your name.")
            elif not phone_clean:
                st.error("Please enter your phone number.")
            elif (not phone_clean.isdigit()) or len(phone_clean) != 10:
                st.error("Phone number must be exactly 10 digits (only numbers).")
            elif available <= 0:
                st.error(
                    "No slots available for this time range. "
                    "You can use the waitlist button instead."
                )
            else:
                bookings_df = load_bookings()
                if bookings_df.empty:
                    new_id = 1
                else:
                    max_id = bookings_df["booking_id"].max()
                    new_id = int(max_id) + 1 if pd.notna(max_id) else 1

                captcha = generate_captcha_code()

                new_row = {
                    "booking_id": new_id,
                    "location_id": int(selected_loc["location_id"]),
                    "start_time": start_dt,
                    "end_time": end_dt,
                    "user_id": phone_clean,
                    "user_name": user_name.strip(),
                    "phone": phone_clean,
                    "status": "active",
                    "captcha_code": captcha,
                }
                bookings_df = pd.concat(
                    [bookings_df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                save_bookings(bookings_df)
                update_slot_status(locations_df, bookings_df)

                st.success("Slot booked successfully!")

                send_sms(
                    phone_clean,
                    "Booking done successfully for your parking slot.",
                )

                st.markdown(f"**Your Booking Code (Captcha):** `{captcha}`")

                qr_payload = (
                    f"BOOKING_ID={new_id};"
                    f"LOCATION_ID={int(selected_loc['location_id'])};"
                    f"LOCATION_NAME={selected_loc['name']};"
                    f"USER_NAME={user_name.strip()};"
                    f"PHONE={phone_clean};"
                    f"START={start_dt.isoformat()};"
                    f"END={end_dt.isoformat()};"
                    f"CAPTCHA={captcha}"
                )
                qr_img = generate_qr_image(qr_payload)

                qr_bytes = BytesIO()
                qr_img.save(qr_bytes, format="PNG")
                qr_bytes.seek(0)

                st.markdown("### Your Booking QR Code")
                st.image(
                    qr_bytes,
                    caption="Show this QR code OR your booking code at the entrance.",
                    use_column_width=False,
                )

                qr_bytes_download = BytesIO(qr_bytes.getvalue())
                st.download_button(
                    label="Download QR Code",
                    data=qr_bytes_download.getvalue(),
                    file_name=f"booking_{new_id}_qr.png",
                    mime="image/png",
                )

        st.markdown("### Bookings for this location on selected date")
        bookings_df = load_bookings()
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


# --- Mode 4: My Bookings ---
elif mode == "My Bookings":
    st.subheader("My Bookings")

    bookings_df = load_bookings()
    locations_df = load_locations()

    phone_input = st.text_input("Enter your phone number (10 digits):")
    phone_clean = phone_input.strip()

    if phone_input.strip():
        if (not phone_clean.isdigit()) or len(phone_clean) != 10:
            st.error("Phone number must be exactly 10 digits (only numbers).")
        else:
            bookings_df["phone"] = bookings_df["phone"].astype(str).str.strip()

            # Hide cancelled bookings from user panel
            user_bookings = bookings_df[
                (bookings_df["phone"] == phone_clean)
                & (bookings_df["status"] != "cancelled")
            ].copy()

            if user_bookings.empty:
                st.info("No bookings found for this phone number.")
            else:
                user_bookings = user_bookings.sort_values("start_time")
                st.markdown("### All Your Bookings")
                st.dataframe(user_bookings)

                # Loyalty info
                completed_df = bookings_df[
                    (bookings_df["phone"] == phone_clean)
                    & (bookings_df["status"] == "completed")
                ]
                total_completed = len(completed_df)
                level = get_membership_level(total_completed)
                st.markdown(
                    f"**Total bookings completed:** {total_completed}  |  "
                    f"**Membership Level:** {level}"
                )

                now = datetime.now()
                upcoming = user_bookings[user_bookings["end_time"] >= now]
                if upcoming.empty:
                    st.info("No upcoming bookings to manage.")
                else:
                    st.markdown("### Manage Upcoming Bookings")
                    upcoming_ids = upcoming["booking_id"].dropna().astype(int).tolist()
                    selected_id = st.selectbox(
                        "Select a booking to manage:",
                        upcoming_ids,
                    )

                    selected_row = upcoming[upcoming["booking_id"] == selected_id].iloc[0]
                    st.write("Selected Booking:")
                    st.write(selected_row[["booking_id", "location_id", "start_time", "end_time", "status"]])

                    col1, col2, col3 = st.columns(3)

                    # --- USER EXTEND ---
                    with col1:
                        if st.button("Extend by 1 hour"):
                            bookings_df = load_bookings()
                            locations_df = load_locations()
                            loc_row = locations_df[
                                locations_df["location_id"] == int(selected_row["location_id"])
                            ].iloc[0]
                            old_end = selected_row["end_time"]
                            new_end = old_end + timedelta(hours=1)

                            base_seq = X[-1]
                            available_ext, pred_ext, booked_ext = get_available_slots(
                                loc_row, old_end, new_end, model, scaler, base_seq, bookings_df
                            )

                            if available_ext <= 0:
                                st.error("Cannot extend; no slots available in the extra time window.")
                            else:
                                idx = bookings_df["booking_id"] == selected_id
                                bookings_df.loc[idx, "end_time"] = new_end
                                bookings_df.loc[idx, "status"] = "extended"
                                save_bookings(bookings_df)
                                update_slot_status(locations_df, bookings_df)
                                st.success(f"Booking {selected_id} extended by 1 hour (until {new_end}).")

                    # --- USER END NOW ---
                    with col2:
                        if st.button("End Now"):
                            bookings_df = load_bookings()
                            now_dt = datetime.now()
                            if now_dt <= selected_row["start_time"]:
                                st.error("Cannot end before the booking start time.")
                            elif now_dt >= selected_row["end_time"]:
                                st.info("Booking time already passed.")
                            else:
                                idx = bookings_df["booking_id"] == selected_id
                                bookings_df.loc[idx, "end_time"] = now_dt
                                bookings_df.loc[idx, "status"] = "completed"
                                save_bookings(bookings_df)
                                locations_df = load_locations()
                                update_slot_status(locations_df, bookings_df)
                                st.success(f"Booking {selected_id} ended now at {now_dt}.")

                                # Show feedback popup for this booking
                                st.session_state["rating_booking_id"] = selected_id
                                st.rerun()

                    # --- USER CANCEL ---
                    with col3:
                        if st.button("Cancel Booking"):
                            bookings_df = load_bookings()
                            idx = bookings_df["booking_id"] == selected_id
                            bookings_df.loc[idx, "status"] = "cancelled"
                            save_bookings(bookings_df)
                            locations_df = load_locations()
                            update_slot_status(locations_df, bookings_df)
                            st.success(f"Booking {selected_id} marked as cancelled.")

                            phone_for_sms = selected_row["phone"]
                            send_sms(
                                phone_for_sms,
                                "Your booking is cancelled successfully.",
                            )

                            st.rerun()

                # ---- Rating Popup for last ended booking ----
                bookings_df = load_bookings()
                completed_for_user = bookings_df[
                    (bookings_df["phone"] == phone_clean)
                    & (bookings_df["status"] == "completed")
                ].copy()

                highlight_id = st.session_state.get("rating_booking_id")
                if highlight_id is not None:
                    match_row = completed_for_user[
                        completed_for_user["booking_id"] == highlight_id
                    ]
                    if not match_row.empty:
                        st.markdown("---")
                        st.markdown("### Please rate your last visit")
                        row = match_row.iloc[0]
                        loc_id = int(row["location_id"])
                        location_name = (
                            locations_df[locations_df["location_id"] == loc_id]["name"]
                            .iloc[0]
                            if loc_id in locations_df["location_id"].values
                            else f"Location {loc_id}"
                        )
                        st.write(
                            f"Booking ID: {highlight_id} | Location: {location_name} | "
                            f"End Time: {row['end_time']}"
                        )

                        rating_popup = st.slider(
                            "Your rating (1 = Poor, 5 = Excellent):",
                            min_value=1,
                            max_value=5,
                            value=5,
                            step=1,
                            key=f"popup_rating_{highlight_id}",
                        )
                        feedback_popup = st.text_area(
                            "Write your feedback about this parking location:",
                            "",
                            height=80,
                            key=f"popup_feedback_{highlight_id}",
                        )

                        if st.button("Submit Feedback for Last Visit"):
                            rdf = load_ratings()
                            new_rating_row = {
                                "booking_id": int(highlight_id),
                                "location_id": loc_id,
                                "phone": phone_clean,
                                "rating": float(rating_popup),
                                "feedback": feedback_popup.strip(),
                                "timestamp": datetime.now(),
                            }
                            rdf = pd.concat(
                                [rdf, pd.DataFrame([new_rating_row])],
                                ignore_index=True,
                            )
                            save_ratings(rdf)
                            st.success("Thank you! Your feedback has been saved.")
                            st.session_state["rating_booking_id"] = None


# --- Mode 5: QR Verify (Staff) ---
elif mode == "QR Verify (Staff)":
    st.subheader("QR / Booking Verification")

    bookings_df = load_bookings()
    locations_df = load_locations()

    st.markdown("### Upload QR Image")
    qr_file = st.file_uploader("Upload QR image", type=["png", "jpg", "jpeg"])

    if qr_file is not None:
        try:
            file_bytes = np.asarray(bytearray(qr_file.read()), dtype=np.uint8)
            img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(img_cv)

            if not data:
                st.error("Could not detect/decipher any QR code in the image.")
            else:
                st.success("QR code decoded successfully.")
                st.text_area("Decoded QR Text", data, height=80)
                parsed = parse_qr_payload(data)
                booking_id_str = parsed.get("BOOKING_ID")
                if booking_id_str:
                    try:
                        bid = int(booking_id_str)
                        bookings_df = load_bookings()
                        row = bookings_df[bookings_df["booking_id"] == bid]
                        if row.empty:
                            st.error("No booking found for this BOOKING_ID.")
                        else:
                            row = row.iloc[0]
                            st.write("Booking Details from QR:")
                            st.write(row)

                            now = datetime.now()
                            status = row.get("status", "active")
                            start = row["start_time"]
                            end = row["end_time"]

                            if status in ["cancelled", "no_show"]:
                                st.error(f"Booking is {status.upper()}. Not valid.")
                            elif now < start:
                                st.warning("Booking is valid but not started yet.")
                            elif now > end:
                                st.error("Booking has expired.")
                            else:
                                st.success("Booking is VALID and ACTIVE right now.")
                    except ValueError:
                        st.error("Invalid BOOKING_ID in QR data.")
        except Exception as e:
            st.error(f"Error reading QR image: {e}")

    st.markdown("---")
    st.markdown("### Verify by Booking Code (Captcha)")

    cap_input = st.text_input("Enter Booking Code (Captcha):").upper().strip()
    if st.button("Verify Booking Code"):
        if not cap_input:
            st.error("Please enter a booking code.")
        else:
            bookings_df = load_bookings()
            matches = bookings_df[
                bookings_df["captcha_code"].astype(str).str.upper() == cap_input
            ]
            if matches.empty:
                st.error("No booking found for this booking code.")
            else:
                row = matches.iloc[0]
                st.write("Booking Details:")
                st.write(row)

                now = datetime.now()
                status = row.get("status", "active")
                start = row["start_time"]
                end = row["end_time"]

                if status in ["cancelled", "no_show"]:
                    st.error(f"Booking is {status.upper()}. Not valid.")
                elif now < start:
                    st.warning("Booking is valid but not started yet.")
                elif now > end:
                    st.error("Booking has expired.")
                else:
                    st.success("Booking is VALID and ACTIVE right now.")

    st.markdown("---")
    st.markdown("### Verify by Booking ID")

    bid_manual = st.number_input("Enter Booking ID manually", min_value=1, step=1, value=1)
    if st.button("Verify Booking ID"):
        bookings_df = load_bookings()
        row = bookings_df[bookings_df["booking_id"] == bid_manual]
        if row.empty:
            st.error("No booking found for this Booking ID.")
        else:
            row = row.iloc[0]
            st.write("Booking Details:")
            st.write(row)

            now = datetime.now()
            status = row.get("status", "active")
            start = row["start_time"]
            end = row["end_time"]

            if status in ["cancelled", "no_show"]:
                st.error(f"Booking is {status.upper()}. Not valid.")
            elif now < start:
                st.warning("Booking is valid but not started yet.")
            elif now > end:
                st.error("Booking has expired.")
            else:
                st.success("Booking is VALID and ACTIVE right now.")


# --- Mode 6: Management Dashboard (Admin side) ---
elif mode == "Management Dashboard":
    st.subheader("Parking Management Dashboard")

    admin_password = st.text_input("Enter admin password:", type="password")
    if admin_password != ADMIN_PASSWORD:
        st.error("Invalid password. Access denied.")
        st.stop()

    locations_df = load_locations()
    bookings_df = load_bookings()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Refresh Data"):
            bookings_df = load_bookings()
            st.success("Data refreshed from file.")
    with col_b:
        if st.button("Rebuild Slot Status Summary"):
            update_slot_status(locations_df, bookings_df)
            st.success("Slot status summary updated.")

    st.markdown("### Locations Overview on Map")
    if not locations_df.empty:
        st.map(locations_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
        st.table(locations_df)

    st.markdown("### Filter Bookings")
    bookings_df = load_bookings()
    if bookings_df.empty:
        st.info("No bookings yet.")
    else:
        loc_options = ["All"] + locations_df["name"].tolist()
        selected_loc_name = st.selectbox("Filter by Location", loc_options)

        bookings_df["phone"] = bookings_df["phone"].astype(str).str.strip()

        min_dt = bookings_df["start_time"].min().date()
        max_dt = bookings_df["start_time"].max().date()
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            from_date = st.date_input("From date", value=min_dt)
        with col_f2:
            to_date = st.date_input("To date", value=max_dt)

        phone_filter = st.text_input("Filter by phone (full or partial):")

        # Exclude cancelled and no-show from filtered bookings table
        filtered = bookings_df[~bookings_df["status"].isin(["cancelled", "no_show"])].copy()

        if selected_loc_name != "All":
            loc_id_sel = int(
                locations_df[locations_df["name"] == selected_loc_name]["location_id"].iloc[0]
            )
            filtered = filtered[filtered["location_id"] == loc_id_sel]

        filtered = filtered[
            (filtered["start_time"].dt.date >= from_date)
            & (filtered["start_time"].dt.date <= to_date)
        ]

        if phone_filter.strip():
            pf = phone_filter.strip()
            phone_series = filtered["phone"].astype(str).str.strip()
            mask_phone = phone_series.str.contains(pf, na=False)
            filtered = filtered[mask_phone]

        st.markdown("### Filtered Bookings")
        if filtered.empty:
            st.info("No bookings match the selected filters.")
        else:
            st.dataframe(filtered.sort_values("start_time"))

            csv_data = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered bookings as CSV",
                data=csv_data,
                file_name="bookings_report.csv",
                mime="text/csv",
            )

        st.markdown("### Manage Individual Booking (Admin)")

        active_bookings = bookings_df[
            bookings_df["status"].isin(["active", "extended"])
        ]
        if active_bookings.empty:
            st.info("No active bookings available to manage.")
        else:
            active_ids = (
                active_bookings["booking_id"].dropna().astype(int).sort_values().tolist()
            )
            selected_id = st.selectbox(
                "Select active booking_id:", active_ids
            )

            row = active_bookings[active_bookings["booking_id"] == selected_id].iloc[0]
            st.write("Selected Booking Details:")
            st.write(row)

            col_m1, col_m2, col_m3 = st.columns(3)

            # --- ADMIN CANCEL ---
            with col_m1:
                if st.button("Cancel Booking (Admin)"):
                    bookings_df = load_bookings()
                    idx = bookings_df["booking_id"] == selected_id
                    bookings_df.loc[idx, "status"] = "cancelled"
                    save_bookings(bookings_df)
                    update_slot_status(locations_df, bookings_df)
                    st.success(f"Booking {selected_id} marked as cancelled.")

                    phone_for_sms = row["phone"]
                    send_sms(
                        phone_for_sms,
                        "Your booking is cancelled successfully.",
                    )
                    st.rerun()

            # --- ADMIN NO-SHOW ---
            with col_m2:
                if st.button("Mark as No-Show"):
                    bookings_df = load_bookings()
                    idx = bookings_df["booking_id"] == selected_id
                    bookings_df.loc[idx, "status"] = "no_show"
                    save_bookings(bookings_df)
                    update_slot_status(locations_df, bookings_df)
                    st.success(f"Booking {selected_id} marked as NO-SHOW.")
                    st.rerun()

            # --- ADMIN EXTEND ---
            with col_m3:
                if st.button("Extend by 1 hour (Admin)"):
                    bookings_df = load_bookings()
                    loc_row = locations_df[
                        locations_df["location_id"] == int(row["location_id"])
                    ].iloc[0]
                    old_end = row["end_time"]
                    new_end = old_end + timedelta(hours=1)

                    base_seq = X[-1]
                    available_ext, pred_ext, booked_ext = get_available_slots(
                        loc_row, old_end, new_end, model, scaler, base_seq, bookings_df
                    )
                    if available_ext <= 0:
                        st.error("Cannot extend; no slots available in extra time window.")
                    else:
                        idx = bookings_df["booking_id"] == selected_id
                        bookings_df.loc[idx, "end_time"] = new_end
                        bookings_df.loc[idx, "status"] = "extended"
                        save_bookings(bookings_df)
                        update_slot_status(locations_df, bookings_df)
                        st.success(f"Booking {selected_id} extended to {new_end}.")
                        st.rerun()

    st.markdown("### Slot Status Summary (per location & date)")
    if os.path.exists(SLOT_STATUS_FILE):
        summary_df = pd.read_csv(SLOT_STATUS_FILE)
        st.dataframe(summary_df)
        csv_summary = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download slot status summary as CSV",
            data=csv_summary,
            file_name="slot_status_summary.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Summary file not generated yet. Make a booking or click "
            "'Rebuild Slot Status Summary'."
        )

    # Cancelled bookings latest 20
    st.markdown("### Cancelled Bookings (Latest 20)")
    bookings_df = load_bookings()
    cancelled = bookings_df[bookings_df["status"] == "cancelled"].copy()
    if cancelled.empty:
        st.info("No cancelled bookings yet.")
    else:
        cancelled_sorted = cancelled.sort_values("start_time")
        if len(cancelled_sorted) > 20:
            cancelled_display = cancelled_sorted.iloc[-20:]
        else:
            cancelled_display = cancelled_sorted
        st.dataframe(cancelled_display)

    # No-show bookings
    st.markdown("### No-Show Bookings")
    no_show = bookings_df[bookings_df["status"] == "no_show"].copy()
    if no_show.empty:
        st.info("No no-show bookings yet.")
    else:
        no_show_sorted = no_show.sort_values("start_time")
        st.dataframe(no_show_sorted)

    # No-show stats (ban list)
    st.markdown("### Frequent No-Show Users (Ban List)")
    no_show_stats = get_no_show_stats(bookings_df)
    if no_show_stats.empty:
        st.info("No no-shows recorded yet.")
    else:
        banned_stats = no_show_stats[no_show_stats["no_show_count"] >= 3].copy()
        st.write("All No-Show Stats:")
        st.dataframe(no_show_stats)
        if banned_stats.empty:
            st.info("No users currently banned.")
        else:
            st.write("Users with 3 or more no-shows (banned from booking):")
            st.dataframe(banned_stats)

    # Loyalty stats
    st.markdown("### Loyalty Stats (Completed Bookings per Phone)")
    loyalty_stats = get_loyalty_stats(bookings_df)
    if loyalty_stats.empty:
        st.info("No completed bookings yet for loyalty stats.")
    else:
        loyalty_stats["membership_level"] = loyalty_stats["total_completed"].apply(
            get_membership_level
        )
        st.dataframe(loyalty_stats)

    # Waitlist table
    st.markdown("### Waitlist Entries")
    wdf = load_waitlist()
    if wdf.empty:
        st.info("No users are currently on the waitlist.")
    else:
        wdf_merged = wdf.merge(
            locations_df[["location_id", "name"]],
            on="location_id",
            how="left",
        ).rename(columns={"name": "location_name"})
        wdf_merged = wdf_merged.sort_values("timestamp")
        st.dataframe(wdf_merged)

    # Ratings & feedback
    st.markdown("### Ratings & Feedback")
    rdf = load_ratings()
    if rdf.empty:
        st.info("No ratings submitted yet.")
    else:
        rdf_merged = rdf.merge(
            locations_df[["location_id", "name"]],
            on="location_id",
            how="left",
        ).rename(columns={"name": "location_name"})

        st.markdown("**Average Rating per Location**")
        avg_rating = (
            rdf_merged.groupby("location_name")["rating"]
            .mean()
            .round(2)
            .reset_index()
            .sort_values("rating", ascending=False)
        )
        st.dataframe(avg_rating)

        st.markdown("**Latest Feedback Entries**")
        latest_feedback = rdf_merged.sort_values("timestamp", ascending=False).head(20)
        st.dataframe(
            latest_feedback[
                ["timestamp", "location_name", "phone", "rating", "feedback"]
            ]
        )

    # Analytics
    st.markdown("### Analytics (Demand & Usage)")

    bookings_df = load_bookings()
    if bookings_df.empty:
        st.info("Not enough data for analytics yet.")
    else:
        # --- Bookings per Hour ---
        fig1, ax1 = plt.subplots()
        hourly = (
            bookings_df.assign(hour=bookings_df["start_time"].dt.hour)
            .groupby("hour")["booking_id"]
            .count()
        )
        hourly_values = hourly.values.astype(int)  # ensure integers
        ax1.bar(hourly.index, hourly_values)
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Number of Bookings")
        ax1.set_title("Bookings per Hour")
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # <<< integer ticks
        st.pyplot(fig1)

        # --- Bookings per Day ---
        fig2, ax2 = plt.subplots()
        daily = (
            bookings_df.assign(date=bookings_df["start_time"].dt.date)
            .groupby("date")["booking_id"]
            .count()
        )
        daily_values = daily.values.astype(int)
        ax2.plot(daily.index, daily_values, marker="o")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Number of Bookings")
        ax2.set_title("Bookings per Day")
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # <<< integer ticks
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # --- Bookings per Location ---
        fig3, ax3 = plt.subplots()
        merged = bookings_df.merge(
            locations_df[["location_id", "name"]],
            on="location_id",
            how="left",
        )
        loc_counts = merged.groupby("name")["booking_id"].count().sort_values(ascending=False)
        loc_values = loc_counts.values.astype(int)
        ax3.barh(loc_counts.index, loc_values)
        ax3.set_xlabel("Number of Bookings")
        ax3.set_ylabel("Location")
        ax3.set_title("Bookings per Location")
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))  # <<< integer ticks
        st.pyplot(fig3)
