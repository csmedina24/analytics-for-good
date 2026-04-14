# ══════════════════════════════════════════════
# SAVED: LIVE MUNI TRACKER TAB
# Removed from app.py on 2025-04-14. Paste back into app.py to restore.
# Requires: tab4 defined in st.tabs(), get_muni_routes(), get_live_arrivals(),
#           API_KEY_511, PURPLE, RED, AMBER constants, datetime import
# ══════════════════════════════════════════════

# --- Helper functions (place near top of app.py) ---

# @st.cache_data(ttl=120)
# def get_muni_routes():
#     """Fetch list of all Muni routes from 511 API."""
#     url = (f"https://api.511.org/transit/lines?api_key={API_KEY_511}"
#            f"&operator_id=SF&format=json")
#     try:
#         req = urllib.request.Request(url)
#         with urllib.request.urlopen(req, timeout=15) as resp:
#             raw = resp.read().decode("utf-8-sig")
#         data = json.loads(raw)
#         routes = {r["Id"]: r["Name"] for r in data}
#         return routes
#     except Exception:
#         return {}
#
#
# @st.cache_data(ttl=60)
# def get_live_arrivals(route_id):
#     """Fetch live StopMonitoring data for a specific route."""
#     url = (f"https://api.511.org/transit/StopMonitoring?api_key={API_KEY_511}"
#            f"&agency=SF&format=json")
#     try:
#         req = urllib.request.Request(url)
#         with urllib.request.urlopen(req, timeout=20) as resp:
#             raw = resp.read().decode("utf-8-sig")
#         data = json.loads(raw)
#         delivery = data.get("ServiceDelivery", {}).get("StopMonitoringDelivery", {})
#         if isinstance(delivery, list):
#             visits = delivery[0].get("MonitoredStopVisit", []) if delivery else []
#         else:
#             visits = delivery.get("MonitoredStopVisit", [])
#
#         results = []
#         for visit in visits:
#             j = visit.get("MonitoredVehicleJourney", {})
#             if j.get("LineRef", "") != route_id:
#                 continue
#             call = j.get("MonitoredCall", {})
#             loc = j.get("VehicleLocation", {})
#
#             aimed = call.get("AimedArrivalTime", "")
#             expected = call.get("ExpectedArrivalTime", "")
#
#             delay = None
#             if aimed and expected:
#                 try:
#                     a = datetime.fromisoformat(aimed.replace("Z", "+00:00"))
#                     e = datetime.fromisoformat(expected.replace("Z", "+00:00"))
#                     delay = int((e - a).total_seconds())
#                 except Exception:
#                     pass
#
#             results.append({
#                 "stop": call.get("StopPointName", "Unknown"),
#                 "direction": j.get("DirectionRef", ""),
#                 "vehicle": j.get("VehicleRef", ""),
#                 "aimed": aimed,
#                 "expected": expected,
#                 "delay_sec": delay,
#                 "lat": float(loc.get("Latitude", 0) or 0),
#                 "lon": float(loc.get("Longitude", 0) or 0),
#                 "occupancy": j.get("Occupancy", ""),
#             })
#         return results
#     except Exception as e:
#         return []

# --- Tab content (place inside "with tab4:") ---

# with tab4:
#     st.subheader("Live Muni Tracker")
#     st.markdown("Real-time bus arrival data from the [511 SF Bay API](https://511.org/open-data). "
#                 "Select a route to see current delays and reliability.")
#
#     routes = get_muni_routes()
#
#     if not routes:
#         st.error("Could not load Muni routes from 511 API. Try refreshing.")
#     else:
#         # Sort routes: numeric first, then alphanumeric
#         def route_sort_key(r):
#             num = "".join(c for c in r if c.isdigit())
#             return (int(num) if num else 999, r)
#         sorted_routes = sorted(routes.keys(), key=route_sort_key)
#
#         route_options = [f"{r} - {routes[r]}" for r in sorted_routes]
#
#         # Default to route 49 (Van Ness)
#         default_idx = next((i for i, r in enumerate(sorted_routes) if r == "49"), 0)
#         selected = st.selectbox("Select a Muni route:", route_options, index=default_idx)
#         route_id = selected.split(" - ")[0]
#
#         if st.button("Refresh Data", type="primary"):
#             st.cache_data.clear()
#
#         with st.spinner(f"Fetching live data for Route {route_id}..."):
#             arrivals = get_live_arrivals(route_id)
#
#         if not arrivals:
#             st.warning(f"No active vehicles found for Route {route_id}. "
#                        "The line may not be running right now.")
#         else:
#             df_live = pd.DataFrame(arrivals)
#
#             # Summary metrics
#             valid_delays = df_live["delay_sec"].dropna()
#             col1, col2, col3, col4 = st.columns(4)
#
#             with col1:
#                 st.metric("Active Vehicles",
#                           f"{df_live['vehicle'].nunique()}")
#             with col2:
#                 st.metric("Stops Served",
#                           f"{df_live['stop'].nunique()}")
#             with col3:
#                 if len(valid_delays) > 0:
#                     avg_delay = valid_delays.mean()
#                     label = "early" if avg_delay < 0 else "late"
#                     st.metric("Avg Delay", f"{abs(avg_delay):.0f}s {label}")
#                 else:
#                     st.metric("Avg Delay", "N/A")
#             with col4:
#                 if len(valid_delays) > 0:
#                     pct_late = (valid_delays > 0).mean() * 100
#                     st.metric("% Running Late", f"{pct_late:.0f}%")
#                 else:
#                     st.metric("% Running Late", "N/A")
#
#             st.markdown("---")
#
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 # Delay distribution
#                 if len(valid_delays) > 0:
#                     fig, ax = plt.subplots(figsize=(7, 4))
#                     delay_mins = valid_delays / 60
#                     ax.hist(delay_mins, bins=30, color=PURPLE, alpha=0.8,
#                             edgecolor="white")
#                     ax.axvline(0, color=RED, linewidth=2, linestyle="--",
#                                label="On time")
#                     ax.axvline(delay_mins.mean(), color=AMBER, linewidth=2,
#                                linestyle="-", label=f"Avg: {delay_mins.mean():.1f} min")
#                     ax.set_xlabel("Delay (minutes) -- negative = early")
#                     ax.set_ylabel("Count")
#                     ax.set_title(f"Route {route_id}: Current Delay Distribution",
#                                  fontweight="bold")
#                     ax.legend(frameon=True)
#                     plt.tight_layout()
#                     st.pyplot(fig)
#                     plt.close()
#
#             with col2:
#                 # Map of vehicle positions
#                 map_data = df_live[df_live["lat"] != 0].copy()
#                 if len(map_data) > 0:
#                     # Deduplicate by vehicle
#                     map_data = map_data.drop_duplicates(subset=["vehicle"])
#                     map_data = map_data.rename(columns={"lat": "latitude",
#                                                          "lon": "longitude"})
#                     st.map(map_data[["latitude", "longitude"]], zoom=12)
#
#             st.markdown("---")
#
#             # Delays by stop
#             st.markdown("**Delays by Stop**")
#             stop_delays = df_live.dropna(subset=["delay_sec"]).groupby("stop").agg(
#                 avg_delay=("delay_sec", "mean"),
#                 count=("delay_sec", "size")
#             ).round(0).sort_values("avg_delay", ascending=False)
#
#             stop_delays["avg_delay_min"] = (stop_delays["avg_delay"] / 60).round(1)
#             stop_delays["status"] = stop_delays["avg_delay"].apply(
#                 lambda x: "Late" if x > 60 else "On Time" if x > -60 else "Early"
#             )
#
#             display_stops = stop_delays[["avg_delay_min", "count", "status"]].copy()
#             display_stops.columns = ["Avg Delay (min)", "Predictions", "Status"]
#             st.dataframe(display_stops, use_container_width=True)
#
#             ts = datetime.now().strftime("%I:%M %p")
#             st.caption(f"Data as of {ts} | Source: 511 SF Bay API | "
#                        "Negative delay = ahead of schedule")
