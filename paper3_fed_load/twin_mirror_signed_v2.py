# twin_mirror_signed.py -- the in-process simulator and the Kafka/MQTT
# production mirror, both under the signed nonconformity score.
#
# Replaces two notebook cells:
#   (a) "Simulator: the twin loop in-process, hour by hour"
#   (b) "Production mirror v7"
#
#   exec(open(f"{PROJECT_DIR}/twin_mirror_signed.py").read())
#
# Run it through run_twin_demo.py, which supplies OUT_DIR, K, z, SEED_H,
# WINDOW, GAMMA, ALPHA_STAR and n_days, starts Mosquitto and Kafka, and resets
# the topics. Publishes to the same kpi/intervals topics, so the dashboard
# needs no change.
#
# SIGNED and N_SEASONS below are bare assignments at the start of their lines
# because the runner rewrites them by regex and asserts the match. Keep them
# that way.
#
# WHAT CHANGED, and nothing else did: every ACL state now holds two score
# windows instead of one, the radius is the (1 - alpha/2) quantile of each,
# and the residual keeps its sign. Pacing, seasons, the bandit, the tariff,
# checkpointing and the published fields are untouched, so the mirror against
# simulator comparison at the end is like for like.
#
# Set SIGNED = False to reproduce the original symmetric run exactly.

import json, time, threading, pickle, os
import numpy as np
import paho.mqtt.client as mqtt
from kafka import KafkaProducer, KafkaConsumer

SIGNED = True
SHOW_K = 0

# The runner supplies these. Fail here rather than 200 lines into the loop.
for _n in ("OUT_DIR", "K", "z", "n_days", "SEED_H", "WINDOW", "GAMMA", "ALPHA_STAR"):
    assert _n in globals(), f"{_n} not set; run this through run_twin_demo.py"


def _q(w, p):
    return float(np.quantile(np.asarray(w), min(0.999, max(0.001, p))))


def seed_state(y0, p0):
    """clean signed residuals, trimmed to the window"""
    e = y0 - p0
    e = e[np.isfinite(e)][:WINDOW]
    if SIGNED:
        return {"alpha": ALPHA_STAR, "up": list(e), "lo": list(-e)}
    return {"alpha": ALPHA_STAR, "up": list(np.abs(e)), "lo": list(np.abs(e))}


def radii(st):
    """(r_lo, r_up) from a state dict, under whichever score is active"""
    a = float(np.clip(st["alpha"], .001, .999))
    if SIGNED:
        return _q(st["lo"], 1 - a / 2), _q(st["up"], 1 - a / 2)
    r = _q(st["up"], 1 - a)
    return r, r


def push(st, e):
    """append one signed residual to the window(s)"""
    if SIGNED:
        st["up"].append(e); st["lo"].append(-e)
        st["up"] = st["up"][-WINDOW:]; st["lo"] = st["lo"][-WINDOW:]
    else:
        st["up"].append(abs(e)); st["up"] = st["up"][-WINDOW:]
        st["lo"] = st["up"]


TAG = "signed" if SIGNED else "symmetric"
print(f"twin mirror, {TAG} score")

# The per-building streams. This line used to sit at the top of the simulator
# cell; it belongs here now that the cell is gone, so the file stands alone.
data = [(z[f"y0_{k}"], z[f"p0_{k}"], z[f"y1_{k}"], z[f"p1_{k}"])
        for k in range(K)]


# ======================================================================
# (a) in-process simulator: the baseline the mirror must reproduce
# ======================================================================
event_day = n_days // 2
state = [seed_state(y0, p0) for y0, p0, _, _ in data]

picp = np.zeros(n_days); width = np.zeros(n_days)
alpha_h = np.zeros(n_days); r_h = np.zeros(n_days)
sy = np.full(n_days * 24, np.nan)
slo = np.full(n_days * 24, np.nan)
sup = np.full(n_days * 24, np.nan)

for d in range(n_days):
    covs, wids = [], []
    for k, (y0, p0, y1, p1) in enumerate(data):
        y, p = (y0, p0) if d < event_day else (y1, p1)
        h0 = SEED_H + d * 24
        st = state[k]
        r_lo, r_up = radii(st)
        yy, pp = y[h0:h0 + 24], p[h0:h0 + 24]
        m = np.isfinite(yy) & np.isfinite(pp)
        if m.sum():
            e = yy[m] - pp[m]
            covs.append(np.mean((-r_lo <= e) & (e <= r_up)))
            wids.append(r_lo + r_up)
        if k == SHOW_K:
            idx = np.arange(d * 24, d * 24 + 24)
            sy[idx[m]] = yy[m]
            slo[idx[m]] = pp[m] - r_lo
            sup[idx[m]] = pp[m] + r_up
            alpha_h[d] = st["alpha"]; r_h[d] = 0.5 * (r_lo + r_up)
        for e in (yy[m] - pp[m]):
            err = 0.0 if (-r_lo <= e <= r_up) else 1.0
            st["alpha"] = float(np.clip(st["alpha"] + GAMMA * (ALPHA_STAR - err),
                                        .001, .999))
            push(st, e)
    picp[d] = np.mean(covs); width[d] = np.mean(wids)

r7 = np.array([np.nanmean(picp[max(0, i - 6):i + 1]) for i in range(n_days)])
print(f"simulated {n_days} days; event at day {event_day}")
print(f"fleet PICP pre-event={np.nanmean(picp[:event_day]):.2f}  "
      f"post-event={np.nanmean(picp[event_day+7:]):.2f}")
print(f"steady-state width (last 30 days) = {np.nanmean(width[-30:]):.1f} kW")


# ======================================================================
# (b) production mirror
# ======================================================================
N_SEASONS = 1            # 1 = single real test season, the release default
DAY_PACE = 0.5 if N_SEASONS == 1 else 0.25
BASE_DAYS = n_days
n_days = BASE_DAYS * N_SEASONS
DEGRADED = lambda d: (d % BASE_DAYS) >= (BASE_DAYS // 2)      # noqa: E731
RUN_ID = str(int(time.time()))
C_H = 0.12                          # USD/kWh, EIA 2024 commercial average
RATIOS = {True: 19.0, False: 4.0}   # peak, off-peak c_p/c_h
# eq. (13) gives alpha* = 2/(r+1): 0.10 at peak, 0.40 off-peak. Both sit in
# ARMS, so the bandit is able to reach them.
ARMS = [0.05, 0.10, 0.20, 0.30, 0.40]
UCB_C = 0.3
PEAKH = lambda i: 8 <= i < 20                                  # noqa: E731
CKPT = os.path.join(OUT_DIR, "acl_state_ckpt.pkl")
print(f"mirror run {RUN_ID}, {TAG} score: {n_days} days "
      f"({N_SEASONS} season(s) x {BASE_DAYS}) x {K} buildings")


def producer():
    c = mqtt.Client(); c.connect("localhost", 1883); c.loop_start()
    for d in range(n_days):
        h0 = SEED_H + (d % BASE_DAYS) * 24
        for k, (y0, p0, y1, p1) in enumerate(data):
            y, p = (y1, p1) if DEGRADED(d) else (y0, p0)
            c.publish(f"meters/{k}", json.dumps(
                {"k": k, "d": d,
                 "y": [float(v) if np.isfinite(v) else None for v in y[h0:h0+24]],
                 "p": [float(v) if np.isfinite(v) else None for v in p[h0:h0+24]]}),
                qos=1)
        time.sleep(DAY_PACE)
    time.sleep(3); c.publish("meters/EOF", RUN_ID, qos=1); time.sleep(2)
    c.loop_stop()


done = threading.Event(); ready = threading.Event()


def bridge():
    kp = KafkaProducer(bootstrap_servers="localhost:9092",
                       key_serializer=lambda k: k.encode(),
                       value_serializer=lambda v: v.encode())

    def on_msg(cl, u, m):
        if m.topic.endswith("EOF"):
            kp.send("telemetry", key="EOF", value=m.payload.decode()); kp.flush()
            done.set(); return
        kp.send("telemetry", key=m.topic.split("/")[1], value=m.payload.decode())

    c = mqtt.Client(); c.on_message = on_msg
    c.connect("localhost", 1883); c.subscribe("meters/#", qos=1); c.loop_start()
    time.sleep(1); ready.set()
    done.wait(); c.loop_stop()


threading.Thread(target=bridge, daemon=True).start()
ready.wait()
threading.Thread(target=producer, daemon=True).start()


class Bandit:
    def __init__(self):
        self.A = {(p, a): 1.0 for p in (True, False) for a in range(len(ARMS))}
        self.b = {(p, a): 0.0 for p in (True, False) for a in range(len(ARMS))}
        self.pending = {}

    def pick(self, d, p):
        best, arg = -1e18, 0
        for a in range(len(ARMS)):
            u = (self.b[(p, a)] / self.A[(p, a)]
                 + UCB_C * np.sqrt(np.log(max(d, 2)) / self.A[(p, a)]))
            if u > best: best, arg = u, a
        self.pending[p] = arg
        return ARMS[arg]

    def reward(self, p, bill):
        a = self.pending[p]
        self.A[(p, a)] += 1.0
        self.b[(p, a)] += -bill / (K * 30.0)


# keyed ACL state: newsvendor stream, two bandit streams, and the frozen pair
nv = {}; bd = {}; f_lo = {}; f_up = {}
for k, (y0, p0, _, _) in enumerate(data):
    nv[k] = seed_state(y0, p0)
    for seg in (True, False):
        bd[(k, seg)] = seed_state(y0, p0)
    e = y0 - p0; e = e[np.isfinite(e)][:WINDOW]
    n = max(len(e), 1)
    if SIGNED:
        f_up[k] = _q(e, min(0.999, (1 - ALPHA_STAR / 2) * (1 + 1 / n)))
        f_lo[k] = _q(-e, min(0.999, (1 - ALPHA_STAR / 2) * (1 + 1 / n)))
    else:
        f_up[k] = f_lo[k] = _q(np.abs(e), min(0.999, (1 - ALPHA_STAR) * (1 + 1 / n)))

B = Bandit()
tgt_by_day = {0: {True: B.pick(0, True), False: B.pick(0, False)}}


def get_tgt(d):
    return tgt_by_day[max(dd for dd in tgt_by_day if dd <= d)]


cd_f = {}; cd_nv = {}; cd_bd = {}; rd_nv = {}; rd_bd = {}; billday = {}
closed = {}; nmsg = 0
picp_nv = np.full(n_days, np.nan); t0 = time.time()

# ---- latency instrumentation -------------------------------------------
# Three costs are separated so the paper can attribute the number it quotes.
#   ns_sizing  the morning quantile evaluation, once per building per day
#   ns_update  the evening loop: eq. (16) plus the window push, per reading
#   ns_total   everything the consumer does with one message
# The evening loop is timed once per message and divided by the readings it
# processed, since timing each iteration separately would cost more than the
# iteration does.
ns_sizing = []; ns_update = []; ns_total = []; n_read = 0

kp_out = KafkaProducer(bootstrap_servers="localhost:9092",
                       value_serializer=lambda v: json.dumps(v).encode())
cons = KafkaConsumer("telemetry", bootstrap_servers="localhost:9092",
                     auto_offset_reset="earliest", consumer_timeout_ms=300000)

for msg in cons:
    if msg.key == b"EOF":
        continue
    _t_msg = time.perf_counter_ns()
    r = json.loads(msg.value); k, d = r["k"], r["d"]; nmsg += 1
    tgt = get_tgt(d)

    _t = time.perf_counter_ns()
    nv_lo, nv_up = radii(nv[k])
    bd_r = {seg: radii(bd[(k, seg)]) for seg in (True, False)}
    ns_sizing.append(time.perf_counter_ns() - _t)

    lo_nv = []; up_nv = []; lo_bd = []; up_bd = []
    bday = billday.setdefault(d, {True: 0.0, False: 0.0})
    for i, (y, p) in enumerate(zip(r["y"], r["p"])):
        seg = PEAKH(i); b_lo, b_up = bd_r[seg]
        if y is None or p is None:
            lo_nv.append(None); up_nv.append(None)
            lo_bd.append(None); up_bd.append(None); continue
        lo_nv.append(p - nv_lo); up_nv.append(p + nv_up)
        lo_bd.append(p - b_lo); up_bd.append(p + b_up)
        e = y - p                                   # SIGNED
        cd_f.setdefault(d, []).append((-f_lo[k] <= e) and (e <= f_up[k]))
        cd_nv.setdefault(d, []).append((-nv_lo <= e) and (e <= nv_up))
        cd_bd.setdefault(d, []).append((-b_lo <= e) and (e <= b_up))
        rd_nv.setdefault(d, []).append(nv_lo + nv_up)
        rd_bd.setdefault(d, []).append(b_lo + b_up)
        # the bill charges only the upper breach, as in eq. (10)
        bday[seg] += (C_H * max(b_up - e, 0)
                      + C_H * RATIOS[seg] * max(e - b_up, 0))

    kp_out.send("intervals", {"k": k, "d": d, "y": r["y"],
                              "lo": lo_nv, "up": up_nv,
                              "lo_b": lo_bd, "up_b": up_bd})

    _t = time.perf_counter_ns(); _nread = 0
    for i, (y, p) in enumerate(zip(r["y"], r["p"])):
        if y is None or p is None: continue
        _nread += 1
        e = y - p; seg = PEAKH(i)
        err = 0.0 if (-nv_lo <= e <= nv_up) else 1.0
        nv[k]["alpha"] = float(np.clip(nv[k]["alpha"] + GAMMA*(ALPHA_STAR - err),
                                       .001, .999))
        push(nv[k], e)
        b_lo, b_up = bd_r[seg]
        errb = 0.0 if (-b_lo <= e <= b_up) else 1.0
        st = bd[(k, seg)]
        st["alpha"] = float(np.clip(st["alpha"] + GAMMA*(tgt[seg] - errb),
                                    .001, .999))
        push(st, e)
    if _nread:
        ns_update.append((time.perf_counter_ns() - _t) / _nread)
        n_read += _nread
    ns_total.append(time.perf_counter_ns() - _t_msg)

    closed[d] = closed.get(d, 0) + 1
    if closed[d] == K:
        closed.pop(d); bday = billday.pop(d)
        picp_nv[d] = np.mean(cd_nv.pop(d, [np.nan]))
        for seg in (True, False): B.reward(seg, bday[seg])
        tgt_by_day[d+1] = {True: B.pick(d+1, True), False: B.pick(d+1, False)}
        alphas = [float(nv[k2]["alpha"]) for k2 in range(K)]
        kp_out.send("kpi", {"d": d,
                            "picp_f": float(np.mean(cd_f.pop(d, [np.nan]))),
                            "picp": float(picp_nv[d]),
                            "picp_b": float(np.mean(cd_bd.pop(d, [np.nan]))),
                            "width": float(np.mean(rd_nv.pop(d, [np.nan]))),
                            "width_b": float(np.mean(rd_bd.pop(d, [np.nan]))),
                            "width_f": float(np.mean([f_lo[k2] + f_up[k2]
                                                      for k2 in range(K)])),
                            "alpha": float(np.mean(alphas)), "alphas": alphas,
                            "tgt_peak": get_tgt(d)[True],
                            "tgt_off": get_tgt(d)[False],
                            "bill": float(bday[True] + bday[False]),
                            "event": int(DEGRADED(d)),
                            "nmsg": int(nmsg), "exp": int(n_days*K),
                            "ckpt": int(d - d % 10),
                            "nd": int(n_days),
                            "season": int(d // BASE_DAYS) + 1,
                            "nseasons": int(N_SEASONS)})
        if d % 10 == 0:
            with open(CKPT, "wb") as f:
                pickle.dump({"nv": nv, "bd": bd, "signed": SIGNED}, f)
            print(f"day {d:3d}  NV PICP={picp_nv[d]:.2f}  "
                  f"bandit tgt={get_tgt(d)}  [{time.time()-t0:.0f}s]")
        if nmsg >= n_days*K: break

print(f"\nstream complete, {time.time()-t0:.0f}s wall clock")
print(f"messages received {nmsg} / expected {n_days*K}")
print(f"unclosed days: {int(np.isnan(picp_nv).sum())}")
_gap = float(np.nanmax(np.abs(picp_nv[:BASE_DAYS] - picp)))
print(f"max |mirror - simulator| daily fleet PICP (season 1): {_gap:.4f}")
print(f"final learned targets: peak={get_tgt(n_days-1)[True]}, "
      f"off={get_tgt(n_days-1)[False]}")
if _gap > 1e-3:
    print("\nThat gap is larger than the 4e-4 the paper reports. The two paths")
    print("should be arithmetically identical, so a gap means the mirror and")
    print("the simulator are not running the same score. Check SIGNED.")


# ---- latency ------------------------------------------------------------
# Reported as medians. The mean is pulled around by Colab scheduling and by
# the Kafka consumer's fetch boundaries, which are not calibration work.
def _us(a):
    return float(np.median(a)) / 1000.0


_sizing = _us(ns_sizing)          # per building-day
_update = _us(ns_update)          # per reading
_total = _us(ns_total)            # per message
_calib = _sizing / 24.0 + _update  # calibration work amortized per reading

print(f"\nlatency on {nmsg} messages, {n_read} readings ({TAG} score)")
print(f"  interval sizing, per building-day   {_sizing:8.1f} us")
print(f"  state update eq.(16) + window push  {_update:8.1f} us  per reading")
print(f"  calibration total                   {_calib:8.1f} us  per reading")
print(f"  whole consumer path                 {_total/24.0:8.1f} us  per reading")
print(f"                                      {_total:8.1f} us  per message")
print("\nSection IV-G quotes two of these: 26 us for the state update and")
print("330 us for the whole consumer path, both per reading. The sizing cost")
print("is the quantile evaluation, and is the term the signed score doubles,")
print("since it takes a quantile from each of two windows.")

# Read by run_twin_demo.py for the comparison table.
RESULTS = {"update_us": _update, "sizing_us": _sizing, "calib_us": _calib,
           "path_us": _total / 24.0, "gap": _gap,
           "tgt_peak": get_tgt(n_days - 1)[True],
           "tgt_off": get_tgt(n_days - 1)[False]}
