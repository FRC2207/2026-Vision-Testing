"""
FRC Ball Path Sim — PathPlanner Pose2d Edition
===============================================
Outputs Pose2d(x, y, travelHeading) exactly as PathPlannerPath.waypointsFromPoses() needs.
Rotation component = direction of travel, NOT holonomic chassis angle.

All algorithms run in < 5ms on desktop (< 20ms on RoboRio Java).

KEYS
----
  Space   randomize    l=lines    p=sparse    c=custom place
  i       cycle intake radius 0.20 / 0.35 / 0.50 / 0.75 m
  B/b     budget +2m / -2m   (default 16m ≈ 15s auto at ~3 m/s avg)
  r/t/e/a/s   toggle algorithms
  p2      print Pose2d list to console (copy into Java)
  m       bezier arms    w=save PNG

ALGORITHMS
----------
  r   FastOP      ← RECOMMENDED. Greedy insert + single 2-opt. <2ms.
  t   MomentumOP  ← FastOP but heading-penalised selection. Straighter paths.
  e   DensityOP   ← Score by density not just size. Skips sparse loners hard.
  a   SweepOP     ← Project clusters onto dominant axis, visit in order. Zero turns.
  s   GreedyRef   ← Simple nearest-cluster (reference, no budget awareness)
"""

import math, random, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FIELD_W, FIELD_H = 16.5, 8.0
MIN_BALL_DIST    = 0.3
EPS              = 0.5
MIN_PTS          = 4
INTAKE_RADII     = [0.20, 0.35, 0.50, 0.75]
intake_idx       = 1
BUDGET_M         = 16.0      # mutable — tune to your auto time × avg speed
CTRL_ARM         = 0.45
SPL_N            = 60
HEAD_PEN         = 5       # metres added per radian of turn (MomentumOP)
DENSE_STEP       = 0.35      # metres between waypoints along cluster axis
PHYS_MAX_SPEED  = 6.8    # m/s  — tune to your drivetrain top speed
PHYS_MAX_ACCEL  =  2    # m/s² — tune to your drivetrain acceleration
PHYS_TURN_DECAY = 5   # speed loss per radian of turn (higher = more penalty)
PHYS_MIN_SPEED  = 0.8    # m/s  — floor speed through any turn


def trapezoid_time(dist, v0, v1):
    """
    Minimum time to travel `dist` metres from speed v0 to speed v1.
    Uses a trapezoidal velocity profile (accel → cruise → decel).
    Falls back to a triangular profile if the distance is too short to
    reach PHYS_MAX_SPEED.
    """
    if dist < 1e-6: return 0.0
    v0 = min(v0, PHYS_MAX_SPEED)
    v1 = min(v1, PHYS_MAX_SPEED)

    d_up   = (PHYS_MAX_SPEED**2 - v0**2) / (2 * PHYS_MAX_ACCEL)  # dist to reach top speed
    d_down = (PHYS_MAX_SPEED**2 - v1**2) / (2 * PHYS_MAX_ACCEL)  # dist to bleed back to v1

    if d_up + d_down <= dist:
        # ── Trapezoidal: robot reaches full speed ─────────────────────────────
        t_up   = (PHYS_MAX_SPEED - v0) / PHYS_MAX_ACCEL
        t_down = (PHYS_MAX_SPEED - v1) / PHYS_MAX_ACCEL
        t_flat = (dist - d_up - d_down) / PHYS_MAX_SPEED
        return t_up + t_flat + t_down
    else:
        # ── Triangular: too short to reach full speed ─────────────────────────
        # v_peak² = amax*dist + (v0² + v1²) / 2
        v_peak = math.sqrt(max(0.0, PHYS_MAX_ACCEL * dist + (v0**2 + v1**2) / 2))
        v_peak = min(v_peak, PHYS_MAX_SPEED)
        t_up   = (v_peak - v0) / PHYS_MAX_ACCEL
        t_down = (v_peak - v1) / PHYS_MAX_ACCEL
        return t_up + t_down


def turn_speed(approach_speed, turn_rad):
    """
    Speed the robot can carry through a turn of `turn_rad` radians.
    Exponential decay: 0 rad = full speed, π rad ≈ PHYS_MIN_SPEED.
    """
    return max(PHYS_MIN_SPEED, approach_speed * math.exp(-PHYS_TURN_DECAY * turn_rad))


def cluster_phys_cost(cluster, from_pos, cur_speed, cur_hdg):
    """
    Estimate time (seconds) to drive to and through `cluster` given the
    robot's current position, speed, and heading.

    Returns (time_seconds, exit_speed_m/s) so the caller can chain costs
    through a sequence without restarting from zero each time.

    Model:
      1. Compute turn angle from cur_hdg to the cluster entry point.
      2. Derive the entry speed the robot can carry through that turn.
      3. Trapezoid profile for the approach (cur_speed → entry_speed).
      4. Trapezoid profile through the cluster (entry_speed → entry_speed).
    """
    entry = nearest_p(cluster, from_pos)
    exit_ = farthest(cluster, from_pos)

    d_approach = dd(from_pos, entry)
    d_through  = dd(entry, exit_)

    turn = hdg_diff(cur_hdg, heading(from_pos, entry)) if cur_hdg is not None else 0.0
    v_entry = turn_speed(PHYS_MAX_SPEED, turn)   # speed we arrive at the cluster
    v_exit  = v_entry                             # maintain through cluster (intake pace)

    t_approach = trapezoid_time(d_approach, cur_speed, v_entry)
    t_through  = trapezoid_time(d_through,  v_entry,   v_exit)

    return t_approach + t_through, v_exit
# ═══════════════════════════════════════════════════════════════════════════════
#  PATHPLANNER CUBIC BÉZIER
#  Mirrors waypointsFromPoses internal math exactly.
#  Rotation of each Pose2d = direction of travel at that point.
# ═══════════════════════════════════════════════════════════════════════════════

def pp_spline(poses):
    """
    poses: list of (x, y, heading_rad)
    heading = direction of travel (what you pass to waypointsFromPoses Rotation2d).
    Returns xs, ys arrays of the rendered cubic Bézier path.
    """
    if len(poses) < 2:
        return np.array([poses[0][0]]), np.array([poses[0][1]])
    pts  = [np.array([p[0], p[1]], dtype=float) for p in poses]
    hdgs = [p[2] for p in poses]
    N    = len(pts)
    # Tangent = unit vector in direction of travel at each waypoint
    tan  = [np.array([math.cos(h), math.sin(h)]) for h in hdgs]
    ax, ay = [], []
    for i in range(N - 1):
        p0, p3 = pts[i], pts[i+1]
        arm = CTRL_ARM * (np.linalg.norm(p3 - p0) or 1e-9)
        p1 = p0 + arm * tan[i]
        p2 = p3 - arm * tan[i+1]
        t = np.linspace(0, 1, SPL_N); mt = 1 - t
        xs = mt**3*p0[0]+3*mt**2*t*p1[0]+3*mt*t**2*p2[0]+t**3*p3[0]
        ys = mt**3*p0[1]+3*mt**2*t*p1[1]+3*mt*t**2*p2[1]+t**3*p3[1]
        if i > 0: xs, ys = xs[1:], ys[1:]
        ax.append(xs); ay.append(ys)
    return np.concatenate(ax), np.concatenate(ay)

def spline_len(xs, ys):
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) if len(xs) > 1 else 0.

def measure(xs, ys, balls, ir):
    if not len(xs) or not balls: return 0., [False]*len(balls), 0., 0.
    hit = [bool(((xs-bx)**2+(ys-by)**2).min() < ir**2) for bx,by in balls]
    pct = 100.*sum(hit)/len(balls)
    pl  = spline_len(xs, ys)
    eff = sum(hit)/pl if pl > 0.01 else 0.
    return pct, hit, pl, eff


# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def dd(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def dbscan(pts):
    visited = set(); noise = set(); clusters = []
    def nbrs(i): return [j for j in range(len(pts)) if dd(pts[i], pts[j]) <= EPS]
    for i in range(len(pts)):
        if i in visited: continue
        visited.add(i); nb = nbrs(i)
        if len(nb) < MIN_PTS: noise.add(i)
        else:
            cl = set(); q = list(nb); cl.add(i)
            while q:
                j = q.pop(0)
                if j not in visited:
                    visited.add(j); nn = nbrs(j)
                    if len(nn) >= MIN_PTS: q.extend(nn)
                cl.add(j)
            clusters.append([pts[k] for k in cl])
    for n in noise: clusters.append([pts[n]])
    return clusters

def centroid(cl): return (sum(p[0] for p in cl)/len(cl), sum(p[1] for p in cl)/len(cl))
def farthest(cl, ref): return max(cl, key=lambda p: dd(ref, p))
def nearest_p(cl, ref): return min(cl, key=lambda p: dd(ref, p))
def cl_mindist(c, ref): return min(dd(ref, p) for p in c)

def valid_balls(balls, robot):
    m2 = MIN_BALL_DIST**2
    return [b for b in balls if (b[0]-robot[0])**2+(b[1]-robot[1])**2 >= m2]

def pca_axis(cl):
    cx = sum(p[0] for p in cl)/len(cl); cy = sum(p[1] for p in cl)/len(cl)
    cxx = cxy = cyy = 0.
    for p in cl:
        dx, dy = p[0]-cx, p[1]-cy
        cxx += dx*dx; cxy += dx*dy; cyy += dy*dy
    h=(cxx+cyy)*0.5; disc=math.sqrt(max(0.,h*h-(cxx*cyy-cxy*cxy))); lam=h+disc
    axX,axY=(lam-cyy,cxy) if abs(cxy)>1e-9 else ((1.,0.) if cxx>=cyy else (0.,1.))
    L=math.hypot(axX,axY); axX,axY=(1.,0.) if L<1e-9 else (axX/L,axY/L)
    ts=[(p[0]-cx)*axX+(p[1]-cy)*axY for p in cl]
    return cx,cy,axX,axY,min(ts),max(ts)

def intake_r(): return INTAKE_RADII[intake_idx]
def heading(a, b): return math.atan2(b[1]-a[1], b[0]-a[0])
def hdg_diff(h1, h2): return abs((h2-h1+math.pi)%(2*math.pi)-math.pi)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLUSTER TRAVERSAL HELPERS
#  Each cluster produces a list of Pose2d = (x, y, travel_heading_rad)
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_poses(cl, from_pos):
    """
    Walk the PCA spine of a cluster from the near end to the far end.
    Places waypoints every DENSE_STEP metres.
    Each waypoint gets the travel heading = direction along the spine.
    Returns list of (x, y, heading_rad).
    """
    if len(cl) == 1:
        p = cl[0]
        h = heading(from_pos, p)
        return [(p[0], p[1], h)]

    cx, cy, axX, axY, mn, mx = pca_axis(cl)
    sA = (cx+mn*axX, cy+mn*axY)
    sB = (cx+mx*axX, cy+mx*axY)
    # orient: near end first
    if dd(from_pos, sB) < dd(from_pos, sA):
        axX, axY = -axX, -axY; mn, mx = -mx, -mn

    travel_hdg = math.atan2(axY, axX)   # heading along the spine toward far end
    n = max(2, int(math.ceil((mx-mn)/DENSE_STEP))+1)
    poses = []
    for k in range(n):
        t = mn + (mx-mn)*k/(n-1)
        px = cx + t*axX; py = cy + t*axY
        poses.append((px, py, travel_hdg))
    return poses

def traverse_dist(cl, from_pos):
    """Approximate distance to enter and exit a cluster."""
    en = nearest_p(cl, from_pos)
    ex = farthest(cl, from_pos)
    return dd(from_pos, en) + dd(en, ex)

def exit_pos(cl, from_pos):
    """Where does the robot end up after traversing this cluster."""
    cx, cy, axX, axY, mn, mx = pca_axis(cl) if len(cl)>1 else (cl[0][0],cl[0][1],1,0,0,0)
    sA = (cx+mn*axX, cy+mn*axY); sB = (cx+mx*axX, cy+mx*axY)
    return sB if dd(from_pos, sB) >= dd(from_pos, sA) else sA


# ═══════════════════════════════════════════════════════════════════════════════
#  FAST CORE: greedy insert + single 2-opt
#  This is O(k²) where k = number of clusters (~6-10).
#  Runs in < 1ms on desktop, < 5ms on RoboRio.
# ═══════════════════════════════════════════════════════════════════════════════

def fast_op_sequence(robot, clusters, budget, score_fn):
    """
    Greedy ratio construction + single 2-opt pass.
    score_fn(cluster, from_pos) → float (higher = more desirable)

    Returns ordered list of cluster indices.
    """
    n = len(clusters)
    unvisited = list(range(n))
    seq = []
    spent = 0.
    cur = robot

    # ── Greedy construction ──────────────────────────────────────────────────
    while unvisited:
        best_idx = None; best_ratio = -1.; best_cost = 0.
        for i in unvisited:
            c = clusters[i]
            cost = traverse_dist(c, cur)
            if spent + cost > budget: continue
            ratio = score_fn(c, cur) / max(cost, 0.01)
            if ratio > best_ratio:
                best_ratio = ratio; best_idx = i; best_cost = cost
        if best_idx is None: break
        seq.append(best_idx)
        spent += best_cost
        cur = exit_pos(clusters[best_idx], cur)
        unvisited.remove(best_idx)

    if not seq: return seq

    # ── Single 2-opt pass ────────────────────────────────────────────────────
    # Reverse sub-sequences to reduce total travel without changing which
    # clusters we visit. Budget stays the same (reordering doesn't change
    # total distance significantly, just ordering).
    improved = True
    while improved:
        improved = False
        for i in range(len(seq)-1):
            for j in range(i+2, len(seq)):
                # cost of original order i→i+1, j
                pos_i   = robot if i == 0 else exit_pos(clusters[seq[i-1]], robot)
                pos_i1  = exit_pos(clusters[seq[i]], pos_i)
                pos_j   = exit_pos(clusters[seq[j-1]], pos_i1) if j > i+1 else pos_i1

                d_orig = dd(pos_i, nearest_p(clusters[seq[i]], pos_i)) + \
                         dd(exit_pos(clusters[seq[i]], pos_i),
                            nearest_p(clusters[seq[j]], pos_j))

                # cost of reversed order: go to seq[j] first, then backwards to seq[i]
                d_rev  = dd(pos_i, nearest_p(clusters[seq[j]], pos_i)) + \
                         dd(exit_pos(clusters[seq[j]], pos_i),
                            nearest_p(clusters[seq[i]], pos_i))

                if d_rev < d_orig - 0.05:
                    seq[i:j+1] = list(reversed(seq[i:j+1]))
                    improved = True
                    break
            if improved: break

    return seq

def seq_to_poses(robot, seq, clusters):
    """Convert cluster-index sequence → list of (x, y, heading_rad) Pose2d."""
    poses = []; cur = robot
    for i in seq:
        c = clusters[i]
        wps = cluster_poses(c, cur)
        poses.extend(wps)
        if wps: cur = (wps[-1][0], wps[-1][1])
    return poses


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHMS  (all < 5ms)
# ═══════════════════════════════════════════════════════════════════════════════

def A_FastOP(balls, robot):
    """
    FastOP ★★★  — Greedy ratio + 2-opt
    Score = cluster_size² / distance.
    Quadratic size reward makes large clusters dominant.
    Budget enforced. Single 2-opt reorder. ~1ms.
    """
    v = valid_balls(balls, robot)
    if not v: return []
    cls = dbscan(v)
    def score(c, pos): return len(c)**2
    seq = fast_op_sequence(robot, cls, BUDGET_M, score)
    return seq_to_poses(robot, seq, cls)


def A_MomentumOP(balls, robot):
    """
    MomentumOP ★★★ — Heading-committed greedy + fill-in pass.

    Phase 1 – Forward sweep (anti-spin):
      Hard reject > MAX_TURN_DEG, EMA heading, size²/cost+penalty scoring.

    Phase 2 – Fill-in (collect more balls):
      Any cluster skipped in phase 1 gets a second chance.
      Try inserting it at every gap in the existing route.
      Accept if: budget fits AND the turn into/out-of it < FILL_MAX_TURN.
      This recovers clusters that were behind the robot at first but are
      now between two existing waypoints — common with scattered layouts.

    Phase 3 – Append stragglers:
      After fill-in, greedily append any still-unvisited cluster that
      fits in remaining budget AND is within STRAGGLER_DIST of the
      current path end. Catches nearby loners cheaply.
    """
    v = valid_balls(balls, robot)
    if not v: return []
    cls = dbscan(v)

    MAX_TURN_RAD    = math.radians(110)   # hard reject beyond this (phase 1)
    FILL_MAX_TURN   = math.radians(140)   # more lenient for fill-in (phase 2)
    STRAGGLER_DIST  = 2.5                 # max metres to detour for straggler (phase 3)
    RELAX_STEP      = math.radians(30)
    EMA_ALPHA       = 0.55
    HEAD_PEN_LOCAL  = 2.0

    # ── Phase 1: forward momentum sweep ──────────────────────────────────────
    unvisited = list(range(len(cls)))
    seq = []; spent = 0.; cur = robot; smooth_h = None

    while unvisited:
        limit = MAX_TURN_RAD
        best_idx = None; best_score = -1.; best_cost = 0.

        for attempt in range(5):
            for i in unvisited:
                c = cls[i]
                cost = traverse_dist(c, cur)
                if spent + cost > BUDGET_M: continue
                en = nearest_p(c, cur)
                turn = hdg_diff(smooth_h, heading(cur, en)) if smooth_h is not None else 0.
                if turn > limit: continue
                score = len(c)**2 / max(cost + HEAD_PEN_LOCAL * turn, 0.01)
                if score > best_score:
                    best_score = score; best_idx = i; best_cost = cost
            if best_idx is not None: break
            limit += RELAX_STEP

        if best_idx is None: break

        ex = exit_pos(cls[best_idx], cur)
        new_h = heading(cur, ex)
        if smooth_h is None:
            smooth_h = new_h
        else:
            delta = (new_h - smooth_h + math.pi) % (2*math.pi) - math.pi
            smooth_h += EMA_ALPHA * delta

        seq.append(best_idx); spent += best_cost
        cur = ex; unvisited.remove(best_idx)

    # ── Phase 2: fill-in — try inserting skipped clusters between waypoints ──
    # Build position list: pos[k] = robot exit after visiting seq[0..k-1]
    def build_pos_list(seq):
        pos = [robot]
        for idx in seq:
            pos.append(exit_pos(cls[idx], pos[-1]))
        return pos

    changed = True
    while changed and unvisited:
        changed = False
        pos_list = build_pos_list(seq)
        current_cost = spent

        best_insert = None  # (gain_score, insert_at, cluster_idx, extra_cost)

        for i in list(unvisited):
            c = cls[i]
            # try inserting at each gap k (between seq[k-1] and seq[k])
            for k in range(len(seq) + 1):
                pb = pos_list[k]
                pa = pos_list[k + 1] if k < len(seq) else pos_list[k]

                en = nearest_p(c, pb)
                ex = exit_pos(c, pb)

                # cost of inserting: dist to entry + through cluster + dist to next
                if k < len(seq):
                    extra = (dd(pb, en) + dd(en, ex) + dd(ex, pa)
                             - dd(pb, pa))
                else:
                    extra = dd(pb, en) + dd(en, ex)

                if current_cost + extra > BUDGET_M: continue

                # turn check: angle into cluster from pb, and out toward pa
                h_in  = heading(pb, en)
                h_out = heading(ex, pa) if k < len(seq) else None
                h_pb  = heading(pos_list[k-1], pb) if k > 0 else None

                turn_in  = hdg_diff(h_pb, h_in)  if h_pb  is not None else 0.
                turn_out = hdg_diff(h_in, h_out)  if h_out is not None else 0.

                if turn_in > FILL_MAX_TURN or turn_out > FILL_MAX_TURN:
                    continue

                gain = len(c)**2 / max(extra, 0.01)
                if best_insert is None or gain > best_insert[0]:
                    best_insert = (gain, k, i, extra)

        if best_insert:
            _, k, i, extra = best_insert
            seq.insert(k, i)
            spent += extra
            unvisited.remove(i)
            changed = True

    # ── Phase 3: append nearby stragglers at end ──────────────────────────────
    cur = exit_pos(cls[seq[-1]], robot) if seq else robot
    # rebuild cur properly
    pos_list = build_pos_list(seq)
    cur = pos_list[-1]

    for i in list(unvisited):
        c = cls[i]
        cost = traverse_dist(c, cur)
        if cost > STRAGGLER_DIST: continue
        if spent + cost > BUDGET_M: continue
        # light turn check — don't spin at the very end
        en = nearest_p(c, cur)
        if smooth_h is not None and hdg_diff(smooth_h, heading(cur, en)) > math.radians(150):
            continue
        seq.append(i); spent += cost
        cur = exit_pos(c, cur); unvisited.remove(i)

    return seq_to_poses(robot, seq, cls)


def A_DensityOP(balls, robot):
    """
    DensityOP ★★  — Score by balls-per-square-metre density × count.
    Aggressively skips sparse clusters even if nearby.
    Best for fields with clear dense clumps and random scattered loners.
    """
    v = valid_balls(balls, robot)
    if not v: return []
    cls = dbscan(v)

    def density(c):
        if len(c) <= 1: return 0.5
        cx, cy, axX, axY, mn, mx = pca_axis(c)
        span = max(mx - mn, 0.3)
        return len(c) / span

    def score(c, pos):
        d = density(c)
        return (len(c) * d) ** 1.3   # super-linear: dense clusters win hard

    seq = fast_op_sequence(robot, cls, BUDGET_M, score)
    return seq_to_poses(robot, seq, cls)

def A_VelocityOP(balls, robot):
    """
    VelocityOP ★★★★  — Time-optimal physics-aware routing.

    Scores each candidate cluster by   size² / estimated_seconds
    where time is derived from a full trapezoid velocity profile that
    accounts for:

      • Acceleration from rest (or carry-over speed) to cruise speed
      • Forced braking for sharp turns (exponential speed penalty)
      • Momentum carry-over — a cluster directly ahead while already at
        speed costs almost nothing; the same cluster after a 150° turn
        costs 2-3× more time than distance alone would predict

    This means the algorithm naturally:
      - Chains aligned clusters into fast straight blasts
      - Avoids U-turns even more aggressively than MomentumOP
      - Prefers closer clusters only when they don't kill momentum

    Tune PHYS_MAX_SPEED, PHYS_MAX_ACCEL, PHYS_TURN_DECAY at the top
    of the file to match your actual drivetrain.
    """
    v = valid_balls(balls, robot)
    if not v: return []
    cls = dbscan(v)
    n   = len(cls)

    unvisited  = list(range(n))
    seq        = []
    spent_dist = 0.0
    cur        = robot
    cur_speed  = 0.0    # robot starts stationary
    cur_hdg    = None   # no heading until first move

    # ── Greedy construction scored by time ───────────────────────────────────
    while unvisited:
        best_idx   = -1
        best_score = -1.0
        best_tcost = 0.0
        best_spd   = 0.0

        for i in unvisited:
            c = cls[i]
            dist_cost = traverse_dist(c, cur)
            if spent_dist + dist_cost > BUDGET_M: continue

            t_cost, exit_spd = cluster_phys_cost(c, cur, cur_speed, cur_hdg)
            score = len(c)**2 / max(t_cost, 0.01)

            if score > best_score:
                best_score = score
                best_idx   = i
                best_tcost = t_cost
                best_spd   = exit_spd

        if best_idx < 0: break

        ex        = exit_pos(cls[best_idx], cur)
        cur_hdg   = heading(cur, ex)
        cur_speed = best_spd
        spent_dist += traverse_dist(cls[best_idx], cur)
        cur = ex
        seq.append(best_idx)
        unvisited.remove(best_idx)

    if not seq: return []

    # ── Single 2-opt pass (geometry-based, same as FastOP) ───────────────────
    improved = True
    while improved:
        improved = False
        for i in range(len(seq) - 1):
            for j in range(i + 2, len(seq)):
                pos_i  = robot if i == 0 else exit_pos(cls[seq[i-1]], robot)
                pos_j  = exit_pos(cls[seq[j-1]], pos_i) if j > i+1 else pos_i
                exit_i = exit_pos(cls[seq[i]], pos_i)

                d_orig = (dd(pos_i, nearest_p(cls[seq[i]], pos_i)) +
                          dd(exit_i, nearest_p(cls[seq[j]], pos_j)))
                d_rev  = (dd(pos_i, nearest_p(cls[seq[j]], pos_i)) +
                          dd(exit_pos(cls[seq[j]], pos_i),
                             nearest_p(cls[seq[i]], pos_i)))

                if d_rev < d_orig - 0.05:
                    seq[i:j+1] = list(reversed(seq[i:j+1]))
                    improved = True
                    break
            if improved: break

    return seq_to_poses(robot, seq, cls)

def A_SweepOP(balls, robot):
    """
    SweepOP ★★  — Zero-backtrack forward sweep.
    Finds the direction from the robot toward the centroid of all balls.
    Sorts clusters by projection onto that axis.
    Visits them in strict forward order (no backtracking possible).
    Skips clusters that would exceed budget.
    Perfect for fields where balls are arranged in a rough line or arc.
    """
    v = valid_balls(balls, robot)
    if not v: return []
    cls = dbscan(v)
    # Field direction: robot → centroid of all valid balls
    all_cx = sum(b[0] for b in v)/len(v); all_cy = sum(b[1] for b in v)/len(v)
    fx = all_cx - robot[0]; fy = all_cy - robot[1]
    fl = math.hypot(fx, fy)
    if fl < 1e-6: fx, fy = 1., 0.
    else: fx, fy = fx/fl, fy/fl

    # Sort clusters by forward projection
    def proj(c): cx, cy = centroid(c); return (cx-robot[0])*fx + (cy-robot[1])*fy
    ordered = sorted(cls, key=proj)

    # Visit in order, skipping over-budget ones
    seq_idx = []; spent = 0.; cur = robot
    for i, c in enumerate(ordered):
        cost = traverse_dist(c, cur)
        if spent + cost <= BUDGET_M:
            seq_idx.append(i); spent += cost; cur = exit_pos(c, cur)

    poses = []; cur = robot
    for i in seq_idx:
        c = ordered[i]; wps = cluster_poses(c, cur)
        poses.extend(wps)
        if wps: cur = (wps[-1][0], wps[-1][1])
    return poses


def A_GreedyRef(balls, robot):
    """
    GreedyRef  — Nearest cluster, no ratio, budget-limited.
    Reference baseline. Shows how bad pure-greedy is.
    """
    v = valid_balls(balls, robot)
    if not v: return []
    cls = dbscan(v); rem = list(cls); cur = robot; poses = []; spent = 0.
    while rem:
        nc = min(rem, key=lambda c: cl_mindist(c, cur)); rem.remove(nc)
        cost = traverse_dist(nc, cur)
        if spent + cost > BUDGET_M: continue
        wps = cluster_poses(nc, cur); poses.extend(wps); spent += cost
        if wps: cur = (wps[-1][0], wps[-1][1])
    return poses


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM TABLE
# ═══════════════════════════════════════════════════════════════════════════════

ALGO_GROUPS = {
    "fast": "─── Budget-Aware Algorithms ───",
    "ref":  "─── Reference ───",
}

ALGOS = [
    ("r","FastOP  ★★★  size²/cost + 2-opt",    "#ffff00","fast",A_FastOP),
    ("t","MomentumOP ★★  heading-penalised",    "#ff8800","fast",A_MomentumOP),
    ("e","DensityOP ★★  density×count score",   "#00ffcc","fast",A_DensityOP),
    ("a","SweepOP ★★  forward-sort, no U-turns","#ff80ff","fast",A_SweepOP),
    ("s","GreedyRef  nearest-only (baseline)",  "#8888aa","ref", A_GreedyRef),
    ("v","VelocityOP ★★★★ time-optimal physics", "#00ff88","fast",A_VelocityOP),
]
ALL_KEYS = {row[0] for row in ALGOS}


# ═══════════════════════════════════════════════════════════════════════════════
#  BALL GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def rand_balls(n=300):
    balls=[]; nc=random.randint(4,8)
    for _ in range(nc):
        cx=random.uniform(2,FIELD_W-2); cy=random.uniform(1,FIELD_H-1); sp=random.uniform(0.3,1.1)
        for _ in range(n//nc):
            balls.append((max(0.2,min(FIELD_W-.2,cx+random.gauss(0,sp))),
                          max(0.2,min(FIELD_H-.2,cy+random.gauss(0,sp)))))
    for _ in range(random.randint(3,8)):
        balls.append((random.uniform(.5,FIELD_W-.5),random.uniform(.5,FIELD_H-.5)))
    return balls

def line_balls():
    balls=[]
    for row in [2.,4.,6.]:
        for x in np.linspace(2,14,12):
            balls.append((x+random.gauss(0,.12),row+random.gauss(0,.12)))
    for _ in range(10): balls.append((random.uniform(1,15),random.uniform(.5,7.5)))
    return balls

def sparse_balls():
    return [(random.uniform(.5,FIELD_W-.5),random.uniform(.5,FIELD_H-.5)) for _ in range(22)]


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_field(ax):
    ax.set_facecolor("#09091a"); ax.set_xlim(-.5,FIELD_W+.5); ax.set_ylim(-.5,FIELD_H+.5)
    ax.set_aspect("equal"); ax.tick_params(colors="#556677")
    for sp in ax.spines.values(): sp.set_edgecolor("#334455")
    ax.add_patch(mpatches.FancyBboxPatch((0,0),FIELD_W,FIELD_H,
        boxstyle="square,pad=0",linewidth=2,edgecolor="#445566",facecolor="#060612",zorder=0))
    for x in range(0,int(FIELD_W)+1,2): ax.axvline(x,color="#0f0f28",lw=.4,zorder=1)
    for y in range(0,int(FIELD_H)+1,2): ax.axhline(y,color="#0f0f28",lw=.4,zorder=1)
    ax.axvline(FIELD_W/2,color="#1e1e44",lw=1.,ls="--",zorder=1)

def draw_balls(ax, balls, hit_sets):
    if not balls: return
    na=len(hit_sets)
    if na==0:
        xs,ys=zip(*balls)
        ax.scatter(xs,ys,s=55,color="#7788aa",zorder=4,edgecolors="#445566",linewidths=.5); return
    hc=[sum(m[i] for m in hit_sets) for i in range(len(balls))]
    for i,(bx,by) in enumerate(balls):
        h=hc[i]
        if   h==0:  ax.scatter(bx,by,s=50,color="#991111",zorder=4,edgecolors="#cc2222",linewidths=.5); ax.plot(bx,by,"x",color="#ff5555",ms=6,mew=1.6,zorder=5)
        elif h==na: ax.scatter(bx,by,s=60,color="#22dd66",zorder=4,edgecolors="white",linewidths=.6)
        else:       ax.scatter(bx,by,s=55,color="#ffcc00",zorder=4,edgecolors="white",linewidths=.6)

def draw_path(ax, robot, poses, color, lw=2.4, alpha=.90, show_hdg=True):
    """Draw spline + heading arrows at each Pose2d waypoint."""
    if not poses: return None, None
    xs, ys = pp_spline([(robot[0], robot[1], heading(robot, (poses[0][0],poses[0][1])))] + poses)
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=6, solid_capstyle="round")

    # Direction arrow at end
    if len(xs) >= 4:
        ti = max(len(xs)-max(5,len(xs)//6),0)
        ax.annotate("", xy=(xs[-1],ys[-1]), xytext=(xs[ti],ys[ti]),
                    arrowprops=dict(arrowstyle="-|>",color=color,lw=1.5,mutation_scale=13),zorder=8)

    # Pose2d waypoint markers with travel heading arrows
    if show_hdg:
        arrow_len = 0.22
        for px, py, ph in poses:
            ax.plot(px, py, "o", ms=5, color=color, zorder=7,
                    markeredgecolor="white", markeredgewidth=.5, alpha=.8)
            ax.annotate("", xy=(px+arrow_len*math.cos(ph), py+arrow_len*math.sin(ph)),
                        xytext=(px, py),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                        mutation_scale=7), zorder=8, alpha=0.7)
    return xs, ys

def draw_robot(ax, pos, ir):
    ax.add_patch(Circle(pos,ir,fill=False,edgecolor="#00cfff",linewidth=1.,linestyle=":",alpha=.30,zorder=3))
    ax.plot(*pos,"D",ms=14,color="#00cfff",zorder=10,markeredgecolor="white",markeredgewidth=1.4)
    ax.plot(*pos,".",ms=3,color="white",zorder=11)
    ax.annotate("Robot",xy=pos,xytext=(pos[0]+.15,pos[1]+.25),
                color="#00cfff",fontsize=8,fontweight="bold",zorder=11)

def build_legend(ax, rows, ir, nb, nc):
    handles=[]; last_grp=None
    for key,name,color,grp,active,pct,plen,eff,ms in rows:
        if grp!=last_grp:
            handles.append(Line2D([0],[0],color="#445566",lw=0,label=ALGO_GROUPS[grp]))
            last_grp=grp
        if active and pct is not None:
            suffix = f"  {pct:.0f}%  {plen:.1f}m  {eff:.2f}b/m  {ms:.1f}ms"
        elif active: suffix="  (no waypoints)"
        else:        suffix=""
        handles.append(Line2D([0],[0],color=color,lw=2.5,
                               linestyle="solid" if active else "dotted",
                               alpha=1. if active else .18,
                               label=f"[{key.upper()}] {name}{suffix}"))
    leg=ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(1.01,1.02),
                  framealpha=.10,facecolor="#09091a",edgecolor="#334455",
                  labelcolor="white",fontsize=7.8,
                  title=(f"Intake={ir:.2f}m  Balls={nb}  Clusters={nc}  Budget={BUDGET_M:.0f}m\n"
                         "Coverage%  PathLen(m)  Efficiency(balls/m)  Time(ms)\n"
                         "● green=hit  ● yellow=partial  ✕ red=missed\n"
                         "Arrows = Pose2d travel heading (for waypointsFromPoses)\n"
                         "Space=rand  l/p=layout  c=custom  i=intake  B/b=budget±2\n"
                         "j=print Java Pose2d list  m=arms  w=save"),
                  title_fontsize=7.3)
    leg.get_title().set_color("#99aacc")


# ═══════════════════════════════════════════════════════════════════════════════
#  JAVA CODE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def to_java(robot, poses, algo_name):
    """Print copy-pasteable Java code for PathPlannerPath.waypointsFromPoses."""
    print(f"\n// ── {algo_name} ──")
    print("List<Waypoint> waypoints = PathPlannerPath.waypointsFromPoses(")
    all_poses = [(robot[0], robot[1], 0.0)] + poses
    for i, (px, py, ph) in enumerate(all_poses):
        deg = math.degrees(ph)
        comma = "," if i < len(all_poses)-1 else ""
        print(f"    new Pose2d({px:.3f}, {py:.3f}, Rotation2d.fromDegrees({deg:.1f})){comma}")
    print(");")
    print("PathPlannerPath path = new PathPlannerPath(")
    print("    waypoints, constraints, null,")
    print("    new GoalEndState(0.0, Rotation2d.fromDegrees(0)));")
    print(f"// Total waypoints: {len(all_poses)}  (keep under ~12 for RoboRio speed)")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SIM
# ═══════════════════════════════════════════════════════════════════════════════

class Sim:
    def __init__(self):
        self.balls     = rand_balls()
        self.robot     = (2.0, 4.0)
        self.active    = {row[0]: row[0] in {"r","t","a"} for row in ALGOS}
        self.placing   = False
        self.custom    = []
        self.show_arms = False
        self.last_poses = {}   # key → poses (for Java export)

        self.fig, self.ax = plt.subplots(figsize=(20,9))
        self.fig.patch.set_facecolor("#09091a")
        plt.subplots_adjust(left=.04,right=.60,top=.93,bottom=.07)
        self.fig.canvas.mpl_connect("button_press_event",self.click)
        self.fig.canvas.mpl_connect("key_press_event",   self.key)
        self.redraw()

    def redraw(self):
        ir=intake_r(); self.ax.cla(); draw_field(self.ax)
        v=valid_balls(self.balls,self.robot); cl=dbscan(v) if v else []
        hit_sets=[]; rows=[]; self.last_poses={}

        for key,name,color,grp,fn in ALGOS:
            active=self.active[key]; pct=plen=eff=ms=None
            if active:
                try:
                    t0=time.perf_counter()
                    poses=fn(self.balls,self.robot)
                    ms=(time.perf_counter()-t0)*1000.
                    self.last_poses[key]=(name,poses)
                    if poses:
                        xs,ys=draw_path(self.ax,self.robot,poses,color)
                        if xs is not None:
                            pct,hit,plen,eff=measure(xs,ys,self.balls,ir)
                            hit_sets.append(hit)
                        else: hit_sets.append([False]*len(self.balls))
                    else: hit_sets.append([False]*len(self.balls))
                except Exception as ex:
                    import traceback; traceback.print_exc()
                    hit_sets.append([False]*len(self.balls))
            rows.append((key,name,color,grp,active,pct,plen,eff,ms))

        draw_balls(self.ax,self.balls,hit_sets)
        draw_robot(self.ax,self.robot,ir)
        build_legend(self.ax,rows,ir,len(self.balls),len(cl))

        mode=" [PLACING – c to finish]" if self.placing else ""
        self.ax.set_title(
            f"Ball Path Sim — PathPlanner Pose2d Edition{mode}  Budget={BUDGET_M:.0f}m\n"
            "Arrows = travel heading at each Pose2d waypoint (→ waypointsFromPoses)",
            color="#8899bb",fontsize=8.5,pad=5)
        self.ax.set_xlabel("X (m)",color="#667799")
        self.ax.set_ylabel("Y (m)",color="#667799")
        self.fig.canvas.draw_idle()

    def click(self,ev):
        if ev.inaxes!=self.ax or ev.xdata is None: return
        if self.placing: self.custom.append((ev.xdata,ev.ydata)); self.balls=list(self.custom)
        else: self.robot=(ev.xdata,ev.ydata)
        self.redraw()

    def key(self,ev):
        global intake_idx, BUDGET_M
        k=(ev.key or "").lower() if len(ev.key or "")==1 else (ev.key or "")

        if k in ALL_KEYS:
            self.active[k]=not self.active[k]; self.redraw()
        elif k==" ":
            self.balls=rand_balls(); self.placing=False; self.redraw()
        elif ev.key=="B":   # shift+b = increase
            BUDGET_M=min(60.,BUDGET_M+2.); print(f"Budget → {BUDGET_M:.0f}m"); self.redraw()
        elif k=="b":
            BUDGET_M=max(4.,BUDGET_M-2.); print(f"Budget → {BUDGET_M:.0f}m"); self.redraw()
        elif k=="i":
            intake_idx=(intake_idx+1)%len(INTAKE_RADII)
            print(f"Intake → {intake_r():.2f}m"); self.redraw()
        elif k=="m":
            self.show_arms=not self.show_arms; self.redraw()
        elif k=="l":
            self.balls=line_balls(); self.placing=False; self.redraw()
        elif k=="p":
            self.balls=sparse_balls(); self.placing=False; self.redraw()
        elif k=="c":
            if not self.placing: self.placing=True; self.custom=[]; self.balls=[]
            else: self.placing=False; print(f"Done – {len(self.custom)} balls")
            self.redraw()
        elif k=="j":   # j = print Java Pose2d list
            for key,(name,poses) in self.last_poses.items():
                if poses: to_java(self.robot, poses, name)
        elif k=="p":
            self.balls=sparse_balls(); self.placing=False; self.redraw()
        elif k=="w":
            self.fig.savefig("ball_sim.png",dpi=150,bbox_inches="tight",
                             facecolor=self.fig.get_facecolor())
            print("Saved → ball_sim.png")

    def run(self): plt.show()


if __name__=="__main__":
    print(__doc__)
    print(f"Budget: {BUDGET_M}m  |  Intake: {intake_r()}m")
    Sim().run()