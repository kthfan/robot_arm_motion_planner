"""Local motion planning: turn Cartesian goals into joint-space waypoints."""
from __future__ import annotations

import math
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

from motion_planner.line_search import (  # noqa: F401 - PathDiscontinuityError re-exported
    JUMP,
    LINE,
    Event,
    LineSearch,
    PathDiscontinuityError,
)
from motion_planner.path import CartesianPath
from motion_planner.self_motion import SelfMotion
from motion_planner.solver import (
    IKError,
    IKSelector,
    KinematicsSolver,
    closest_to_reference,
)


@dataclass
class JointWaypoint:
    """One joint-space (PTP_J style) waypoint of a planned motion.

    ``positions`` are the 6 joint angles [rad]; ``velocity`` is the speed % of
    the joint speed limits (0, 100]; ``acc_time`` the ramp time [s].
    """

    positions: list[float]
    velocity: float
    acc_time: float
    blend_percentage: int
    fine_goal: bool
    blocked: bool

# What LINE_T does where the path cannot be followed continuously.
DISCONTINUITY_MODES = ("raise", "hold", "jump")


def _search_summary(search: LineSearch) -> dict:
    """Counters of a LINE_T search for diagnostics (also after a failure; never raises)."""
    try:
        goals = getattr(search, "_goals", None)
        return {
            "effort": int(getattr(search, "_effort", 0)),
            "ik_queries": int(getattr(search, "_ik_queries", 0)),
            "exhausted": sorted(getattr(search, "_exhausted", ())),
            "n_goal_configs": None if goals is None else len(goals),
            "stats": dict(getattr(search, "stats", {})),
            "windows_tried": [
                {"s": float(w.s), "sigma": float(w.sigma), "end": bool(w.end),
                 "families_reached": len(reached), "families_attached": len(attached)}
                for w, reached, attached in getattr(search, "_tried", [])[:30]
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


class MotionPlanner(ABC):
    """Computes TM-style Cartesian motions locally and returns joint waypoints.

    ``q0`` is the joint configuration at the start of the motion (radians).
    ``goal`` is the 4x4 TCP pose in the robot base frame.
    """

    @abstractmethod
    def fk(self, q: Sequence[float]) -> np.ndarray:
        """Return the 4x4 TCP pose for joint vector ``q`` (the model the planner uses)."""

    @abstractmethod
    def plan_ptp_t(
        self,
        q0: Sequence[float],
        goal: np.ndarray,
        velocity: float,
        acc_time: float,
        blend_percentage: int = 0,
        fine_goal: bool = False,
        blocked: bool = True,
        goal_joints: Optional[Sequence[float]] = None,
        diagnostics: Optional[dict] = None,
    ) -> list[JointWaypoint]:
        """Point-to-point to a Cartesian goal; TCP path unconstrained.

        ``velocity`` is the PTP speed percentage (0-100]. ``goal_joints``: see
        :meth:`plan_line_t`.
        """

    @abstractmethod
    def plan_line_t(
        self,
        q0: Sequence[float],
        goal: np.ndarray,
        velocity: float,
        acc_time: float,
        blend_percentage: int = 0,
        fine_goal: bool = False,
        blocked: bool = True,
        on_discontinuity: str = "raise",
        goal_joints: Optional[Sequence[float]] = None,
        diagnostics: Optional[dict] = None,
    ) -> list[JointWaypoint]:
        """Cartesian path from FK(q0) to ``goal``.

        ``velocity`` is the speed % (0, 100], like TM LINE_T; ``acc_time`` the
        ramp time [s].
        ``on_discontinuity`` handles sections that cannot be followed
        continuously (singularity, joint limit):

        * ``"raise"``: :class:`PathDiscontinuityError` (or :class:`IKError`).
        * ``"hold"``: the joints are reconfigured to another IK branch with the
          TCP held near the line (see ``max_flip_deviation``), where the arm is
          closest to a singularity; fewest reconfigurations first, then little
          joint travel and a large joint-limit margin. Raises
          :class:`PathDiscontinuityError` with a diagnosis if impossible.
        * ``"jump"``: like ``"hold"``, but where no held reconfiguration
          leads to the goal the joints jump (the TCP leaves the line; fewest
          jumps first, then fewest held reconfigurations, then small jumps
          and little joint travel), and points without IK are skipped.
          Where two jump targets (or goal configurations) cost nearly the
          same, a small change of the start or goal can switch between them
          (a different goal configuration and jump): pass ``goal_joints`` to
          fix the goal configuration.

        ``goal_joints``: with ``goal_selection="closest"`` (default; "raise" /
        "hold") the goal configuration is the IK solution of the goal pose
        closest to ``goal_joints`` (joint-space distance, joint limits
        respected); if the line cannot reach it, the planner falls back to any
        configuration on the IK branch of ``goal_joints``, with a warning.
        Without ``goal_joints``, in "hold" mode, the plan to the configuration
        closest to ``q0`` and the plan to any configuration (fewest
        reconfigurations first) are compared and the one with less joint
        travel is used. With ``goal_selection="branch"``, and
        always in "jump" mode: same IK branch (shoulder / elbow / wrist) and
        within 180 deg on joints 1, 4 and 6. :class:`IKError` if the
        goal pose has no such solution. The goal itself must always be
        reachable.

        ``diagnostics``: if given, filled with what the planner found (path,
        reconfiguration events, search counters, warnings), also when it raises.
        """


GOAL_SELECTIONS = ("closest", "branch")


class KinematicMotionPlanner(MotionPlanner):
    """Motion planner built on any :class:`KinematicsSolver`.

    Args:
        solver: FK/IK backend.
        selector: Chooses among IK solutions for PTP_T (default: closest to q0).
        linear_tolerance: Max TCP deviation from the LINE_T path [m].
        angular_tolerance: Max TCP orientation deviation from the path [rad].
        max_joint_step: Max joint change between consecutive LINE_T waypoints [rad].
        angular_velocity: TCP rotation speed at 100% LINE_T speed [rad/s]; the
            motion takes as long as the slower of translation and rotation.
        max_linear_velocity: TCP speed at 100% LINE_T speed [m/s].
        max_depth: Finest LINE_T path step is ``2**-max_depth`` of the line.
        max_joint_acceleration: Joint acceleration limits [rad/s^2] (scalar or
            per joint) between LINE_T waypoints. Defaults to reaching the joint
            velocity limit in 0.2 s. If the requested speed would exceed the
            velocity or acceleration limits (too fast, or near a singularity),
            the whole line is uniformly slowed down and a warning is issued.
            With ``on_discontinuity`` "hold" / "jump" it slows down locally
            instead: segments too fast for the joints run at 100% PTP_J, and
            joint reconfigurations run as PTP_J at the requested speed %.
        min_velocity: Lowest PTP_J speed percentage sent to the TM robot,
            in (0, 100]. Slower LINE_T segments are raised to it, so they
            finish sooner than planned.
        max_flip_deviation: Max TCP deviation [m] from the crossing point
            while LINE_T reconfigures the joints (``on_discontinuity`` "hold"
            / "jump"). 0 disables held reconfigurations.
        max_flip_angle: Max TCP orientation deviation [rad] during such a
            reconfiguration. Next to (not on) a singularity the self-motion
            holds the pose only approximately and mostly tilts the TCP.
        max_held_length: Max length [m] of the line along which a held
            reconfiguration keeps the TCP within ``max_flip_deviation`` /
            ``max_flip_angle`` instead of exactly on the line.
        goal_selection: How LINE_T ``goal_joints`` picks the goal
            configuration: ``"closest"`` (the goal IK solution closest to
            ``goal_joints``; near the wrist singularity the IK branch label
            does not tell the two wrist postures apart) or ``"branch"`` (any
            solution on the IK branch of ``goal_joints``, the former behaviour).
        smooth_timing: With ``on_discontinuity`` "hold" / "jump", time the
            LINE_T waypoints with one speed profile along the line that keeps
            every joint within its velocity and acceleration limits (slow
            sections are entered and left gradually). False: the former local
            slowdown, where only the segments too fast for the joints are capped
            at 100% and the speed % can step from one waypoint to the next.
        simplify_line: Drop LINE_T waypoints on the line that are not needed to
            keep the joint interpolation within ``linear_tolerance`` /
            ``angular_tolerance`` and ``max_joint_step`` (fewer, longer PTP_J
            segments for the robot). Held reconfigurations and jumps are kept.
        held_smoothing: Sweeps of the smoothing of held reconfigurations
            (0: off). Spreads their joint motion evenly over the waypoints within
            ``max_flip_deviation`` / ``max_flip_angle``, so the joint speeds do
            not alternate from one PTP_J segment to the next (jitter).
        held_smoothing_tolerance: Share of ``max_flip_deviation`` /
            ``max_flip_angle`` the smoothing may use (a held route that already
            deviates more keeps its own deviation as the bound).
        held_deviation_weight: Pull of the smoothing towards the line (0: none).
            Brings the TCP of held reconfigurations as close to the line as the
            joint path allows (the search itself accepts anything within half
            of ``max_flip_deviation`` / ``max_flip_angle``).
        timing_on_dense: Time the line on all waypoints before
            ``simplify_line`` merges them (a merged segment gets the time of the
            waypoints it replaces). Without it a long merged segment next to
            short ones distorts the speed profile (a slow segment at the joint
            between a held route and the rest of the line).
        simplify_held: Also merge waypoints of held reconfigurations whose
            neighbours can be joined directly, within the deviation the held
            route already has (at least ``linear_tolerance`` /
            ``angular_tolerance``). Fewer, longer PTP_J segments: the robot
            needs a minimum time per segment, which caps the speed of routes
            made of many short segments.
        held_smoothing_batch: Smooth every other waypoint at once (much faster;
            False: one waypoint after the other, the earlier behaviour).
        max_plan_time: Planning time budget of one LINE_T motion [s] (None: no
            budget). Past it the search stops and the motion fails instead of
            searching for minutes.
    """

    def __init__(
        self,
        solver: KinematicsSolver,
        selector: Optional[IKSelector] = None,
        linear_tolerance: float = 1e-3,
        angular_tolerance: float = math.radians(0.5),
        max_joint_step: float = 0.2,
        angular_velocity: float = math.pi / 2,
        max_depth: int = 16,
        max_joint_acceleration: Optional[Union[float, Sequence[float]]] = None,
        min_velocity: float = 1.0,
        max_linear_velocity: float = 1.0,
        max_flip_deviation: float = 3e-3,
        max_flip_angle: float = math.radians(3.0),
        max_held_length: float = 0.15,
        goal_selection: str = "closest",
        smooth_timing: bool = True,
        simplify_line: bool = True,
        held_smoothing: int = 10,
        held_smoothing_tolerance: float = 0.5,
        held_deviation_weight: float = 0.1,
        held_smoothing_batch: bool = True,
        max_plan_time: Optional[float] = 30.0,
        timing_on_dense: bool = True,
        simplify_held: bool = False,
    ) -> None:
        if goal_selection not in GOAL_SELECTIONS:
            raise ValueError(f"goal_selection must be one of {GOAL_SELECTIONS}, got {goal_selection!r}")
        if not 0.0 < min_velocity <= 100.0:
            raise ValueError(f"min_velocity must be in (0, 100] %, got {min_velocity}")
        if not max_linear_velocity > 0.0:
            raise ValueError(f"max_linear_velocity must be > 0 m/s, got {max_linear_velocity}")
        if not max_flip_deviation >= 0.0:
            raise ValueError(f"max_flip_deviation must be >= 0 m, got {max_flip_deviation}")
        if not max_flip_angle >= 0.0:
            raise ValueError(f"max_flip_angle must be >= 0 rad, got {max_flip_angle}")
        if not max_held_length >= 0.0:
            raise ValueError(f"max_held_length must be >= 0 m, got {max_held_length}")
        self.solver = solver
        self.selector = selector or closest_to_reference
        self.linear_tolerance = linear_tolerance
        self.angular_tolerance = angular_tolerance
        self.max_joint_step = max_joint_step
        self.angular_velocity = angular_velocity
        self.max_linear_velocity = max_linear_velocity
        self.max_depth = max_depth
        self.min_velocity = min_velocity
        self.max_flip_deviation = max_flip_deviation
        self.max_flip_angle = max_flip_angle
        self.max_held_length = max_held_length
        self.goal_selection = goal_selection
        self.smooth_timing = smooth_timing
        self.simplify_line = simplify_line
        self.held_smoothing = int(held_smoothing)
        self.held_smoothing_tolerance = float(held_smoothing_tolerance)
        self.held_deviation_weight = float(held_deviation_weight)
        self.held_smoothing_batch = bool(held_smoothing_batch)
        self.max_plan_time = max_plan_time
        self.timing_on_dense = bool(timing_on_dense)
        self.simplify_held = bool(simplify_held)
        if max_joint_acceleration is None:
            self.max_joint_acceleration = np.asarray(solver.velocity_limits) / 0.2
        else:
            self.max_joint_acceleration = np.broadcast_to(
                np.asarray(max_joint_acceleration, dtype=float), (solver.nq,)
            ).copy()

    def fk(self, q: Sequence[float]) -> np.ndarray:
        return self.solver.fk(q)

    # ------------------------------------------------------------ PTP_T

    def plan_ptp_t(
        self,
        q0: Sequence[float],
        goal: np.ndarray,
        velocity: float,
        acc_time: float,
        blend_percentage: int = 0,
        fine_goal: bool = False,
        blocked: bool = True,
        goal_joints: Optional[Sequence[float]] = None,
        diagnostics: Optional[dict] = None,
    ) -> list[JointWaypoint]:
        q0 = np.asarray(q0, dtype=float)
        solutions = self.solver.ik_all(goal, seeds=[q0])
        if not solutions:
            raise IKError("no IK solution for the PTP_T goal pose")
        if goal_joints is None:
            q = self.selector(solutions, q0)
        else:
            same, _ = self._goal_filter(goal, q0, goal_joints, "PTP_T")
            ref = np.asarray(goal_joints, dtype=float)
            q = min(
                [q for q in self.solver.ik_analytical(goal) + solutions if same(q)],
                key=lambda q: float(np.linalg.norm(q - ref)),
            )
        if diagnostics is not None:
            diagnostics["n_ik_solutions"] = len(solutions)
        return [
            JointWaypoint([float(x) for x in q], velocity, acc_time,
                          blend_percentage, fine_goal, blocked)
        ]

    def _self_motion(self) -> SelfMotion:
        return SelfMotion(
            self.solver,
            self.linear_tolerance,
            self.angular_tolerance,
            self.max_joint_step,
            self.max_flip_deviation,
            self.max_flip_angle,
        )

    def _goal_filter(
        self,
        goal: np.ndarray,
        q0: np.ndarray,
        goal_joints: Optional[Sequence[float]],
        motion: str,
        select: str = "branch",
        diag: Optional[dict] = None,
    ) -> tuple[Callable[[np.ndarray], bool], list[np.ndarray]]:
        """Accepts the goal configurations allowed by ``goal_joints``, and the IK
        solutions of the goal; raises :class:`IKError` if the goal pose has none.

        ``select="closest"``: only the configuration family (IK branch and
        2 pi turns) of the goal solution closest to ``goal_joints``."""
        solutions = self.solver.ik_analytical(goal) or self.solver.ik_all(goal, seeds=[q0])
        if not solutions:
            raise IKError(f"no IK solution for the {motion} goal pose")
        if goal_joints is None:
            return (lambda q: True), solutions
        ref = np.asarray(goal_joints, dtype=float)
        if ref.shape != (self.solver.nq,):
            raise ValueError(f"goal_joints must have {self.solver.nq} values, got {ref.shape}")
        sm = self._self_motion()
        family = sm.family(ref, ref)
        if select == "closest":
            lo, hi = self.solver.lower_limits, self.solver.upper_limits
            inside = [q for q in solutions if np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)]
            if inside:
                target = min(inside, key=lambda q: float(np.linalg.norm(q - ref)))
                family = sm.family(target, ref)
                if diag is not None:
                    diag["goal_target"] = [float(x) for x in target]

        def same(q: np.ndarray) -> bool:
            return sm.family(q, ref) == family

        if not any(same(q) for q in solutions):
            closest = min(solutions, key=lambda q: float(np.max(np.abs(q - ref))))
            raise IKError(
                f"no IK solution of the {motion} goal pose in the configuration of goal_joints "
                f"(branch {self.solver.branch(ref)}, joints 1/4/6 within 180 deg); the closest "
                f"solution differs by up to {math.degrees(np.max(np.abs(closest - ref))):.1f} deg"
            )
        return same, solutions

    # ------------------------------------------------------------ LINE_T

    def plan_line_t(
        self,
        q0: Sequence[float],
        goal: np.ndarray,
        velocity: float,
        acc_time: float,
        blend_percentage: int = 0,
        fine_goal: bool = False,
        blocked: bool = True,
        on_discontinuity: str = "raise",
        goal_joints: Optional[Sequence[float]] = None,
        diagnostics: Optional[dict] = None,
    ) -> list[JointWaypoint]:
        if on_discontinuity not in DISCONTINUITY_MODES:
            raise ValueError(
                f"on_discontinuity must be one of {DISCONTINUITY_MODES}, got {on_discontinuity!r}"
            )
        if not 0.0 < velocity <= 100.0:
            raise ValueError(f"LINE_T velocity must be a speed % in (0, 100], got {velocity}")
        if acc_time < 0:
            raise ValueError(f"acc_time must be >= 0, got {acc_time}")

        t_start = time.perf_counter()
        deadline = None if self.max_plan_time is None else t_start + float(self.max_plan_time)
        q0 = np.asarray(q0, dtype=float)
        goal = np.asarray(goal, dtype=float)
        path = CartesianPath.from_poses(self.solver.fk(q0), goal)
        if path.translation_length < 1e-9 and path.rotation_angle < 1e-9:
            return []
        diag = diagnostics if diagnostics is not None else {}
        diag["path"] = {
            "translation_m": float(path.translation_length),
            "rotation_rad": float(path.rotation_angle),
        }
        messages = diag.setdefault("warnings", [])

        def warn(msg: str) -> None:
            warnings.warn(msg)
            messages.append(msg)

        # "closest": the goal configuration nearest to goal_joints. Without them
        # (hold mode) the plan to the goal configuration nearest to the start
        # competes with the plan to any configuration: less joint travel wins.
        # Jump mode keeps the IK-branch rule, where the fewest jumps decide.
        select = self.goal_selection if on_discontinuity != "jump" else "branch"
        if goal_joints is None and not (select == "closest" and on_discontinuity == "hold"):
            select = "any"
        reference = goal_joints if goal_joints is not None else q0
        goal_ok, goal_solutions = self._goal_filter(
            goal, q0, None if select == "any" else reference, "LINE_T", select, diag)
        diag["n_goal_solutions"] = len(goal_solutions)
        diag["goal_selection"] = select
        timing = diag["timing_s"] = {"goal_filter": time.perf_counter() - t_start}

        def make_search(accept):
            return LineSearch(
                self.solver,
                self._self_motion(),
                path,
                q0,
                on_discontinuity,
                accept,
                self.linear_tolerance,
                self.angular_tolerance,
                self.max_joint_step,
                self.max_depth,
                self.max_held_length,
                goal_solutions,
                deadline,
            )

        search = make_search(goal_ok)
        try:
            try:
                if select == "closest" and goal_joints is None:
                    plan, search = self._least_travel(make_search, goal_ok, q0, diag)
                else:
                    plan = search.run()
            except PathDiscontinuityError as exc:
                if select != "closest" or goal_joints is None:
                    raise
                # The closest goal configuration is not reachable along the line:
                # accept any configuration on the IK branch of goal_joints.
                diag["search_closest"] = _search_summary(search)
                diag["closest_error"] = str(exc)
                goal_ok, _ = self._goal_filter(goal, q0, goal_joints, "LINE_T", "branch")
                search = make_search(goal_ok)
                plan = search.run()
                diag["goal_selection"] = "closest-fallback"
                warn(
                    "LINE_T: the goal configuration closest to "
                    + ("goal_joints" if goal_joints is not None else "the start")
                    + " cannot be reached along the line; "
                    + ("another configuration on the IK branch of goal_joints"
                       if goal_joints is not None else "another goal configuration")
                    + " is used"
                )
        finally:
            diag["search"] = _search_summary(search)
            timing["search"] = time.perf_counter() - t_start - timing["goal_filter"]
        samples = plan.samples
        diag["n_samples_raw"] = len(samples)
        if self.held_smoothing > 0 and on_discontinuity != "raise":
            t0 = time.perf_counter()
            samples, diag["held_smoothing"] = search.smooth_held(
                q0, samples, self.held_smoothing, self.held_smoothing_tolerance,
                self.held_deviation_weight, self.held_smoothing_batch)
            timing["smoothing"] = time.perf_counter() - t0
        dense = samples
        if self.simplify_line:
            samples = search.simplify(q0, samples, held=self.simplify_held)
        diag["events"] = [asdict(e) for e in plan.events]
        diag["samples"] = [[float(s), kind] for s, _, kind in samples]
        self._warn_events(plan.events, warn)
        reconfigure = on_discontinuity != "raise"
        frac = velocity / 100.0
        nominal = max(
            path.translation_length / (frac * self.max_linear_velocity),
            path.rotation_angle / (frac * self.angular_velocity),
        )
        times = [self._time_at(s, nominal, acc_time) for s in [0.0] + [s for s, _, _ in samples]]
        qs = np.array([q0] + [q for _, q, _ in samples])
        dqs = np.diff(qs, axis=0)
        dts = np.diff(times)
        # Reconfigurations at a point of the line (and jumps) are timed as plain
        # PTP_J moves; held reconfigurations along the line are timed as line.
        ss = [0.0] + [s for s, _, _ in samples]
        jumps = np.array([kind == JUMP or (kind != LINE and s <= s_prev)
                          for (s, _, kind), s_prev in zip(samples, ss)], dtype=bool)
        if reconfigure and self.smooth_timing:
            # One speed profile along the line within the joint velocity and
            # acceleration limits: slow sections are entered and left gradually.
            if self.timing_on_dense and len(dense) != len(samples):
                # time the waypoints before merging; a merged segment takes the
                # time of the waypoints it replaces
                qd = np.array([q0] + [q for _, q, _ in dense])
                sd = [0.0] + [s for s, _, _ in dense]
                jd = np.array([kind == JUMP or (kind != LINE and s <= s_prev)
                               for (s, _, kind), s_prev in zip(dense, sd)], dtype=bool)
                dts_dense = self._smooth_durations(np.diff(qd, axis=0), sd, jd, nominal, acc_time)
                where = {id(x): i for i, x in enumerate(dense)}
                cum = np.concatenate([[0.0], np.cumsum(dts_dense)])
                dts = np.diff(cum[[0] + [where[id(x)] + 1 for x in samples]])
            else:
                dts = self._smooth_durations(dqs, ss, jumps, nominal, acc_time)
            scale = 1.0
        elif reconfigure:
            # Local slowdown: segments too fast for the joints are capped at
            # 100% below, so only they run slower than the requested speed.
            scale = 1.0
        else:
            scale = self._time_scale(dqs, dts)
            if scale > 1.0 + 1e-9:
                warn(
                    f"LINE_T slowed down x{scale:.2f} to respect joint velocity/acceleration "
                    "limits (requested speed too high or path near a singularity)"
                )
        dts = dts * scale
        diag["time_scale"] = float(scale)

        vmax = self.solver.velocity_limits
        actions = []
        capped = 0
        last = len(samples) - 1
        for i, q in enumerate(qs[1:]):
            dq = np.abs(dqs[i])
            # TM PTP: segment time = max_j |dq_j| / (pct/100 * vmax_j)
            pct = velocity if jumps[i] else 100.0 * float(np.max(dq / vmax)) / dts[i]
            capped += pct > 100.0 + 1e-9
            pct = min(max(pct, self.min_velocity), 100.0)
            duration = float(np.max(dq / (pct / 100.0 * vmax)))
            ramp = acc_time if jumps[i] else acc_time * scale
            is_last = i == last
            actions.append(
                JointWaypoint(
                    [float(x) for x in q],
                    pct,
                    min(ramp, duration),
                    blend_percentage if is_last else 100,
                    fine_goal if is_last else False,
                    blocked if is_last else False,
                )
            )
        diag["capped_segments"] = int(capped)
        timing["total"] = time.perf_counter() - t_start
        if capped:
            warn(
                f"LINE_T: {capped} segment(s) capped at 100% joint speed, slower "
                "than the requested TCP speed there (near a singularity)"
            )
        return actions

    @staticmethod
    def _least_travel(make_search, closest_ok, q0: np.ndarray, diag: dict):
        """Hold mode without goal_joints: the plan to any goal configuration
        (fewest reconfigurations first) against the plan to the configuration
        closest to the start; the one with less joint travel."""

        def travel(plan) -> float:
            qs = np.array([q0] + [q for _, q, _ in plan.samples])
            return float(np.sum(np.linalg.norm(np.diff(qs, axis=0), axis=1)))

        any_search = make_search(lambda q: True)
        plan_any, error = None, None
        try:
            plan_any = any_search.run()
        except PathDiscontinuityError as exc:
            error = exc
        finally:
            diag["search_any"] = _search_summary(any_search)
        if plan_any is not None and (not plan_any.samples or closest_ok(plan_any.samples[-1][1])):
            diag["goal_selection"] = "closest"
            return plan_any, any_search
        closest_search = make_search(closest_ok)
        try:
            plan_closest = closest_search.run()
        except PathDiscontinuityError:
            diag["search_closest"] = _search_summary(closest_search)
            if plan_any is None:
                raise error
            diag["goal_selection"] = "least-travel:any"
            return plan_any, any_search
        if plan_any is None:
            diag["goal_selection"] = "closest"
            return plan_closest, closest_search
        t_any, t_closest = travel(plan_any), travel(plan_closest)
        diag["travel_rad"] = {"any": t_any, "closest": t_closest}
        if t_closest <= t_any:
            diag["goal_selection"] = "least-travel:closest"
            return plan_closest, closest_search
        diag["goal_selection"] = "least-travel:any"
        return plan_any, any_search

    def _smooth_durations(
        self,
        dqs: np.ndarray,
        ss: Sequence[float],
        jumps: np.ndarray,
        nominal: float,
        acc_time: float,
    ) -> np.ndarray:
        """Segment durations of a path-speed profile (discrete time-optimal
        parameterization along the waypoints, at rest at both ends).

        The path parameter runs over the waypoint index (one unit per segment),
        so held reconfigurations and jumps, which do not advance ``s``, are
        part of the same profile. Per segment the speed is limited by the
        requested TCP speed (the trapezoid of ``nominal`` / ``acc_time``), the
        joint velocity limits, and the joint acceleration limits, including the
        change of joint direction at each waypoint.
        """
        n = len(dqs)
        vmax = np.asarray(self.solver.velocity_limits, dtype=float)
        amax = np.asarray(self.max_joint_acceleration, dtype=float)
        ds_line = np.maximum(np.diff(np.asarray(ss, dtype=float)), 0.0)
        tiny = 1e-12
        # u = segment index units per second. TCP limit: a line segment covers
        # ds_line[i] of the line, which takes ds_line[i] * nominal at cruise.
        u_max = np.full(n, np.inf)
        a_tcp = np.full(n, np.inf)
        for i in range(n):
            if ds_line[i] > tiny and not jumps[i]:
                u_max[i] = 1.0 / (ds_line[i] * nominal)
                if acc_time > 0:
                    a_tcp[i] = u_max[i] / acc_time
        g = np.abs(dqs)  # joint change per index unit
        with np.errstate(divide="ignore"):
            u_max = np.minimum(u_max, np.min(np.where(g > tiny, vmax / g, np.inf), axis=1))
            a_joint = np.min(np.where(g > tiny, amax / g, np.inf), axis=1)
        a_max = np.minimum(a_joint, a_tcp)
        u_max = np.where(np.isfinite(u_max), u_max, 1.0 / max(nominal, 1e-3))
        a_max = np.where(np.isfinite(a_max), a_max, u_max / 0.05)
        # waypoint speed caps: segment caps on both sides, and the joint
        # velocity change through the waypoint |dq_i+1 - dq_i| u^2 <= amax
        w = np.zeros(n + 1)
        w[1:-1] = np.minimum(u_max[:-1], u_max[1:])
        turn = np.abs(dqs[1:] - dqs[:-1])
        with np.errstate(divide="ignore"):
            w_turn = np.sqrt(np.min(np.where(turn > tiny, amax / turn, np.inf), axis=1))
        w[1:-1] = np.minimum(w[1:-1], w_turn)
        # forward / backward passes, constant acceleration within a segment
        for k in range(1, n + 1):
            w[k] = min(w[k], math.sqrt(w[k - 1] ** 2 + 2.0 * a_max[k - 1]))
        for k in range(n - 1, -1, -1):
            w[k] = min(w[k], math.sqrt(w[k + 1] ** 2 + 2.0 * a_max[k]))
        # each segment: from w[k] up to at most its cap, then down to w[k + 1]
        dts = np.empty(n)
        for i in range(n):
            a, w0, w1 = a_max[i], w[i], w[i + 1]
            peak = min(u_max[i], math.sqrt(max(a + 0.5 * (w0 * w0 + w1 * w1), 0.0)))
            peak = max(peak, w0, w1, tiny)
            ramps = (2.0 * peak * peak - w0 * w0 - w1 * w1) / (2.0 * a)
            dts[i] = (peak - w0) / a + (peak - w1) / a + max(1.0 - ramps, 0.0) / peak
        return dts

    def _time_scale(self, dqs: np.ndarray, dts: np.ndarray) -> float:
        """Smallest uniform time stretch keeping joint velocity/acceleration in limits.

        Joint motion is modelled as constant velocity per segment, starting and
        ending at rest; acceleration is the velocity change at each waypoint.
        """
        vel = dqs / dts[:, None]
        scale = max(1.0, float(np.max(np.abs(vel) / self.solver.velocity_limits)))
        padded = np.vstack([np.zeros(vel.shape[1]), vel, np.zeros(vel.shape[1])])
        spans = np.concatenate([dts[:1], 0.5 * (dts[:-1] + dts[1:]), dts[-1:]])
        acc = np.abs(np.diff(padded, axis=0)) / spans[:, None]
        return max(scale, math.sqrt(float(np.max(acc / self.max_joint_acceleration))))

    @staticmethod
    def _warn_events(events: list[Event],
                     warn: Callable[[str], None] = warnings.warn) -> None:
        for ev in events:
            change = math.degrees(ev.joint_change)
            if ev.kind != JUMP:
                warn(
                    f"LINE_T reconfigured the joints at s in [{ev.s_from:.4f}, {ev.s_to:.4f}] "
                    f"(max joint change {change:.2f} deg) with the TCP held "
                    f"within {ev.deviation * 1000:.2f} mm of the line"
                )
                continue
            msg = (
                f"LINE_T left the straight line for s in [{ev.s_from:.4f}, {ev.s_to:.4f}]: "
                f"max joint jump {change:.2f} deg"
            )
            if ev.no_ik:
                msg += ", points without IK skipped"
            warn(msg)

    @staticmethod
    def _time_at(s: float, nominal: float, acc_time: float) -> float:
        """Time at path parameter ``s`` for a trapezoidal speed profile.

        Cruise speed covers the path in ``nominal`` seconds; ramps last
        ``acc_time`` (triangular profile if the path is too short).
        """
        if acc_time <= 0:
            return s * nominal
        v = 1.0 / nominal
        a = v / acc_time
        s_a = min(acc_time / (2.0 * nominal), 0.5)
        t_a = math.sqrt(2.0 * s_a / a)
        total = 2.0 * t_a + (1.0 - 2.0 * s_a) / v
        if s <= s_a:
            return math.sqrt(2.0 * s / a)
        if s >= 1.0 - s_a:
            return total - math.sqrt(2.0 * max(1.0 - s, 0.0) / a)
        return t_a + (s - s_a) / v
