Experiments and field logs
==========================

``scripts/line_motion_experiments/`` is a regression and diagnosis harness for
LINE_T. It plans seeded cases (random lines, wrist / elbow / shoulder
singularities, joint limits, large rotations, long and tiny lines, IK holes,
``goal_joints``, and reported user cases). Each plan is checked with an
independent checker that samples every PTP_J segment through FK.

.. code-block:: bash

   python scripts/line_motion_experiments/cases.py 0.5            # results/cases.json
   python scripts/line_motion_experiments/run.py base --workers 8  # results/base.jsonl
   python scripts/line_motion_experiments/report.py base

Change the planner, run again under a new tag, and compare with
``report.py base new``. ``--perturb K`` adds perturbed copies of each case to
test stability (similar inputs should give similar plans). ``oracle.py`` checks
whether a failed case is feasible at all with held reconfigurations.

Field logs written by the robot application can be summarised and exported as
cases with ``motion_log_tool.py``, then planned again with the current code.
See ``scripts/line_motion_experiments/README.md`` for all options.
