Getting started
===============

Requirements
------------

* Python >= 3.10
* ``numpy``, ``scipy``
* ``pin`` (pinocchio >= 3) for FK, Jacobians and numerical IK
* the EAIK fork `kthfan/EAIK <https://github.com/kthfan/EAIK>`_
  (``eaik==1.2.2+robust.1``) for analytical IK, plus ``urchin``. The fork
  adds ``tip_link``, idealization of near-parallel / near-intersecting axes,
  and ``pos_error`` / ``rot_error`` on the original model; see its
  ``ROBUST_CHANGES.md``. The stock ``eaik`` package on PyPI does not have
  these APIs.

Install
-------

``requirements.txt`` builds the EAIK fork from git. This needs a C++ compiler
and takes 1-2 minutes:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .              # or: pip install -e ".[test,docs]"

To install the fork on its own, or from a local checkout (the submodules hold
Eigen and ik-geo):

.. code-block:: bash

   pip install "eaik @ git+https://github.com/kthfan/EAIK.git@v1.2.2+robust.1"
   git clone --recursive https://github.com/kthfan/EAIK && pip install ./EAIK

Run the tests and build these docs:

.. code-block:: bash

   pytest
   sphinx-build -b html docs docs/_build/html

First motion
------------

.. code-block:: python

   import numpy as np
   from motion_planner import default_planner, matrix_to_pose, pose_to_matrix

   planner = default_planner("pinocchio")       # TM5S + bundled calibration
   q0 = np.radians([0, 17, 69, 11, 86, 6])      # current joints [rad]

   x, y, z, rx, ry, rz = matrix_to_pose(planner.fk(q0))
   goal = pose_to_matrix([x - 0.10, y, z, rx, ry, rz])   # 10 cm along -x

   waypoints = planner.plan_line_t(q0, goal, velocity=50, acc_time=0.2,
                                   on_discontinuity="hold")
   for wp in waypoints:
       print(np.degrees(wp.positions), wp.velocity, wp.acc_time, wp.blend_percentage)

``examples/quickstart.py`` is the same example as a script.

Command line
------------

``scripts/plan_motion.py`` runs FK, IK, PTP_T or LINE_T from the shell. With
``--deg`` / ``--mm`` it uses teach-pendant units:

.. code-block:: bash

   python scripts/plan_motion.py fk   --q0 0 17 69 11 86 6 --deg --mm
   python scripts/plan_motion.py ik   --goal 400 -100 300 180 0 90 --deg --mm --q0 0 17 69 11 86 6
   python scripts/plan_motion.py line --q0 0 17 69 11 86 6 --goal 300 -100 300 180 0 90 \
       --deg --mm --speed 50 --mode hold
   python scripts/plan_motion.py line ... --json      # waypoints + diagnostics as JSON

The exit code is 1 when the planner raises :class:`~motion_planner.solver.IKError` or
:class:`~motion_planner.planner.PathDiscontinuityError`.
