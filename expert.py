"""
Scripted expert for the PandaPickAndPlace-v3 environment.

The expert runs a finite-state machine with six phases:

  1. PRE_GRASP   — move end-effector to directly above the cube
  2. DESCEND     — lower the end-effector to the grasp height
  3. CLOSE       — close the gripper (N steps)
  4. LIFT        — raise the cube to a safe carry height
  5. TRANSPORT   — move horizontally to above the goal
  6. PLACE       — lower the cube to the goal height
  7. OPEN        — open the gripper (N steps)
  8. DONE        — no-op

Key design choices
------------------
* The EEF in panda-gym cannot descend below z≈0.04 (physical robot limit).
  The cube sits on the table with its centre at z≈0.02.  The gripper fingers
  extend ~0.025 m below the EEF centre, so the correct grasp height is
  EEF z ≈ obj_z + GRASP_HEIGHT_OFFSET.

* When entering DESCEND the object's x-y position is *frozen* so that the
  EEF tracks a fixed point instead of chasing a sliding cube.

* Phase-completion for DESCEND checks only the x-y plane distance so that
  the physical z-floor of the robot is not falsely counted as failure.

Observation indices (flat = obs + desired_goal):
  ee_pos  : [0:3]
  obj_pos : [7:10]
  goal    : [19:22]
"""

from __future__ import annotations

import numpy as np
from enum import IntEnum

from envs.pick_place_env import PickPlaceEnv


class Phase(IntEnum):
    PRE_GRASP = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    TRANSPORT = 4
    PLACE = 5
    OPEN = 6
    DONE = 7



class ScriptedExpert:
    """
    Phase-based scripted expert for pick-and-place.

    Parameters
    ----------
    pos_gain       : proportional gain for position controller
    reach_thresh_xy: x-y plane distance to consider a phase complete (m)
    reach_thresh_3d: 3-D distance threshold for non-descend phases (m)
    close_steps    : number of steps to hold gripper-close action
    open_steps     : number of steps to hold gripper-open action
    grasp_z_offset : metres above the object centre to hover before descent
    carry_height   : absolute z-height used during transport (m)
    place_z_offset : metres above the goal surface to finish placing (m)
    """

    def __init__(
        self,
        pos_gain: float = 10.0,
        reach_thresh_xy: float = 0.025,
        reach_thresh_3d: float = 0.025,
        close_steps: int = 20,
        open_steps: int = 15,
        grasp_z_offset: float = 0.12,
        carry_height: float = 0.25,
        place_z_offset: float = 0.02,
    ):
        self.gain = pos_gain
        self.thresh_xy = reach_thresh_xy
        self.thresh_3d = reach_thresh_3d
        self.close_steps = close_steps
        self.open_steps = open_steps
        self.grasp_z_offset = grasp_z_offset
        self.carry_height = carry_height
        self.place_z_offset = place_z_offset

        self._phase = Phase.PRE_GRASP
        self._phase_counter = 0
        # Frozen x-y object position, set when entering DESCEND
        self._grasp_xy: np.ndarray | None = None

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._phase = Phase.PRE_GRASP
        self._phase_counter = 0
        self._grasp_xy = None

    # ------------------------------------------------------------------

    def get_action(self, flat_obs: np.ndarray) -> np.ndarray:
        """
        Compute the next 4-dim action for the current observation.

        Returns np.ndarray of shape (4,) with values in [-1, 1].
        """
        ee_pos = flat_obs[PickPlaceEnv.IDX_EE_POS].copy()
        obj_pos = flat_obs[PickPlaceEnv.IDX_OBJ_POS].copy()
        goal_pos = flat_obs[PickPlaceEnv.IDX_GOAL].copy()

        # Grasp height: EEF centre at the cube centre height gives 3.1cm of
        # finger-side contact with the cube (fingers close in Y, tips at z=cube-0.011)
        # The robot can reach this height when the gripper is OPEN.
        grasp_z = obj_pos[2]  # cube centre z ≈ 0.02 m

        action = np.zeros(4, dtype=np.float32)

        if self._phase == Phase.PRE_GRASP:
            target = np.array([obj_pos[0], obj_pos[1],
                                obj_pos[2] + self.grasp_z_offset])
            action[:3] = self._pos_ctrl(ee_pos, target)
            action[3] = 1.0  # OPEN gripper (prepare for grasp)
            if self._dist3d(ee_pos, target) < self.thresh_3d:
                # Freeze xy for the descent so the cube is not chased
                self._grasp_xy = obj_pos[:2].copy()
                self._phase = Phase.DESCEND

        elif self._phase == Phase.DESCEND:
            # Use frozen xy so we descend vertically above the cube.
            # Keep the gripper OPEN: with closed fingers the robot cannot
            # descend as low (finger collision with table), but with open
            # fingers (spreading in Y) it can reach obj_pos[2] = 0.02 m.
            target = np.array([self._grasp_xy[0], self._grasp_xy[1], grasp_z])
            action[:3] = self._pos_ctrl(ee_pos, target)
            action[3] = 1.0  # keep OPEN
            # Use 3D distance so we don't close until EEF is truly at grasp z
            if self._dist3d(ee_pos, target) < self.thresh_3d:
                self._phase = Phase.CLOSE
                self._phase_counter = 0

        elif self._phase == Phase.CLOSE:
            # Hold at grasp position and close gripper
            target = np.array([self._grasp_xy[0], self._grasp_xy[1], grasp_z])
            action[:3] = self._pos_ctrl(ee_pos, target)
            action[3] = -1.0  # CLOSE gripper to grasp
            self._phase_counter += 1
            if self._phase_counter >= self.close_steps:
                self._phase = Phase.LIFT

        elif self._phase == Phase.LIFT:
            target = np.array([obj_pos[0], obj_pos[1], self.carry_height])
            action[:3] = self._pos_ctrl(ee_pos, target)
            action[3] = -1.0  # keep CLOSED (hold object)
            if self._dist3d(ee_pos, target) < self.thresh_3d:
                self._phase = Phase.TRANSPORT

        elif self._phase == Phase.TRANSPORT:
            target = np.array([goal_pos[0], goal_pos[1], self.carry_height])
            action[:3] = self._pos_ctrl(ee_pos, target)
            action[3] = -1.0  # keep CLOSED
            if self._dist3d(ee_pos, target) < self.thresh_3d:
                self._phase = Phase.PLACE

        elif self._phase == Phase.PLACE:
            target = np.array([goal_pos[0], goal_pos[1],
                                goal_pos[2] + self.place_z_offset])
            action[:3] = self._pos_ctrl(ee_pos, target)
            action[3] = -1.0  # keep CLOSED
            if self._dist3d(ee_pos, target) < self.thresh_3d:
                self._phase = Phase.OPEN
                self._phase_counter = 0

        elif self._phase == Phase.OPEN:
            target = np.array([goal_pos[0], goal_pos[1],
                                goal_pos[2] + self.place_z_offset])
            action[:3] = self._pos_ctrl(ee_pos, target)
            action[3] = 1.0  # OPEN gripper to release
            self._phase_counter += 1
            if self._phase_counter >= self.open_steps:
                self._phase = Phase.DONE

        # Phase.DONE → zero action (stay still)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        return self._phase == Phase.DONE

    # ------------------------------------------------------------------

    def _pos_ctrl(
        self, current: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Simple proportional controller; output clipped to [-1, 1]."""
        return np.clip((target - current) * self.gain, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _dist3d(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _dist_xy(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

