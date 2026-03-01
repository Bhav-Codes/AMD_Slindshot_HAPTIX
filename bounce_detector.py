"""
Automated bounce detection for table tennis tracking.

Uses Vertical Velocity Inversion on the ball's pixel Y-coordinate (cy)
to detect the exact frame when the ball impacts the table surface.

Hardware grid: 3 columns × 4 rows (12 vibrating motors).
Table dims:   1.525 m (width) × 2.74 m (length).
"""

from enum import Enum
from collections import deque

# ── Table dimensions (meters) ──────────────────────────────────
TABLE_WIDTH  = 1.525
TABLE_LENGTH = 2.74


class Phase(Enum):
    RISING  = "RISING"   # cy decreasing  (ball going UP in frame)
    FALLING = "FALLING"  # cy increasing  (ball going DOWN in frame)
    IMPACT  = "IMPACT"   # single-frame bounce event


class BounceDetector:
    """
    State-machine bounce detector driven by pixel y-coordinate.

    In a side-angle camera the ball's cy *increases* as it falls towards
    the table and *decreases* as it rises.  A bounce is the instant cy
    hits a local maximum (lowest point in frame) and reverses.

    Parameters
    ----------
    buffer_size : int
        Number of recent cy values kept for median smoothing (odd recommended).
    cooldown_frames : int
        Minimum gap between two accepted bounces.
    """

    def __init__(self, buffer_size: int = 3, cooldown_frames: int = 5):
        self.buffer_size = buffer_size
        self.cooldown_frames = cooldown_frames

        self._buf: deque[float] = deque(maxlen=buffer_size)
        self._prev_smooth: float | None = None
        self._phase: Phase = Phase.FALLING
        self._frames_since_bounce: int = cooldown_frames 
        self._seen_increase: bool = False

        # ── Peak tracking ──
        self._peak_cy: float = 0.0
        self._peak_pos: tuple[int, int] | None = None
        self._peak_frame: int = 0
        self.peak_position: tuple[int, int] | None = None
        self.peak_frame_id: int = 0  # the frame the ball actually hit

    @property
    def phase(self) -> Phase:
        """Current state of the detector."""
        return self._phase

    def update(self, cy: float, cx: int | None = None, frame_id: int = 0) -> bool:
        """
        Near-zero-lag detection. Triggers as soon as raw cy drops
        below the peak by just 1 pixel.
        """
        self._buf.append(cy)
        self._frames_since_bounce += 1

        # Use median only for phase status, not for the trigger itself
        smooth = self._smoothed_cy() if len(self._buf) >= self.buffer_size else cy
        
        if self._prev_smooth is not None:
             delta = smooth - self._prev_smooth
             if delta > 0: self._seen_increase = True
             
             # Update status phase
             if self._phase == Phase.FALLING and delta < 0:
                 # We don't trigger IMPACT here yet or we might trigger late
                 pass 
             if self._phase == Phase.RISING and delta > 0:
                 self._phase = Phase.FALLING
                 self._peak_cy = 0.0
                 self._peak_pos = None

        bounce = False
        
        # Immediate Trigger Logic
        if self._phase == Phase.FALLING and self._seen_increase:
            # Update peak if we are getting lower in frame
            if cy >= self._peak_cy:
                self._peak_cy = cy
                self._peak_frame = frame_id
                if cx is not None:
                    self._peak_pos = (cx, int(cy))
            
            # TRIGGER: 1px drop from peak = reversal confirmed
            REVERSAL_THRESHOLD = 1
            if cy < self._peak_cy - REVERSAL_THRESHOLD:
                if self._frames_since_bounce >= self.cooldown_frames:
                    self._phase = Phase.IMPACT
                    self.peak_position = self._peak_pos
                    self.peak_frame_id = self._peak_frame
                    bounce = True
                    self._frames_since_bounce = 0
                else:
                    self._phase = Phase.RISING

        if self._phase == Phase.IMPACT:
            self._phase = Phase.RISING
            self._peak_cy = 0.0
            self._peak_pos = None

        self._prev_smooth = smooth
        return bounce

    def reset(self):
        """Clear all internal state (e.g. between rallies)."""
        self._buf.clear()
        self._prev_smooth = None
        self._phase = Phase.FALLING
        self._frames_since_bounce = self.cooldown_frames
        self._seen_increase = False
        self._peak_cy = 0.0
        self._peak_pos = None
        self._peak_frame = 0
        self.peak_position = None
        self.peak_frame_id = 0

    # ── internals ──────────────────────────────────────────────
    def _smoothed_cy(self) -> float:
        """Median of the buffer — robust to single-frame noise."""
        vals = sorted(self._buf)
        mid = len(vals) // 2
        return float(vals[mid])


# ── Stateless convenience wrapper ──────────────────────────────
def detect_bounce(current_cy: float, history: list[float],
                  cooldown: int = 15) -> bool:
    """
    Pure-function bounce detection (no persistent state).

    Appends *current_cy* to *history* (which is mutated in place as a
    rolling window), then checks for a local-maximum reversal.

    Parameters
    ----------
    current_cy : float
        The ball's pixel y-coordinate in the current frame.
    history : list[float]
        Mutable list used as a rolling buffer.  Caller keeps this alive
        across calls.  Must also contain a ``'_cooldown'`` key if passed
        as a dict — but for simplicity this version just uses the list.
    cooldown : int
        Minimum frames between accepted bounces.

    Returns
    -------
    bool
        True at the moment of impact.
    """
    BUFFER = 5
    history.append(current_cy)
    if len(history) > BUFFER + 1:
        history.pop(0)

    if len(history) < BUFFER + 1:
        return False

    # median-smooth the last BUFFER values (excluding the previous one)
    window = sorted(history[-BUFFER:])
    smooth_now = window[len(window) // 2]

    prev_window = sorted(history[-(BUFFER + 1):-1])
    smooth_prev = prev_window[len(prev_window) // 2]

    # local max → bounce
    return smooth_now < smooth_prev  # cy decreased after increasing


# ── Grid mapping ───────────────────────────────────────────────
def map_to_grid(table_x: float, table_y: float,
                grid_cols: int = 3, grid_rows: int = 4,
                table_w: float = TABLE_WIDTH,
                table_l: float = TABLE_LENGTH) -> tuple[int, int]:
    """
    Convert real-world table coordinates to a motor grid index.

    Parameters
    ----------
    table_x, table_y : float
        Position on the table surface in metres (from homography).
    grid_cols, grid_rows : int
        Motor grid dimensions.  Default 3 × 4 = 12 motors.
    table_w, table_l : float
        Physical table dimensions in metres.

    Returns
    -------
    (col, row) : tuple[int, int]
        Zero-indexed grid cell.  col ∈ [0, grid_cols-1],
        row ∈ [0, grid_rows-1].
    """
    # clamp to table bounds
    x = max(0.0, min(table_x, table_w))
    y = max(0.0, min(table_y, table_l))

    col = int(x / table_w * grid_cols)
    row = int(y / table_l * grid_rows)

    # edge case: exactly on the far boundary
    col = min(col, grid_cols - 1)
    row = min(row, grid_rows - 1)

    return col, row
