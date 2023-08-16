from typing import Tuple
import pandas as pd
from video_grabber.logging import Logger

MILLI = 1000


class JumpCounter:
    __logger = Logger(__qualname__)

    def __init__(
        self,
        man_height_m=2,
        earth_gravity=9.81,
        acceleration_error_ratio=0.7,
        interpolation_span_p=15,
        interpolation_span_v=60,
        interpolation_span_a=60,
        max_milliseconds_between_jumps=800,
        min_n_frames=4,
        min_jump_ratio_to_body=0.001,
    ):
        self.man_height_m = man_height_m
        self.earth_gravity = earth_gravity
        self.acceleration_error = acceleration_error_ratio * self.earth_gravity
        self.interpolation_span_p = interpolation_span_p
        self.interpolation_span_v = interpolation_span_v
        self.interpolation_span_a = interpolation_span_a
        self.max_milliseconds_between_jumps = max_milliseconds_between_jumps
        self.min_n_frames = min_n_frames
        self.min_jump_ratio_to_body = min_jump_ratio_to_body

        self.init_status()

    def init_status(self):
        self._timestamps = []
        self._boxes = []
        self._all_timestamps = []
        self._all_boxes = []
        self._count = 0
        self._last_jump_timestamp = None
        self.df_check = None

    def _is_height_change(self, new_box):
        # return self._boxes and new_box[3] != self._boxes[-1][3]
        return self._boxes and abs(new_box[3] / self._boxes[-1][3] - 1) > 0.1

    def _check_for_jump(self, df_box: pd.DataFrame = None):
        self.df_box = self.df if df_box is None else df_box
        # self.__logger.debug(f"{df= }")
        m_to_p_ratio = self.man_height_m / self.df_box.box.head(1).item()[3]
        # self.df_check.index = pd.to_datetime(self.df_check.index, unit="ms")
        self.df_box["y"] = self.df_box.box.apply(lambda r: -r[1] * m_to_p_ratio)
        # self.__logger.debug(f"{df= }")
        self.__logger.debug(f"{self.df_box= }")

        self.interpolated = self.df_box.y.resample("L").ffill(limit=1).interpolate()
        self.smoothed = self.interpolated.ewm(span=self.interpolation_span_p).mean()
        self.velocity = (self.smoothed.diff() * MILLI).ewm(span=self.interpolation_span_v).mean()
        self.acceleration = (self.velocity.diff() * MILLI).ewm(span=self.interpolation_span_a).mean()

        person_height = m_to_p_ratio * self.df_box.box[-1][-1]

        self.df_check = pd.DataFrame(
            {
                "y": self.smoothed,
                "v": self.velocity,
                "a": self.acceleration.shift(-int(self.interpolation_span_a / 2)),
            }
        )
        # self.df_check = pd.DataFrame({"y": smoothed, "v": velocity, "a": acceleration})
        self.df_check["freefall"] = (self.df_check.a + self.earth_gravity).abs() < self.acceleration_error
        self.df_check["local_maximum"] = (self.df_check.y.shift(1) < self.df_check.y) & (
            self.df_check.y.shift(-1) <= self.df_check.y
        )
        self.df_check["high_enough"] = (
            self.df_check.y - self.df_check.y.min()
        ) > person_height * self.min_jump_ratio_to_body

        self.__logger.debug(f"{self.df_check=}")
        return any(self.df_check.freefall & self.df_check.local_maximum & self.df_check.high_enough)
        #     self._boxes = self._boxes[-self.min_n_frames :]
        #     self._timestamps = self._timestamps[-self.min_n_frames :]
        #     return True

        # return False

    def count_jumps(self, box: Tuple[int], timestamp: float) -> int:
        """_summary_

        Args:
            box (Tuple[int]): (center_x,center_y,w,h)
            timestamp (float): unit in second

        Returns:
            int: number of total jumps
        """
        if box is None:
            return self._count

        if self._is_height_change(box):
            self.__logger.debug("!!!_is_height_change")
            # self.init_status()

        self._boxes.append(box)
        self._timestamps.append(timestamp)
        self._all_boxes.append(box)
        self._all_timestamps.append(timestamp)

        if len(self._boxes) < self.min_n_frames:
            return self._count

        if self._check_for_jump():
            # if self._last_jump_timestamp and timestamp - self._last_jump_timestamp > self.max_milliseconds_between_jumps:
            #     self.__logger.debug("!!!_check_for_jump")
            #     self.__logger.debug(
            #         f"{timestamp= }, {self._last_jump_timestamp= }, {timestamp - self._last_jump_timestamp= }"
            #     )
            #     self._count = 0

            self._count += 1
            self._last_jump_timestamp = timestamp
            self._boxes = self._boxes[-self.min_n_frames :]
            self._timestamps = self._timestamps[-self.min_n_frames :]

        return self._count

    @property
    def df(self):
        return pd.DataFrame({"box": self._boxes}, index=pd.to_datetime(self._timestamps, unit="s"))

    @property
    def all_df(self):
        return pd.DataFrame({"box": self._all_boxes}, index=pd.to_datetime(self._all_timestamps, unit="s"))

    def dump(self):
        self.__logger.debug(f"{self.all_df= }")
        self.all_df.to_pickle("boxes_2.df")

        from pathlib import Path
        import pickle

        data_to_save = {
            "timestamp": self._all_timestamps,
            "box": self._all_boxes,
        }
        with Path("boxes_2.pk").open(mode="wb") as fp:
            pickle.dump(data_to_save, fp)

    def __del__(self):
        self.dump()
        pass
