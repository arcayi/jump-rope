from pathlib import Path
from typing import Tuple
import cv2
from matplotlib.backend_bases import FigureCanvasBase
import numpy as np
import pandas as pd
from video_grabber.logging import Logger
import matplotlib.pyplot as plt


MILLI = 1000


class JumpCounter:
    __logger = Logger(__qualname__)

    def __init__(
        self,
        man_height_m=1.8,
        earth_gravity=9.81,
        acceleration_error_ratio=0.6,
        interpolation_span_p=80,
        interpolation_span_v=20,
        interpolation_span_a=20,
        min_seconds_between_jumps=0.15,
        max_seconds_between_jumps=1,
        min_n_frames=10,
        min_jump_ratio_to_body=0.005,
        local_maximum_shift=100,
        local_min_span=300,
    ):
        self.man_height_m = man_height_m
        self.earth_gravity = earth_gravity
        self.acceleration_error = acceleration_error_ratio * self.earth_gravity
        self.interpolation_span_p = interpolation_span_p
        self.interpolation_span_v = interpolation_span_v
        self.interpolation_span_a = interpolation_span_a
        self.min_seconds_between_jumps = min_seconds_between_jumps
        self.max_seconds_between_jumps = max_seconds_between_jumps
        self.min_n_frames = min_n_frames
        self.min_jump_ratio_to_body = min_jump_ratio_to_body
        self.local_maximum_shift = local_maximum_shift
        self.local_min_span = local_min_span

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
        # self.__logger.debug(f"{self.df_box= }")

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
        self.df_check["freefall"] = self.df_check.a + self.earth_gravity < self.acceleration_error
        self.df_check["local_maximum"] = (self.df_check.y.shift(self.local_maximum_shift) < self.df_check.y) & (
            self.df_check.y.shift(-self.local_maximum_shift) <= self.df_check.y
        )
        self.df_check["high_enough"] = (
            self.df_check.y - self.df_check.y.rolling(self.local_min_span).min()
        ) > person_height * self.min_jump_ratio_to_body

        # self.__logger.debug(f"{self.df_check=}")
        return any(self.df_check.freefall & self.df_check.local_maximum & self.df_check.high_enough)
        #     self._boxes = self._boxes[-self.min_n_frames :]
        #     self._timestamps = self._timestamps[-self.min_n_frames :]
        #     return True

        # return False

    def __call__(self, box: Tuple[int], timestamp: float) -> int:
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
            # if self._last_jump_timestamp and timestamp - self._last_jump_timestamp > self.max_seconds_between_jumps:
            #     self.__logger.debug("!!!_check_for_jump")
            #     self.__logger.debug(
            #         f"{timestamp= }, {self._last_jump_timestamp= }, {timestamp - self._last_jump_timestamp= }"
            #     )
            #     self._count = 0
            if (
                self._last_jump_timestamp is None
                or timestamp - self._last_jump_timestamp >= self.min_seconds_between_jumps
            ):
                self._count += 1
                self._last_jump_timestamp = timestamp
                self._boxes = self._boxes[-self.min_n_frames :]
                self._timestamps = self._timestamps[-self.min_n_frames :]
            else:
                self.__logger.debug(f"{self._last_jump_timestamp= } {timestamp - self._last_jump_timestamp= }")

        return self._count

    # def visualize(self, df_all: pd.DataFrame = None, lenth: int = None):
    #     pass

    def draw_frame(self, frame, box_color=(0, 255, 0), counter_color=(36, 255, 12),):
        box = self._boxes[-1]
        jumps = self._count
        if box is not None:
            (x, y, w, h) = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            cv2.putText(frame, f"{jumps}", (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, counter_color, 6)
        # cv2.imshow("Video", frame)
        return frame

    def draw_history(self, df_all: pd.DataFrame = None, lenth: int = None):
        # df_all = self.all_df if df_all is None else df_all
        df_all = self.all_df if df_all is None else df_all
        df_box = df_all
        # self.__logger.info(f"{df_box= }")
        # print(f"{df_box= }")
        self._check_for_jump(df_box)
        df1 = self.df_check[-lenth:] if lenth is not None else self.df_check
        # self.__logger.info(f"{df1= }")
        # self.__logger.info(f"{df1.y= }")
        # self.__logger.info(f"{df1.y.shape= }")

        fig, axes = plt.subplots(nrows=2, ncols=1)

        # self.smoothed.plot(ax=axes[0], title="Jump detections and vertical position", color="#0088FF")
        # df1.y.plot(ax=axes[0], title="Jump detections", color="#0088FF")
        # if lenth is None or len(df1) >= lenth / 2:
        if True:
            if any(df1.y.notna()):
                df1.y.plot(ax=axes[0], title="Jump detections", color="#0088FF", label="position")
            if any(df1.a.notna()):
                df1.a.plot(ax=axes[0], secondary_y=True, color="#880000", label="accelaration")
                ser = df1.a.copy()  # use copy() → without modifying the original DF
                ser.values.fill(self.acceleration_error - self.earth_gravity)
                ser.plot(secondary_y=True, linestyle=":", color="#dd0000", label="acc_thresh")
            if any(df1.freefall):
                (df1[(df1.freefall)].a).plot(
                    ax=axes[0], secondary_y=True, marker=".", linestyle="None", color="#888800", label="mark_freefall"
                )
            if any(df1.local_maximum):
                (df1[(df1.local_maximum)].y).plot(
                    ax=axes[0], marker=".", linestyle="None", color="#FF00FF", label="mark_localmax"
                )
                (df1.local_maximum.astype(float)).plot(ax=axes[1], color="#FF00FF", label="local_max")
            if any(df1.freefall):
                (-df1.freefall.astype(float)).plot(ax=axes[1], color="#888800", label="freefall")
            if any(df1.high_enough):
                (df1.high_enough.astype(float) * 0.5).plot(ax=axes[1], color="#0088FF", label="high_enough")
            for ax in axes:
                lines = (ax.get_lines() + ax.right_ax.get_lines()) if hasattr(ax, "right_ax") else ax.get_lines()
                ax.legend(handles=lines)
            # axes[0].legend()
            # axes[1].legend()

        fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @property
    def df(self):
        return pd.DataFrame({"box": self._boxes}, index=pd.to_datetime(self._timestamps, unit="s"))

    @property
    def all_df(self):
        return pd.DataFrame({"box": self._all_boxes}, index=pd.to_datetime(self._all_timestamps, unit="s"))

    def dump(self, output_path: Path = Path("./output")):
        self.__logger.debug(f"{self.all_df= }")
        output_path.mkdir(parents=True, exist_ok=True)
        self.all_df.to_pickle(output_path.joinpath("boxes_2.df").as_posix())

        import pickle

        data_to_save = {
            "timestamp": self._all_timestamps,
            "box": self._all_boxes,
        }
        with output_path.joinpath("boxes_2.pk").open(mode="wb") as fp:
            pickle.dump(data_to_save, fp)

    def __del__(self):
        # self.dump()
        pass
