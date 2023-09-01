import argparse
import logging
import time
import matplotlib.pyplot as plt

import cv2
import numpy as np
from video_grabber.image_misc import cv_show_images
from video_grabber.logging import Logger
from video_grabber.option_key_value import option_key_values
from video_grabber.video_grabber import VideoGrabber
from ai_models.blazepose.blazepose_triton_client import Blazepose, Blazepose_Detect, Blazepose_Landmark, option_key
from ai_models.blazepose.parser import args_parser

from sportai_skiprope.jump_detect import JumpCounter, MILLI

__logger = Logger("Track_Person")

# VIDEO_SOURCE = 0  # Camera input
# VIDEO_SOURCE = "rope_jump_3.mp4"  # File input

# BOUNDING_BOX_SCALE_FACTOR = 0.7

# GREEN = (0, 255, 0)
# RED = (0, 0, 255)


def main(_args, _idx=None):
    FLAGS = _args
    # Rendering flags
    option = FLAGS.draw_verbose if FLAGS.draw_verbose is not None else []
    option_key.update(
        {
            "draw_history": "h",
        }
    )
    okv = option_key_values(option_key=option_key, keys=option)
    logging.debug(f"{okv.__str__()= }")

    draw_shape = FLAGS.draw_shape if FLAGS.draw_shape is not None else (400, 400, 3)

    vg = VideoGrabber(video_path=FLAGS.video_source, simulate_fps=False)
    if vg.source == "video_file":
        if FLAGS.begin_ms > 0:
            vg.stream.set(cv2.CAP_PROP_POS_MSEC, FLAGS.begin_ms)
        elif FLAGS.begin_frame > 0:
            vg.stream.set(cv2.CAP_PROP_POS_FRAMES, FLAGS.begin_frame)

    bp_client = Blazepose(
        Blazepose_Detect(score_threshold=0.5, nms_threshold=0.5, best_only=True),
        Blazepose_Landmark(score_threshold=0.5, postprocess_segmentation=True, use_cuda=False),
        force_detection=_args.force_detection,
    )

    jump_counter = JumpCounter()

    print(">>> Start")

    vg.start()
    regions_from_landmarks = []
    start_timestamp = int(time.time())
    while vg.is_alive:
        try:
            frame = vg.get()
        except RuntimeError as e:
            vg.log(f"Exception: {str(e)}", level=logging.DEBUG)
            break
        logging.info(f"{vg.current_image_name= }, {frame.shape= }")

        # Input frame
        src_h, src_w, src_c = frame.shape
        _, _, regions_from_landmarks = bp_client.process(frame, regions_from_landmarks)

        # logging.info(f"    {len(regions_active)= }")
        # logging.debug(f"        {regions_active= }")
        box = np.array(
            [
                [bp_client.regions_active[0].rect_x_center, bp_client.regions_active[0].rect_y_center],
                [bp_client.regions_active[0].rect_w, bp_client.regions_active[0].rect_h],
            ]
        )
        box[0, :] -= box[1, :] / 2
        scale = np.max(frame.shape[:2])
        logging.debug(f"{box= }, {scale= }")
        box = (box * scale).reshape([4]).astype(int)
        logging.debug(f"{box= }")

        # timestamp = vg.current_image_name * MILLI / vg.fps + start_timestamp
        timestamp = vg.current_image_name / vg.fps + start_timestamp
        __logger.debug(f"{timestamp= }")

        # jumps = _get_jump_count(box, jump_counter, timestamp)
        jumps = jump_counter(box, timestamp)

        # if vg.is_file:
        #     timestamp = vg.stream.get(cv2.CAP_PROP_POS_MSEC)
        # else:
        #     timestamp = int(time.time())
        # jump_counter(_bigger_box(box), timestamp)
        if FLAGS.display:
            # vis_frame = _show_frame(frame, box, GREEN, jumps)
            vis_frame = jump_counter.draw_frame(frame)

            # draw_original = okv.value_by_option("draw_original")
            # draw_scores = okv.value_by_option("draw_scores")
            # draw_pd_box = okv.value_by_option("draw_pd_box")
            # draw_pd_kps = okv.value_by_option("draw_pd_kps")
            # draw_rot_rect = okv.value_by_option("draw_rot_rect")
            # draw_landmarks = okv.value_by_option("draw_landmarks")
            draw_fps = okv.value_by_option("draw_fps")
            # draw_segmentation = okv.value_by_option("draw_segmentation")
            vis_frame = bp_client.visualize(
                vis_frame,
                draw_scores=okv.value_by_option("draw_scores"),
                draw_pd_box=okv.value_by_option("draw_pd_box"),
                draw_pd_kps=okv.value_by_option("draw_pd_kps"),
                draw_landmarks=okv.value_by_option("draw_landmarks"),
                draw_segmentation=okv.value_by_option("draw_segmentation"),
                draw_rot_rect=okv.value_by_option("draw_rot_rect"),
            )
            if draw_fps and vg.out_fps is not None:
                cv2.putText(
                    vis_frame,
                    f"FPS={vg.out_fps.get():.1f}",
                    (10, -10 + vis_frame.shape[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (100, 100, 240),
                    2,
                )

            wfs = {}
            wfs.update({"annotated_frame": vis_frame})
            if okv.value_by_option("draw_history"):
                plot_hist = jump_counter.draw_history(lenth=3000)
                wfs.update({"plot_hist": plot_hist})
            if okv.value_by_option("draw_original"):
                wfs.update({"frame": frame})
            # vis_frame = cv2.cvtColor(annotated_pose_frame, cv2.COLOR_RGB2BGR)

            cv_show_images(wfs, draw_shape)
            if FLAGS.interval < 0:
                key = cv2.pollKey()
            else:
                key = cv2.waitKey(FLAGS.interval)

            if key == 27:
                # is_break = True
                break
            elif key == 32:
                # Pause on space bar
                key = cv2.waitKey(0)
                while key != 32:
                    key = cv2.waitKey(0)
            elif key > 0:
                key = chr(key)
                logging.debug(f"{key= }")
                if okv.toggle_by_key(key):
                    logging.debug(f"{okv.kos[key]= },{okv.value_by_key(key)= }")

        # if is_break:
        #     break
        logging.debug(f"    {vg.statistics()}")
    logging.info(f"        {vg.statistics()}")
    jump_counter.dump()

    print("PASS: infer")
    cv2.destroyAllWindows()
    vg.stop()


if __name__ == "__main__":
    FLAGS = args_parser().parse_args()

    # initialize Log
    logging.basicConfig(
        format="[%(process)d,%(thread)x]%(asctime)s -%(levelname)s- %(name)s: %(message)s",
        level=FLAGS.loglevel,
    )
    plt.set_loglevel("info")
    logging.getLogger("PIL").setLevel(logging.INFO)

    # main_loop(FLAGS)
    main(FLAGS)
