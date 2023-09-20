"""

"""
import os
import unittest
from unittest import TestCase

from multiprocessing import Pool
import cv2
import numpy as np
import dpirsr_engine as E
import torch

PROJECT_DIR = os.path.dirname(__file__)
LOCAL_DIR = os.path.join(PROJECT_DIR, "local")
RESOURCE_DIR = os.path.join(PROJECT_DIR, "resource")
ONNX_PATH = os.path.join(LOCAL_DIR, "dpirsr-fp32.onnx")
DEVICE_ID = 0
INPUT_SAMPLE_IMAGE_PATH = os.path.join(RESOURCE_DIR, "lena_original.png")
OUTPUT_SAMPLE_IMAGE_PATH = os.path.join(RESOURCE_DIR, "lena_sr.png")
NUMBER_OF_GPU = torch.cuda.device_count()
PROVIDERS = [
    (
        "CUDAExecutionProvider",
        {
            "cudnn_conv_use_max_workspace": "1",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        },
    ),
]
UPSACLE_RATE = 2


def single_inference(args):
    "single_inference"
    input_image, input_test_size, onnx_path, device_id, providers = args
    engine = E.DPIRSREngine(
        onnx_path=onnx_path,
        device_id=device_id,
        input_height=input_test_size[1],
        input_width=input_test_size[0],
        providers=providers,
    )
    engine.set_input_data([np.array([input_image, input_image, input_image])])
    engine.convert_data2input()
    engine.move_host2device()
    engine.inference()
    engine.move_device2host()
    engine.convert_output2data()
    result = engine.get_output_data()[0]
    return result


def multiple_inference(args):
    "multiple_inference"
    (
        input_image,
        input_test_size,
        number_of_test,
        onnx_path,
        device_id,
        providers,
    ) = args

    engine = E.DPIRSREngine(
        onnx_path=onnx_path,
        device_id=device_id,
        input_height=input_test_size[1],
        input_width=input_test_size[0],
        providers=providers,
    )

    result = []
    for _ in range(number_of_test):
        engine.set_input_data([np.array([input_image, input_image, input_image])])
        engine.convert_data2input()
        engine.move_host2device()
        engine.inference()
        engine.move_device2host()
        engine.convert_output2data()
        result.append(engine.get_output_data()[0])
    return result


class EngineTests(TestCase):
    """
    EngineTests
    """

    last_result: bool = False

    def setUp(self) -> None:
        print("\n")
        print(f"TEST::{self._testMethodName}::start")
        print(f" - {self._testMethodDoc}")
        self.last_result = False

    def tearDown(self) -> None:
        result_txt = "success" if self.last_result is True else "fail"
        print(f" - result : {result_txt}")

    def test_00_512_512_single_inference(self):
        """512x512x3 이미지에 대한 단일 inference 실험을 수행합니다."""
        try:
            input_test_size = (512, 512)

            output_test_size = (
                input_test_size[0] * UPSACLE_RATE,
                input_test_size[1] * UPSACLE_RATE,
            )
            input_image = cv2.resize(
                cv2.imread(INPUT_SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR),
                input_test_size,
            )
            output_image = cv2.resize(
                cv2.imread(OUTPUT_SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR),
                output_test_size,
            )
            with Pool(1) as p:
                args = (
                    input_image,
                    input_test_size,
                    ONNX_PATH,
                    DEVICE_ID,
                    PROVIDERS,
                )
                pool_result = p.map(single_inference, [args])[0]
            mse = np.square(
                np.subtract(
                    pool_result.astype(np.float32), output_image.astype(np.float32)
                )
            ).mean()
            print(f" - mse : {mse}")
            self.last_result = True
        except Exception as _e:
            print(f" - error : {_e}")
        self.assertEqual(self.last_result, True)

    def test_01_512_512_multi_inference(self):
        """여러 개의 512x512x3 이미지에 대한 inference 실험을 수행합니다."""
        try:
            number_of_test = 10
            input_test_size = (512, 512)
            input_image = cv2.resize(
                cv2.imread(INPUT_SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR),
                input_test_size,
            )
            with Pool(1) as p:
                args = (
                    input_image,
                    input_test_size,
                    number_of_test,
                    ONNX_PATH,
                    DEVICE_ID,
                    PROVIDERS,
                )
                pool_result = p.map(multiple_inference, [args])[0]
                for idx, img in enumerate(pool_result):
                    cv2.imwrite(
                        os.path.join(
                            LOCAL_DIR, f"test_512_512_multi_inference{idx}.png"
                        ),
                        img,
                    )
            self.last_result = True
        except Exception as _e:
            print(f" - error : {_e}")
        self.assertEqual(self.last_result, True)

    def test_02_video_resolution_input_inference(self):
        """여러 비디오 해상도에 대한 inference 실험을 수행합니다."""
        input_test_sizes = [
            (640, 360),  # SD
            (1280, 720),  # HD
            (1920, 1080),  # FHD
            (2560, 1440),  # 2K
        ]
        done = 0
        for input_test_size in input_test_sizes:
            try:
                _s = False

                input_image = np.random.random_integers(
                    0, 255, (input_test_size[1], input_test_size[0], 3)
                )
                with Pool(1) as p:
                    args = (
                        input_image,
                        input_test_size,
                        ONNX_PATH,
                        DEVICE_ID,
                        PROVIDERS,
                    )
                    _ = p.map(single_inference, [args])[0]
                _s = True
                done += 1
            except Exception as _e:
                print(f" - error : {_e}")
            result_txt = "success" if _s is True else "fail"
            print(f" - test {input_test_size}::{result_txt}")
        self.last_result = done == len(input_test_sizes)
        self.assertEqual(self.last_result, True)

    def test_03_512_512_multi_gpu_inference(self):
        """여러 개의 512x512x3 이미지에 대한 inference 실험을 수행합니다."""
        done = 0
        number_of_gpu = NUMBER_OF_GPU
        print(f" - number of available gpu is {number_of_gpu}")
        if number_of_gpu < 2:
            print(" - skip test because number of available gpu < 2")
            self.last_result = True
            return
        input_test_size = (512, 512)
        input_image = cv2.resize(
            cv2.imread(INPUT_SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR),
            input_test_size,
        )
        args = [
            # [ONNX_PATH, INPUT_SAMPLE_IMAGE_PATH, OUTPUT_SAMPLE_IMAGE_PATH, gpu_id]
            [
                input_image,
                input_test_size,
                ONNX_PATH,
                gpu_id,
                PROVIDERS,
            ]
            for gpu_id in range(number_of_gpu)
        ]
        try:
            with Pool(4) as pool:
                _ = pool.map(single_inference, args)
            self.last_result = True
        except Exception as _e:
            print(f" - error : {_e}")

        self.assertEqual(self.last_result, True)


if __name__ == "__main__":
    unittest.main()
raise Exception()
