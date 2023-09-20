"""DPIR Engine Module"""
import os
from typing import Any, List, Optional
import torch
import onnxruntime as ort  # ORT 선언 전 torch import 필수
import numpy as np
from ortei import IORTEngine


__all__ = ["DPIRSREngine"]


class DPIRSREngine(IORTEngine):
    """DPIRSREngine"""

    def __init__(
        self,
        onnx_path: str,
        device_id: int,
        device_name: str = "cuda",
        model_batch=1,
        input_height=1080,
        input_width=1920,
        upscale_rate: int = 2,
        providers: Optional[list] = None,
    ) -> None:
        self.upscale_rate = upscale_rate
        print(f"DPIRSREngine[{device_id}]::Init::Start")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"

        if device_name == "cuda":
            assert torch.cuda.is_available(), "ERROR::CUDA not available."

        self.onnx_path = onnx_path
        self.device_name = device_name
        self.device_id = device_id
        self.providers = providers
        self.io_shape = {
            "input_data": [model_batch, 3 * 3, input_height, input_width],
            "output_data": [
                model_batch,
                3,
                input_height * upscale_rate,
                input_width * upscale_rate,
            ],
        }

        # ort.set_default_logger_severity(0) # Turn on verbose mode for ORT TRT
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        if providers is None:
            self.providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_int8_enable": False,
                        "trt_int8_use_native_calibration_table": False,
                        "trt_engine_cache_enable": False,
                        "trt_int8_calibration_table_name": "calibration.flatbuffers",
                        "trt_engine_cache_path": "./local/",
                        "trt_max_workspace_size": 2147483648,
                    },
                ),
                (
                    "CUDAExecutionProvider",
                    {
                        "cudnn_conv_use_max_workspace": "1",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",  # HEURISTIC',
                        "do_copy_in_default_stream": True,
                    },
                ),
            ]

        for _prov in self.providers:
            _pname, _poption = _prov
            assert all(
                [_key != "device_id" for _key in list(_poption.keys())]
            ), "provider option - device_id is not supported."

        # init
        super().__init__()
        self._bind_model_io()

        # warm-up
        self.inference()
        print(f"DPIREngine[{device_id}]::Init::End")

    def _init_members(self):
        # set variable
        onnx_fp32_path = self.onnx_path
        device_name = self.device_name
        device_id = self.device_id
        providers = self.providers

        # set session options
        self.session_options = ort.SessionOptions()
        self.session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        self.session_options.enable_profiling = False
        self.session_options.log_severity_level = 3
        self.providers = providers

        # build engine
        self.ort_session = ort.InferenceSession(
            path_or_bytes=onnx_fp32_path,
            sess_options=self.session_options,
            providers=providers,
        )

        # set binding data
        self.io_binding = self.ort_session.io_binding()
        self.io_data_cpu = {
            key: np.zeros(self.io_shape[key], np.float32) for key in self.io_shape
        }
        self.io_data_ort = {
            key: ort.OrtValue.ortvalue_from_numpy(
                self.io_data_cpu[key],
                device_name,
                device_id=0,  # device_id 설정시 오류 발생
            )
            for key in self.io_data_cpu
        }

        # set input, output
        _b, _c, _h, _w = self.io_shape["input_data"]
        self.input_data = np.zeros([_b, 3, _h, _w, 3], np.uint8)
        _b, _c, _h, _w = self.io_shape["output_data"]
        self.output_data = np.zeros([_b, _h, _w, _c], np.uint8)
        self.output_shape = self.output_data.shape

    def _bind_model_io(self) -> None:
        io_binding = self.io_binding
        io_data_ort = self.io_data_ort

        # binding
        io_binding.bind_ortvalue_input("input_data", io_data_ort["input_data"])
        io_binding.bind_ortvalue_output("output_data", io_data_ort["output_data"])

    def get_output_data(self) -> Any:
        _b, _h, _w, _c = self.output_shape
        return self.output_data[:_b, :_h, :_w, :_c]

    @property
    def maximum_size(self):
        """
        maximum size to input
        """
        _b, _c, _h, _w = self.io_shape["input_data"]
        return (_w, _h)

    def set_input_data(self, data: List[np.ndarray]) -> None:
        """
        data : images(list of np.ndarrays)
        """
        data = np.array(data)
        _b, _n, _h, _w, _c = data.shape
        _mxw, _mxh = self.maximum_size
        assert _b == self.io_shape["input_data"][0]
        assert _mxh >= _h
        assert _mxw >= _w
        self.output_shape = (_b, _h * self.upscale_rate, _w * self.upscale_rate, _c)
        self.input_data.fill(0.0)
        self.input_data[:, :_n, :_h, :_w, :_c] = data[:]

    def convert_data2input(self) -> None:
        data = np.transpose(self.input_data / 255.0, (0, 1, 4, 2, 3))
        _b, _n, _c, _h, _w = data.shape
        data = data.reshape((_b, _n * _c, _h, _w))
        io_data_cpu = self.io_data_cpu

        # manual work required.
        io_data_cpu["input_data"][:] = data[:]

    def move_host2device(self) -> None:
        # init
        io_data_cpu: dict = self.io_data_cpu
        io_data_ort: dict = self.io_data_ort

        # manual work required.
        io_data_ort["input_data"].update_inplace(io_data_cpu["input_data"])

    def inference(self):
        # init
        ort_session = self.ort_session
        io_binding: dict = self.io_binding

        # inference
        ort_session.run_with_iobinding(io_binding)

    def move_device2host(self):
        # init
        io_data_ort: dict = self.io_data_ort
        io_data_cpu: dict = self.io_data_cpu

        # manual work required.
        io_data_cpu["output_data"][:] = io_data_ort["output_data"].numpy()[:]

    def convert_output2data(self):
        # init
        io_data_cpu: dict = self.io_data_cpu
        result_images: np.ndarray = self.output_data

        # manual work required.
        output = np.clip(np.multiply(io_data_cpu["output_data"], 255.0), 0, 255).astype(
            np.uint8
        )
        for _c in range(3):
            result_images[:, :, :, _c] = output[:, _c, :, :]
