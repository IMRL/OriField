#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import torch
import onnxruntime as ort
import time

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.export import TracingAdapter, dump_torchscript_IR
from detectron2.export.flatten import flatten_to_tuple
from detectron2.modeling import build_model
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from model import setup as setup_cfg
from evaluator.evaluator import TrajectoryEvaluator
from dataloader.dataset_mappers.dataset_mapper import DatasetMapper
from common.envs import *


def basic_inputs(inputs):
    return [{
        "bev_map": input["bev_map"],
        "tangent_init": input["tangent_init"],
        "distance_init": input["distance_init"],
    } for input in inputs]


# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    inputs = basic_inputs(inputs)
    traceable_model = TracingAdapter(torch_model, inputs)

    # outputs = torch_model(inputs)
    if args.format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, traceable_model.flattened_inputs)
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, traceable_model.flattened_inputs, f, opset_version=11)
        if str(torch_model.device) == 'cpu':
            ort_session = ort.InferenceSession(os.path.join(args.output, "model.onnx"), providers=['CPUExecutionProvider'])
        else:
            ort_session = ort.InferenceSession(os.path.join(args.output, "model.onnx"), providers=['CUDAExecutionProvider'])
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if args.format == "torchscript":
        def eval_wrapper(inputs):
            """
            The exported model does not contain the final resize step, which is typically
            unused in deployment but needed for evaluation. We add it manually here.
            """
            inputs = basic_inputs(inputs)
            flattened_inputs, _ = flatten_to_tuple(inputs)
            t0 = time.time()
            flattened_outputs = ts_model(*flattened_inputs)
            t1 = time.time()
            print("inferenc time ", t1-t0, "batch", len(inputs))
            new_outputs = traceable_model.outputs_schema(flattened_outputs)
            return new_outputs
    elif args.format == "onnx":
        def eval_wrapper(inputs):
            inputs = basic_inputs(inputs)
            flattened_inputs, _ = flatten_to_tuple(inputs)
            assert len(inputs) == 1, "not support batch inference"
            ort_inputs = {}
            for ort_input, flatten_input in zip(ort_session.get_inputs(), flattened_inputs):
                ort_inputs[ort_input.name] = flatten_input.numpy()
            ort_output = ort_session.run(None, ort_inputs)
            flattened_outputs = [torch.tensor(item) for item in ort_output]
            new_outputs = traceable_model.outputs_schema(flattened_outputs)
            return new_outputs
    else:
        return None
    return eval_wrapper


def get_sample_inputs():
    # get a first batch from dataset
    mapper = DatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    first_batch = next(iter(data_loader))
    return first_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="torchscript",
    )
    parser.add_argument(
        "--export-method",
        choices=["tracing"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--sample-image", default=None, type=str, help="sample image for input")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.output = os.path.join(args.output, "inference") if not MINISET else os.path.join(args.output, "inference.mini")
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable respecialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg(args)

    # create a torch model
    torch_model = build_model(cfg)
    # torch_model.to('cpu')
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    # get sample data
    sample_inputs = get_sample_inputs()

    # convert and save model
    if args.export_method == "tracing":
        exported_model = export_tracing(torch_model, sample_inputs)

    # run evaluation with the converted model
    if args.run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            f"export_method={args.export_method}, format={args.format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        dataset = cfg.DATASETS.TEST[0]
        mapper = DatasetMapper(cfg, is_train=False)
        data_loader = build_detection_test_loader(cfg, dataset, mapper=mapper)
        evaluator = TrajectoryEvaluator(dataset, distributed=True, output_dir=args.output)
        metrics = inference_on_dataset(exported_model, data_loader, evaluator)
        print_csv_format(metrics)
