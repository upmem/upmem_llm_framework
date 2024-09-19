from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated


@dataclass
class Options:
    report_layers: bool = False
    report_functions: bool = False
    print_log: bool = False
    print_log_summary: bool = False
    simulation: bool = False
    sim_compute: bool = False
    sim_data_type: str = "bfloat16"
    sim_num_key_value_heads: int = -1
    sim_sliding_window: int = -1
    sim_verbose: bool = False
    extra_archs: Optional[Path] = None


options = Options()

class DataType(str, Enum):
    int4 = "int4"
    int8 = "int8"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


def initialize_profiling_options(
    report_layers: Annotated[
        bool,
        typer.Option(
            help="Enable reporting metrics for all executed layers at the end of the forward pass."
        ),
    ] = False,
    report_functions: Annotated[
        bool,
        typer.Option(
            help="Enable reporting metrics for all executed functions at the end of the forward "
            + "pass."
        ),
    ] = False,
    print_log: Annotated[
        bool,
        typer.Option(
            help="Print a trace of the execution of layers and functions.",
        ),
    ] = False,
    print_log_summary: Annotated[
        bool,
        typer.Option(
            help="Print a detailed summary of each layer and function executed. For summarization, "
            + "generation, and both.",
        ),
    ] = False,
    simulation: Annotated[
        bool,
        typer.Option(
            help="Enable simulation according to the layer mapping defined",
        ),
    ] = False,
    sim_compute: Annotated[
        bool,
        typer.Option(
            help="Simulate compute intensive operations. Note that some operations are still "
            + "performed due to constraints in inputs/outputs of other layer/functions. "
            + "CAUTION: Output tokens will be affected",
        ),
    ] = False,
    sim_data_type: Annotated[
        DataType,
        typer.Option(
            help="Set the datatype for weights and inputs.",
        ),
    ] = DataType.bfloat16,
    sim_num_key_value_heads: Annotated[
        int,
        typer.Option(
            help="When using GQA, this value is used to simulate fetching the correct KV caches.",
        ),
    ] = -1,
    sim_sliding_window: Annotated[
        int,
        typer.Option(
            help="When set, a sliding window is simulated according to this value. Note that the "
            + "real underlying execution will run according to the model parameter.",
        ),
    ] = -1,
    sim_verbose: Annotated[
        bool,
        typer.Option(
            help="Set a verbose mode for simulation",
        ),
    ] = False,
    extra_archs: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to a yaml file containing extra architectures to be used in simulation",
        ),
    ] = None,
):
    options.report_layers = report_layers
    options.report_functions = report_functions
    options.print_log = print_log
    options.print_log_summary = print_log_summary
    options.simulation = simulation
    options.sim_compute = sim_compute
    options.sim_data_type = sim_data_type
    options.sim_num_key_value_heads = sim_num_key_value_heads
    options.sim_sliding_window = sim_sliding_window
    options.sim_verbose = sim_verbose
    options.extra_archs = extra_archs
