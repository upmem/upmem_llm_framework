#
# Copyright (c) 2014-2024 - UPMEM
# UPMEM S.A.S France property - UPMEM confidential information covered by NDA
# For UPMEM partner internal use only - no modification allowed without permission of UPMEM
#
# This file implements multiple hardware architectures to be simulated.
# All architecture inherit from the Base_architecture class.
# If an architecture has optimizations for a given operation defined in Base_architecture,
# define them here

import json
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Dict

import yaml
from jsonschema import validate

from upmem_llm_framework.options import options

def read_architecture_file(file: Path, schema: Dict) -> Dict:
    with open(file, "r", encoding="UTF-8") as f:
        architectures = yaml.safe_load(f)
    validate(architectures, schema)
    return architectures


@cache
def read_architectures() -> Dict:
    """
    Read the architectures from the sim_architectures.yaml file
    :return: a dictionary containing the architectures
    """
    with as_file(files("upmem_llm_framework")) as resources_dir:
        with open(
            resources_dir / "architectures_schema.json", "r", encoding="UTF-8"
        ) as f:
            schema = json.load(f)

        architectures = read_architecture_file(
            resources_dir / "sim_architectures.yaml", schema
        )

        if options.extra_archs:
            extra_architectures = read_architecture_file(
                options.extra_archs, schema
            )
            architectures.update(extra_architectures)

    return architectures


@cache
def get_spec(name: str) -> Dict:
    """
    Get an architecture object corresponding to the given name
    :param name: the name of the architecture
    :return: an object corresponding to the architecture
    """
    architectures = read_architectures()

    architecture_spec = architectures.get(name)
    if architecture_spec is None:
        raise ValueError(f"Architecture {name} not found in sim_architectures.yaml")

    return architecture_spec
