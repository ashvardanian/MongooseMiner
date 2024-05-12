import subprocess
import argparse
from dataclasses import dataclass
from typing import Optional, Any
import sys
import os
import inspect_sphinx
import inspect
from dill.source import getsource
import pathlib
import pandas as pd


import importlib


@dataclass
class SingleEntry:
    # name of the pip package
    package: str
    # name of the class or method
    name: str
    # docstring of the class or method
    docstring: Optional[str]
    # code of the class or method
    code: Optional[str]
    # signature of the class or method
    signature: Optional[str]


def get_docs_and_code(cls_or_method: Any) -> SingleEntry:
    try:
        doc = inspect_sphinx.getdoc(cls_or_method)
    except:
        doc = None
    try:
        code = getsource(
            cls_or_method, builtin=True, force=True, lstrip=True, enclosing=True
        )
    except:
        code = None

    try:
        signature = str(inspect_sphinx.signature(cls_or_method))
    except:
        signature = None

    return SingleEntry(
        package=cls_or_method.__module__, docstring=doc, code=code, signature=signature, name=cls_or_method.__name__
    )


def test_get_docs_and_code():
    # get docstring and code of the class
    from transformers import LlamaPreTrainedModel

    entry = get_docs_and_code(LlamaPreTrainedModel)
    assert entry.package.startswith("transformers")
    assert entry.docstring == LlamaPreTrainedModel.__doc__


def _recursive_package_list_modules(package: object):
    # get all classes from a package
    for name, obj in inspect.getmembers(package):
        if inspect.isfunction(obj):
            yield obj
        if inspect.isclass(obj):
            yield obj
            for name, obj in inspect.getmembers(obj):
                if inspect.isfunction(obj):
                    yield obj
                
        elif inspect.ismodule(obj):
            try:
                yield from _recursive_package_list_modules(obj.__name__)
            except:
                continue

def get_all_docstrings_and_code(package_name: str) -> list[SingleEntry]:
    # get all classes from a package
    package = importlib.import_module(package_name)
    all_classes = _recursive_package_list_modules(package)

    # get docstrings and code of all classes
    return [get_docs_and_code(cls) for cls in all_classes]


def test_get_all_docstrings_and_code(library_name: str = "transformers"):
    # get all docstrings and code of the transformers package
    entries = get_all_docstrings_and_code("transformers")
    # print some stats
    print(f"Number of classes: {len(entries)}")
    print(
        f"Number of classes with docstrings: {len([entry for entry in entries if entry.docstring])}"
    )
    print(
        f"Number of classes with code: {len([entry for entry in entries if entry.code])}"
    )
    print(
        f"Number of classes with signature: {len([entry for entry in entries if entry.signature])}"
    )
    # check that all classes have a package name starting with 'transformers'
    assert all([entry.package.startswith("transformers") for entry in entries])


def package_to_s3parquet(package_name: str, s3_bucket: Optional[str]):
    """
    gets all docstrings and code of a package and saves it to a parquet file
    pushes the parquet file to s3
    """
    os.environ["VIRTUAL_ENV"] = sys.prefix
    subprocess.check_call(
        ["uv", "pip", "install", package_name]
    )
    # get all docstrings and code of the package
    print(f"installing package {package_name} done.")
    entries = get_all_docstrings_and_code(package_name)
    # convert to a pandas dataframe
    df = pd.DataFrame([entry.__dict__ for entry in entries])

    path = s3_bucket + f"/{package_name}.parquet"

    # save to parquet
    df.to_parquet(path)
    print(f"Exported parquet to {path}")


if __name__ == "__main__":
    # add a command line interface
    parser = argparse.ArgumentParser(description="Get docstrings and code of a package")
    parser.add_argument(
        "--package", type=str, help="Name of the package", default="transformers"
    )
    parser.add_argument(
        "--s3_bucket", type=str, help="Output bucket / base in s3", default=""
    )
    args = parser.parse_args()
    package_to_s3parquet(args.package, args.s3_bucket)
