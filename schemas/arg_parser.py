from enum import Enum
from typing import TypedDict, Any, Tuple


Long = str
Short = str


class Actions(Enum):
    store = "store"
    count = "count"
    append = "append"
    store_true = "store_true"
    store_const = "store_const"
    store_false = "store_false"
    append_const = "append_const"


class SubCommand(TypedDict):
    name: str
    help: str


class Argument(TypedDict):
    type: Any
    help: str
    flags: Tuple[Long, Short]
    action: Actions
    required: bool


class Arguments(Enum):
    # TODO
    path = Argument(flags=("--path", "-p"), type=str, help="", required=True)
    # TODO
    path2 = Argument(flags=("--path2", "-p2"), type=str, help="", required=True)

    list = Argument(
        flags=("--list", "-l"),
        action=Actions.store_true.value,
    )
    color = Argument(flags=("--color", "-c"), type=str, help="")
    output = Argument(flags=("--output", "-o"), type=str, help="", required=True)
    threshold = Argument(flags=("--threshold", "-t"), type=int, help="")
