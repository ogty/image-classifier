from typing import Any

from schemas.arg_parser import Arguments


def set_arguments(subcommand: Any, arguments: Arguments) -> Any:
    for argument in arguments:
        flags = argument.value.get("flags")
        if "action" in argument.value:
            subcommand.add_argument(*flags, action=argument.value.get("action"))
            continue
        other = {
            "type": argument.value.get("type"),
            "help": argument.value.get("help"),
            "required": argument.value.get("required"),
        }
        subcommand.add_argument(*flags, **other)
    return subcommand
