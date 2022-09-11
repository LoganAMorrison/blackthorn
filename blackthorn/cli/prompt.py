from typing import Tuple, List

import numpy as np
from rich.console import Console
from rich.text import TextType
from rich.prompt import PromptBase, InvalidResponse


class RangedFloatPrompt(PromptBase[float]):
    value_range: Tuple[float, float] = (-np.inf, np.inf)
    response_type: type = float
    validate_error_message = (
        "[prompt.invalid]Please enter float between "
        + f"{value_range[0]} and {value_range[1]}"
    )

    def process_response(self, value: str) -> float:
        low = self.value_range[0]
        high = self.value_range[1]
        self.validate_error_message = (
            f"[prompt.invalid]Please enter integer between {low} and {high}"
        )

        parsed = float(value)
        if not (low < parsed < high):
            raise InvalidResponse(self.validate_error_message)

        return parsed


class RangedIntPrompt(PromptBase[int]):
    value_range: Tuple[int, int] = (-int(1e6), int(1e6))
    response_type: type = int
    validate_error_message = (
        "[prompt.invalid]Please enter integer between "
        + f"{value_range[0]} and {value_range[1]}"
    )

    def process_response(self, value: str) -> float:
        low = self.value_range[0]
        high = self.value_range[1]
        self.validate_error_message = (
            f"[prompt.invalid]Please enter integer between {low} and {high}"
        )

        parsed = int(value)
        if not (low < parsed < high):
            raise InvalidResponse(self.validate_error_message)

        return parsed


class IntListPrompt(PromptBase[List[int]]):
    value_range: Tuple[int, int] = (-int(1e6), int(1e6))
    response_type: type = List[int]
    validate_error_message = (
        "[prompt.invalid]Please enter comma-separated integers between "
        + f"{value_range[0]} and {value_range[1]}"
    )

    def __process_entry(self, value: str) -> int:
        return int(value.strip())

    def process_response(self, response: str) -> List[int]:
        low = self.value_range[0]
        high = self.value_range[1]
        self.validate_error_message = (
            "[prompt.invalid]Please enter comma-separated integers between "
            + f"{low} and {high}"
        )

        values = list(map(self.__process_entry, response.split(",")))
        for value in values:
            if not (low < value < high):
                raise InvalidResponse(self.validate_error_message)

        return values
