from __future__ import annotations
from pydantic import BaseModel, Field


class BaseRequestValidationModel(BaseModel):
    reason: str = Field(description="Reason for the answer.")


class BooleanRequestValidationModel(BaseRequestValidationModel):
    answer: bool = Field(description="The answer.")


class StringRequestValidationModel(BaseRequestValidationModel):
    answer: str = Field(description="The answer.")


class RequestValidationModel(BaseModel):
    is_request_completed: BooleanRequestValidationModel = Field(
        description="Has enough of the plan been executed to successfully complete the original user request? This includes the execution of planned tasks, and the provision of all requested information.")

    is_in_loop: BooleanRequestValidationModel = Field(
        description="Are we in a loop where we are repeating the same requests and/or getting the same responses? Loops can span multiple turns, and can include repeated actions.")

    next_instruction_or_question: StringRequestValidationModel = Field(
        description="What is the next instruction or question to make progress on the request? Phrase it as if the user is asking the system to perform the action, e.g. 'Please provide the weather forecast for tomorrow.'")
