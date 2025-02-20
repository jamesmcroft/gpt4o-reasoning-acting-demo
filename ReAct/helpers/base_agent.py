from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from typing import Callable, List, Any
import inspect
from pydantic import create_model


def _remove_schema_titles(schema: dict) -> dict:
    if isinstance(schema, dict):
        schema.pop("title", None)
        for key, value in schema.items():
            schema[key] = _remove_schema_titles(value)
    elif isinstance(schema, list):
        schema = [_remove_schema_titles(item) for item in schema]
    return schema


def skill(func: Callable) -> Callable:
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Function {func.__name__} has invalid signature: {e}")

    fields = {}
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        annotation = param.annotation if param.annotation != inspect._empty else str
        default = ... if param.default == inspect._empty else param.default
        fields[param.name] = (annotation, default)

    # Dynamically create a Pydantic model for function parameters
    Model = create_model(f"{func.__name__.title()}Model", **fields)

    schema = _remove_schema_titles(Model.model_json_schema())

    # Ensure the skill description is stripped and formatted properly
    desc = func.__doc__ or ""
    desc = desc.strip().replace("\n", " ")

    func.__skill_schema__ = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": desc,
            "parameters": schema,
        },
    }
    func.__is_skill__ = True
    return func


class BaseAgent:
    def __init__(self, name: str, description: str, client: OpenAI, model_deployment: str):
        self.name = name
        self.description = description
        self.client = client
        self.model_deployment = model_deployment
        self.skills = []

        # Register function skills
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, "__is_skill__", False):
                self.skills.append(attr.__skill_schema__)

    def get_agent_details(self) -> str:
        if len(self.skills) == 0:
            return f"- Name: {self.name}\n- Description: {self.description}"

        skills = "\n".join(
            [f"  - {skill['function']['name']}: {skill['function']['description']}" for skill in self.skills])

        return f"- Name: {self.name}\n- Description: {self.description}\n- Skills:\n{skills}"

    def call_function(self, function_name: str, **kwargs) -> Any:
        return getattr(self, function_name)(**kwargs)

    def process_query(self, messages: List[str]) -> ChatCompletionMessage:
        completion = self.client.chat.completions.create(
            model=self.model_deployment,
            messages=messages,
            temperature=0.3,
            top_p=0.3,
        )

        return completion.choices[0].message
