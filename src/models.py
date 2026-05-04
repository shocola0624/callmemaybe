from typing import Dict
from pydantic import BaseModel


class Prompt(BaseModel):
    """Single user input text."""
    prompt: str


class ParameterDef(BaseModel):
    """Expected type of a function parameter."""
    type: str


class FunctionDef(BaseModel):
    """Function's name, description, and signature."""
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class Output(BaseModel):
    """Prompt and its extracted function call results."""
    prompt: str
    name: str
    parameters: Dict[str, str]
