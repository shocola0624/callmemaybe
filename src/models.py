from typing import Dict, TypedDict
from pydantic import BaseModel


class Prompt(BaseModel):
    """TODO"""
    prompt: str


class ParameterDef(BaseModel):
    """TODO"""
    type: str


class FunctionDef(BaseModel):
    """TODO"""
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class Output(TypedDict):
    """TODO"""
    prompt: str
    name: str
    parameters: Dict[str, str]
