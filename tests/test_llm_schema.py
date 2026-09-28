import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pydantic import ValidationError

from llm import Kinds, Rewrite, response_format, system_prompt

steps = ["шаг 1", "шаг 2", "шаг 3"]

for model in (Kinds, Rewrite):
    model_schema = model.model_json_schema()
    assert list(model_schema["properties"]) == ["reason_steps", "answer"]
    assert model_schema["description"], f"{model.__name__}: нет описания модели"
    for field, spec in model_schema["properties"].items():
        assert spec.get("description"), f"{model.__name__}.{field}: нет description"
    assert model_schema["description"] in system_prompt(model.__name__.lower())

assert Kinds(reason_steps=steps, answer=["Уборка"]).answer == ["Уборка"]
assert Rewrite(reason_steps=steps + ["шаг 4"], answer="клининг квартир").answer == "клининг квартир"

for bad in (
    {"reason_steps": steps[:2], "answer": ["Уборка"]},
    {"reason_steps": steps * 3, "answer": ["Уборка"]},
    {"reason_steps": steps, "answer": ["Космос"]},
    {"reason_steps": steps, "answer": []},
):
    try:
        Kinds.model_validate(bad)
    except ValidationError:
        pass
    else:
        raise AssertionError(f"схема пропустила {bad}")

schema = Kinds.model_json_schema()
assert len(schema["properties"]["answer"]["items"]["enum"]) == 26
assert response_format("kinds")["json_schema"]["schema"] == schema
assert json.dumps(schema, ensure_ascii=False, indent=2) in system_prompt("kinds")
print("ok")
