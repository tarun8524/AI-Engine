from pydantic import BaseModel

class RuleOut(BaseModel):
    id: int
    rule: str