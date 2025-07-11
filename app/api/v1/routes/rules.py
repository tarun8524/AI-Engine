from fastapi import APIRouter,HTTPException
from typing import List
from app.schemas.rule import RuleOut
from app.core.config import settings

router = APIRouter()

@router.get("/", response_model=List[RuleOut])
async def get_rules():
    """Get all available rules"""
    return settings.RULE_DATA

@router.get("/{rule_id}", response_model=RuleOut)
async def get_rule(rule_id: int):
    """Get specific rule by ID"""
    rule = next((rule for rule in settings.RULE_DATA if rule["id"] == rule_id), None)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return rule
