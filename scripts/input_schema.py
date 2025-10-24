from pydantic import BaseModel
from datetime import datetime

class InputSchema(BaseModel):
    type: str
    available_from: datetime
    payment_method: str
    governorate: str
    city: str
    district: str
    compound: str
    size_sqm: float
    bedrooms_num: int
    has_maid_room: bool
    bathrooms_num: int
