"""
Base schema definitions for the database models.
"""
from pydantic import BaseModel as PydanticBaseModel, ConfigDict

class BaseSchema(PydanticBaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
        populate_by_name=True,
    )
