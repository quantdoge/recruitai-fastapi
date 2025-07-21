from pydantic import BaseModel,Field

class User(BaseModel):
    username: str= Field(...,description="The username of the user")
    password: str = Field(..., description="The hashed password of the user")

