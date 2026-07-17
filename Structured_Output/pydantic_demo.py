from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='a decimal value representing the cgpa o the student')

new_student = {'name':'aashi', 'age': '35', 'email':'abc@gmail.com'}

# type Coercing is bydefault in pydantic
# field function -> constraints, default value & custom desciption(annotated for llm)

student = Student(**new_student)

print(student)